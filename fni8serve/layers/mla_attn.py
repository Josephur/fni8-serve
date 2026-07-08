# SPDX-License-Identifier: MIT
"""Multi-head Latent Attention (MLA) — DeepSeek-V2/V3/V4 — Track-1 fp16 port.

Ports the DeepSeek `decompress` path: down-project the hidden to compressed latents
c_Q (q_lora_rank) and c_KV (kv_lora_rank), up-project to per-head Q/K/V, apply
decoupled RoPE to a separate small key dim (qk_rope_head_dim, shared across heads),
then standard causal softmax attention. The LoRA projections run on the fni8 dp4a
GEMM (LinearW8A8); the attention itself stays fp16 here because MLA's query/key head
dim (qk_nope+qk_rope, e.g. 192) is not one of fni8's supported attention head dims
and its value dim differs from the key dim.

Track 2 (fni8 csrc/) replaces the fp16 attention with the ABSORB-path int8 kernel
(fold W_UK into Q, W_UV into W_O; MQA against the shared 512-d latent). This module
is the correct oracle for that kernel. `absorb_qk_equiv` proves the identity the
kernel relies on: q_nope·k_nope == (W_UK^T q_nope)·c_KV.

KV cache stores only c_KV (kv_lora_rank) + k_pe (qk_rope_head_dim) per token — see
`models.cache.MLALatentCache`. Decode re-projects the *entire* cached latent through
kv_b_proj every step (no weight absorption yet), so it costs O(N) extra GEMM work
per step; that's exactly the naive path the Track-2 absorb kernel replaces.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fni8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm
from .rotary import RotaryEmbedding


def absorb_qk_equiv(q_nope, w_uk, c_kv) -> torch.Tensor:
    """Identity check helper: the nope score q_nope·k_nope equals
    (W_UK^T q_nope)·c_KV, so attention can run against the latent directly.
    q_nope [.., nope], w_uk [nope, kv_lora], c_kv [.., kv_lora]."""
    q_absorbed = q_nope @ w_uk                         # -> [.., kv_lora]
    return (q_absorbed * c_kv).sum(-1)


class MLAAttention(nn.Module):
    def __init__(self, cfg, *, q_a_proj: QTensor, q_a_norm, q_b_proj: QTensor,
                 kv_a_proj: QTensor, kv_a_norm, kv_b_proj: QTensor, o_proj: QTensor,
                 num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                 qk_rope_head_dim, v_head_dim, rope_theta=1e4, max_pos=8192):
        super().__init__()
        self.nh = num_heads
        self.qk_nope, self.qk_rope, self.vd = qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        self.kv_lora = kv_lora_rank
        self.qk_head = qk_nope_head_dim + qk_rope_head_dim
        self.scale = self.qk_head ** -0.5
        self.q_a_proj = LinearW8A8(q_a_proj)
        self.q_a_norm = RMSNorm(q_lora_rank, cfg.rms_norm_eps, q_a_norm)
        self.q_b_proj = LinearW8A8(q_b_proj)
        self.kv_a_proj = LinearW8A8(kv_a_proj)        # -> kv_lora + qk_rope (MQA k_pe)
        self.kv_a_norm = RMSNorm(kv_lora_rank, cfg.rms_norm_eps, kv_a_norm)
        self.kv_b_proj = LinearW8A8(kv_b_proj)        # -> nh*(qk_nope + v)
        self.o_proj = LinearW8A8(o_proj)
        self.rope = RotaryEmbedding(qk_rope_head_dim, max_pos, base=rope_theta)

    def _up_kv(self, c_kv: torch.Tensor):
        """c_kv [B,N,kv_lora] -> k_nope [B,N,nh,qk_nope], v [B,N,nh,vd] (the up-project
        half of the decompress path; called on the whole cache every decode step since
        weights aren't absorbed here — see Track 2)."""
        B, N, _ = c_kv.shape
        kv = self.kv_b_proj(c_kv).view(B, N, self.nh, self.qk_nope + self.vd)
        return kv.split([self.qk_nope, self.vd], dim=-1)

    def forward(self, hidden, positions, ctx, layer_idx):
        B, L, _ = hidden.shape
        if positions is None:
            positions = torch.arange(L, device=hidden.device).unsqueeze(0).expand(B, L)
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden))).view(B, L, self.nh, self.qk_head)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        kv_mqa = self.kv_a_proj(hidden)
        c_kv, k_pe = kv_mqa.split([self.kv_lora, self.qk_rope], dim=-1)
        c_kv = self.kv_a_norm(c_kv)
        k_pe = k_pe.view(B, L, 1, self.qk_rope)

        # decoupled RoPE on the rope part (k_pe shared across heads, cached post-RoPE)
        q_rope, k_pe = self.rope(positions, q_rope, k_pe)

        causal = True
        if ctx is not None and ctx.kv_cache is not None:
            latent = torch.cat([c_kv, k_pe.squeeze(2)], dim=-1)   # [B,L,kv_lora+qk_rope]
            if ctx.is_prefill:
                ctx.kv_cache.write_prefill(layer_idx, latent)
                k_nope, v = self._up_kv(c_kv)
                k_pe_all = k_pe.expand(B, L, self.nh, self.qk_rope)
            else:
                latent_all = ctx.kv_cache.append_decode(layer_idx, latent)  # [B,N,kv_lora+qk_rope]
                c_kv_all, k_pe_all = latent_all.split([self.kv_lora, self.qk_rope], dim=-1)
                N = latent_all.shape[1]
                k_nope, v = self._up_kv(c_kv_all)
                k_pe_all = k_pe_all.view(B, N, 1, self.qk_rope).expand(B, N, self.nh, self.qk_rope)
                causal = False   # single query attends to all (already-causal) cached keys
        else:
            k_nope, v = self._up_kv(c_kv)
            k_pe_all = k_pe.expand(B, L, self.nh, self.qk_rope)

        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)          # [B,nh,L,qk_head]
        k = torch.cat([k_nope, k_pe_all], dim=-1).transpose(1, 2)        # [B,nh,N,qk_head]
        v = v.transpose(1, 2)                                            # [B,nh,N,vd]

        scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * self.scale
        if causal:
            Lq, Nk = scores.shape[-2], scores.shape[-1]
            mask = torch.ones(Lq, Nk, device=hidden.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        p = torch.softmax(scores, dim=-1).to(v.dtype)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v)                        # [B,nh,L,vd]
        return self.o_proj(o.transpose(1, 2).reshape(B, L, self.nh * self.vd))
