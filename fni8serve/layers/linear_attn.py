# SPDX-License-Identifier: MIT
"""Gated DeltaNet linear attention (Qwen3-Next / Qwen3.5/3.6 backbone, MiniMax
lightning) — Track-1 fp16 port.

Recurrence (Gated DeltaNet, Yang et al. ICLR'25), per head:

    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

with q,k L2-normalized per head, a short causal depthwise conv1d(k=4)+SiLU token
shift on q/k/v, a scalar decay gate alpha_t = exp(-softplus(dt)*exp(A_log)) and a
write gate beta_t = sigmoid(b_t), then a gated RMSNorm on the output. The linear
projections run on the fni8 dp4a GEMM (LinearW8A8); only the recurrence stays fp16.

This module is the CORRECT fp path (the numeric oracle). It uses the O(L) scalar
recurrence — unambiguous and exact. The chunked WY/UT parallel form + its int8
dp4a acceleration are Track 2 (fni8 csrc/): the chunk form needs the log-space
stabilization real kernels use, so it lives with that kernel, not here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fni8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def recurrent_gated_delta_rule(q, k, v, beta, g) -> torch.Tensor:
    """Gated delta rule, scalar form. q,k: [B,H,L,Dk] (pre-L2-normed), v: [B,H,L,Dv],
    beta: [B,H,L] in (0,1], g: [B,H,L] = log(alpha) (<=0). Returns o [B,H,L,Dv]."""
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=q.device)
    alpha = g.exp().float()
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    out = torch.empty(B, H, L, Dv, dtype=torch.float32, device=q.device)
    for t in range(L):
        kt, vt, qt = kf[:, :, t], vf[:, :, t], qf[:, :, t]        # [B,H,D]
        at = alpha[:, :, t][..., None, None]                     # [B,H,1,1]
        bt = bf[:, :, t][..., None]                              # [B,H,1]
        Sk = torch.einsum("bhvk,bhk->bhv", S, kt)                # S_{t-1} k_t
        erase = at * (bt * Sk)[..., None] * kt[..., None, :]
        write = (bt * vt)[..., None] * kt[..., None, :]
        S = at * S - erase + write
        out[:, :, t] = torch.einsum("bhvk,bhk->bhv", S, qt)      # o_t = S_t q_t
    return out.to(v.dtype)


class GatedDeltaNetAttention(nn.Module):
    """Linear-attention block. Projections on dp4a; recurrence fp16. GQA-style:
    `num_v_heads` value/output heads, `num_k_heads` query/key heads (broadcast).

    Weights: qkv_proj (merged Q|K|V), beta_proj + a/dt gate params (A_log, dt_bias),
    conv weight [width, kernel], out_proj, and the gated output RMSNorm gain.
    """

    def __init__(self, cfg, *, qkv_proj: QTensor, out_proj: QTensor,
                 conv_weight, a_log, dt_bias, beta_proj, gate_proj,
                 norm_gain, num_k_heads, num_v_heads, key_dim, value_dim,
                 conv_kernel=4):
        super().__init__()
        self.nk, self.nv = num_k_heads, num_v_heads
        self.kd, self.vd = key_dim, value_dim
        self.conv_kernel = conv_kernel
        self.qkv_proj = LinearW8A8(qkv_proj)
        self.out_proj = LinearW8A8(out_proj)
        self.beta_proj = LinearW8A8(beta_proj)
        self.gate_proj = LinearW8A8(gate_proj)
        self.register_buffer("conv_weight", conv_weight, persistent=False)   # [Wc, K]
        self.A_log = nn.Parameter(a_log)
        self.dt_bias = nn.Parameter(dt_bias)
        self.norm = RMSNorm(value_dim, cfg.rms_norm_eps, norm_gain)

    def _conv(self, x):
        """Causal depthwise conv1d(k) + SiLU. x: [B, L, Wc]."""
        B, L, W = x.shape
        xt = x.transpose(1, 2)                                    # [B,Wc,L]
        xt = F.pad(xt, (self.conv_kernel - 1, 0))
        xt = F.conv1d(xt, self.conv_weight.unsqueeze(1), groups=W)
        return F.silu(xt.transpose(1, 2))

    def forward(self, hidden, positions, ctx, layer_idx):
        B, L, _ = hidden.shape
        qkv = self._conv(self.qkv_proj(hidden))
        qk = self.nk * self.kd
        q, k, v = qkv.split([qk, qk, self.nv * self.vd], dim=-1)
        q = _l2norm(q.view(B, L, self.nk, self.kd)).transpose(1, 2)
        k = _l2norm(k.view(B, L, self.nk, self.kd)).transpose(1, 2)
        v = v.view(B, L, self.nv, self.vd).transpose(1, 2)
        # broadcast k/q heads to value heads (GQA)
        rep = self.nv // self.nk
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
        beta = torch.sigmoid(self.beta_proj(hidden)).transpose(1, 2)         # [B,nv,L]...
        beta = beta.reshape(B, self.nv, L) if beta.dim() == 3 else beta
        dt = self.gate_proj(hidden).transpose(1, 2)
        g = -F.softplus(dt.float() + self.dt_bias.view(1, -1, 1)) * self.A_log.exp().view(1, -1, 1)
        o = recurrent_gated_delta_rule(q, k, v, beta.reshape(B, self.nv, L), g.reshape(B, self.nv, L))
        o = self.norm(o.transpose(1, 2).reshape(B, L, self.nv * self.vd))
        return self.out_proj(o)
