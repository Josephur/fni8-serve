# SPDX-License-Identifier: MIT
"""GQAAttention — the shared full/sliding softmax-attention block on fni8 dp4a.

Config-driven so Qwen3, Qwen3-MoE, Gemma3, GLM, Hunyuan all reuse it unchanged:
  * merged QKV projection (int8 dp4a), split to GQA heads;
  * optional per-head RMSNorm on Q and K over head_dim, applied BEFORE RoPE
    (Qwen3 / Gemma3 QK-norm);
  * RoPE with per-layer theta (Gemma3 local vs global) and partial-rotary (GLM);
  * softmax scale = query_pre_attn_scalar**-0.5 (Gemma) or head_dim**-0.5;
  * sliding-window local layers via fni8's window_left (prefill) / cache slice (decode).

This is the `full`/`sliding` AttentionBackend. `linear` (DeltaNet) and `latent`
(MLA) backends are separate — see fni8serve/models/backends.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import fni8
from fni8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm
from .rotary import RotaryEmbedding


class GQAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_proj: QTensor,
        o_proj: QTensor,
        scale: float,
        rope: RotaryEmbedding,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
        rms_norm_eps: float = 1e-6,
        window_left: int = -1,
        qkv_bias=None,
        o_bias=None,
        causal: bool = True,
    ):
        super().__init__()
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, head_dim
        self.scale = scale
        self.window_left = window_left
        self.causal = causal
        self.qkv_proj = LinearW8A8(qkv_proj, qkv_bias)
        self.o_proj = LinearW8A8(o_proj, o_bias)
        self.rope = rope
        self.q_norm = RMSNorm(head_dim, rms_norm_eps, q_norm) if q_norm is not None else None
        self.k_norm = RMSNorm(head_dim, rms_norm_eps, k_norm) if k_norm is not None else None

    def forward(self, x, positions, ctx, layer_idx: int) -> torch.Tensor:
        """x: [B, S, hidden]; positions: [B, S] or [S]. Returns [B, S, hidden]."""
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(B, S, self.nh, self.hd)
        k = k.view(B, S, self.nkv, self.hd)
        v = v.view(B, S, self.nkv, self.hd)
        if self.q_norm is not None:                 # per-head RMSNorm, pre-RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(positions, q, k)
        # -> [B, H, S, D] for the fni8 kernels
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if ctx.is_prefill:
            slot = ctx.slots[0] if ctx.slots is not None else None
            ctx.kv_cache.write_prefill(layer_idx, k, v, slot=slot)
            out = fni8.attn_int8_fwd(q, k, v, causal=self.causal, scale=self.scale,
                                      window_left=self.window_left)
        elif ctx.slot_lengths is not None or ctx.slot_mapping is not None:
            out = self._decode_batched(q, k, v, ctx, layer_idx)   # engine continuous batch
                                                                    # (or CUDA-graph static path)
        else:
            k_all, v_all = ctx.kv_cache.append_decode(layer_idx, k, v)   # [B,Hkv,N,D]
            k_all, v_all = self._window(k_all, v_all)
            out = fni8.attn_int8_decode(q, k_all, v_all, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, S, self.nh * self.hd)
        return self.o_proj(out)

    def _window(self, k_all, v_all):
        if self.window_left >= 0 and k_all.shape[2] > self.window_left:
            k_all = k_all[:, :, -self.window_left:].contiguous()
            v_all = v_all[:, :, -self.window_left:].contiguous()
        return k_all, v_all

    def _decode_batched(self, q, k, v, ctx, layer_idx):
        """Continuous-batch decode: rows have different KV lengths (already batched
        GEMMs upstream). The paged int8 cache commits every row's new token with ONE
        `quantize_kv_write_paged` call and reads the whole ragged batch back with ONE
        `attn_paged_decode_cached` launch -- no more per-slot Python loop.

        Sliding-window layers are the one gap the paged-decode kernel doesn't cover
        (no window parameter yet), so they fall back to a per-slot dequantized read
        + `attn_int8_decode`, same as before this PR."""
        cache = ctx.kv_cache
        k_new, v_new = k[:, :, 0, :], v[:, :, 0, :]           # [B,Hkv,D]: the one new token
        if self.window_left < 0:
            if ctx.slot_mapping is not None:
                # CUDA-graph decode (engine/cuda_graph.py): slot_mapping/block_tables/
                # context_lens are persistent device buffers refreshed via `copy_`
                # before replay, and max_context_len is a compile-time bucket int --
                # no fresh per-layer tensor allocation and no `.item()` sync, so this
                # whole call is capturable.
                cache.write_decode_static(layer_idx, ctx.slot_mapping, k_new, v_new)
                return cache.decode_attn_static(layer_idx, q, ctx.block_tables, ctx.context_lens,
                                                ctx.max_context_len, scale=self.scale)
            cache.write_decode(layer_idx, ctx.slots, ctx.slot_lengths, k_new, v_new)
            return cache.decode_attn(layer_idx, q, ctx.slots, ctx.slot_lengths, scale=self.scale)
        outs = []
        for b, (slot, n) in enumerate(zip(ctx.slots, ctx.slot_lengths)):
            cache.write_decode(layer_idx, [slot], [n], k_new[b:b + 1], v_new[b:b + 1])
            kb, vb = cache.read_dense(layer_idx, slot, n + 1, window=self.window_left)
            outs.append(fni8.attn_int8_decode(q[b:b + 1], kb, vb, scale=self.scale))
        return torch.cat(outs, dim=0)
