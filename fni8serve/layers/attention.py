# SPDX-License-Identifier: MIT
"""Attention seam — routes to the `fni8` int8 dp4a kernels.

Replaces nano-vllm's fp16 flash-attention:
  * prefill  -> fni8.attn_int8_fwd (causal int8 dp4a; varlen path is fni8.attn_int8_varlen)
  * decode   -> fni8.attn_int8_decode (split-KV) / attn_decode_cached (int8 KV cache)

TODO (needs kernel work in fni8):
  * paged-KV (block-table) support in the decode kernel so we read nano-vllm's paged
    cache directly instead of a contiguous [B,Hkv,N,D] gather;
  * quantize-on-write int8 KV store.
Until then this seam uses the contiguous-cache path (correct; not yet paged).
"""
from __future__ import annotations

import torch
import torch.nn as nn

import fni8


class Fni8Attention(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, *,
                 scale: float | None = None, kv_cache_dtype: str = "int8"):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or head_dim ** -0.5
        self.kv_cache_dtype = kv_cache_dtype

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                *, is_prefill: bool) -> torch.Tensor:
        """q,k,v: [B, H, S, D] fp16. Prefill = causal over the sequence; decode = M=1
        against the cache (here k/v are the full cache slice)."""
        if is_prefill:
            return fni8.attn_int8_fwd(q, k, v, causal=True, scale=self.scale)
        # decode: one query row against the cached K/V.
        if self.kv_cache_dtype == "int8":
            k_i8, k_scale, v_i8, v_scale = fni8.quantize_kv_cache(k, v, rotate=False)
            return fni8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale, scale=self.scale)
        return fni8.attn_int8_decode(q, k, v, scale=self.scale)
