# SPDX-License-Identifier: MIT
"""Contiguous KV cache (v0) — one fp16 [L, B, Hkv, max_len, D] store per model.

This is the simple, correct cache: prefill writes the whole sequence, decode
appends one token. It is NOT paged — paged block-table decode needs the fni8
decode kernel to accept block tables (a kernel TODO). The engine's block manager
(PR3) swaps this out for paged blocks; the model code is unchanged because it only
touches `write_prefill` / `append_decode`.

`length` is the number of valid tokens; the runner calls `advance(n)` once per step
after all layers have written (all layers share one length).
"""
from __future__ import annotations

import torch


class KVCache:
    def __init__(self, num_layers, batch, num_kv_heads, max_len, head_dim,
                 *, device, dtype=torch.float16):
        shape = (num_layers, batch, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def reset(self):
        self.length = 0

    def write_prefill(self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot=None):
        """k, v: [B, Hkv, S, D] at positions [0, S). `slot` is ignored (the simple
        runner cache is single-batch); the engine's BatchedKVCache uses it."""
        s = k.shape[2]
        self.k[layer, :, :, :s] = k
        self.v[layer, :, :, :s] = v

    def append_decode(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [B, Hkv, 1, D] at position `length`. Returns the [B,Hkv,length+1,D]
        cache slice to attend against."""
        p = self.length
        self.k[layer, :, :, p:p + 1] = k
        self.v[layer, :, :, p:p + 1] = v
        return self.k[layer, :, :, :p + 1], self.v[layer, :, :, :p + 1]

    def advance(self, n: int = 1):
        self.length += n


class RecurrentStateCache:
    """Per-layer decode-time state for linear-attention layers (Gated DeltaNet,
    lightning attention): the recurrent state `S` and, for DeltaNet's causal
    depthwise conv, the trailing `kernel-1` raw window feeding it. Keyed by layer
    index; only linear-attention layers touch this (full/sliding/latent layers use
    `KVCache` instead). Prefill starts each layer from scratch (`get_*` returns None);
    decode carries the previous call's state back in so the recurrence doesn't reset
    every step."""

    def __init__(self):
        self._state: dict[int, torch.Tensor] = {}
        self._conv_tail: dict[int, torch.Tensor] = {}

    def get_state(self, layer_idx: int) -> torch.Tensor | None:
        return self._state.get(layer_idx)

    def set_state(self, layer_idx: int, state: torch.Tensor):
        self._state[layer_idx] = state

    def get_conv_tail(self, layer_idx: int) -> torch.Tensor | None:
        return self._conv_tail.get(layer_idx)

    def set_conv_tail(self, layer_idx: int, tail: torch.Tensor):
        self._conv_tail[layer_idx] = tail

    def reset(self):
        self._state.clear()
        self._conv_tail.clear()
