# SPDX-License-Identifier: MIT
"""Contiguous KV cache (v0) — one fp16 [L, B, Hkv, max_len, D] store per model.

This is the simple, correct cache used by the standalone `ModelRunner`: prefill
writes the whole sequence, decode appends one token. The engine uses
`fni8serve.engine.kv_cache.PagedKVCache` instead (int8, block-table paged); the
model code is unchanged either way because it only touches `write_prefill` /
`append_decode`.

`length` is the number of valid tokens; the runner calls `advance(n)` once per step
after all layers have written (all layers share one length).
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self, num_layers, batch, num_kv_heads, max_len, head_dim, *, device, dtype=torch.float16
    ):
        shape = (num_layers, batch, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def reset(self):
        self.length = 0

    def write_prefill(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot=None, start: int = 0
    ):
        """k, v: [B, Hkv, S, D] at positions [0, S). `slot` is ignored (the simple
        runner cache is single-batch); the engine's PagedKVCache uses it."""
        s = k.shape[2]
        self.k[layer, :, :, start:s] = k[:, :, start:, :]
        self.v[layer, :, :, start:s] = v[:, :, start:, :]

    def append_decode(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [B, Hkv, 1, D] at position `length`. Returns the [B,Hkv,length+1,D]
        cache slice to attend against."""
        p = self.length
        self.k[layer, :, :, p : p + 1] = k
        self.v[layer, :, :, p : p + 1] = v
        return self.k[layer, :, :, : p + 1], self.v[layer, :, :, : p + 1]

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


class MLALatentCache:
    """Latent-KV cache for MLA (DeepSeek): stores only the compressed c_KV + k_pe
    per token (`latent_dim = kv_lora_rank + qk_rope_head_dim`), one shared MQA-style
    "head", instead of full per-head K/V. `MLAAttention` up-projects the read-back
    latent through kv_b_proj to get per-head K/V at decode time.

    Engine-compatible: provides ``alloc``/``free`` slot management (simple free-list
    over the fixed ``batch`` dimension), plus no-op stubs for the scheduler's
    prefix-cache and capacity hooks that only ``PagedKVCache`` implements.
    """

    def __init__(self, num_layers, batch, latent_dim, max_len, *, device, dtype=torch.float16):
        self.kv = torch.zeros((num_layers, batch, max_len, latent_dim), device=device, dtype=dtype)
        self.length = 0
        self.num_slots = batch
        self.max_len = max_len
        self._free_slots = list(range(batch))

    # -- slot management (engine / scheduler interface) -----------------------

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("MLALatentCache: no free slots")
        return self._free_slots.pop()

    def free(self, slot: int):
        if slot not in self._free_slots:
            self._free_slots.append(slot)

    def has_free_slot(self) -> bool:
        return len(self._free_slots) > 0

    def has_free_block(self) -> bool:
        # Pre-allocated fixed per-slot region — no shared block pool, so slot
        # availability alone bounds admission; blocks are never the scarce resource.
        return True

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        pass  # MLALatentCache is pre-allocated; capacity is fixed.

    def store_prefix(self, token_ids: list[int], slot: int):
        pass  # Prefix cache not applicable to latent storage.

    def lookup_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        return 0, []

    def share_blocks(self, slot: int, blocks: list[int]):
        pass  # No block sharing for latent cache.

    # -- latent-KV read/write ------------------------------------------------

    def reset(self):
        self.length = 0

    def write_prefill(self, layer: int, latent: torch.Tensor, *, slot: int | None = None):
        """latent: [B, S, D] at positions [0, S). When *slot* is given, B must be 1
        and only that slot row is written (engine path — one seq at a time)."""
        s = latent.shape[1]
        if slot is not None:
            self.kv[layer, slot, :s] = latent[0]
        else:
            self.kv[layer, : latent.shape[0], :s] = latent

    def append_decode(
        self,
        layer: int,
        latent: torch.Tensor,
        *,
        slot: int | None = None,
        slots: list[int] | None = None,
    ) -> torch.Tensor:
        """latent: [B, 1, D] at position ``length``.

        *slot*  (int, engine prefill): B=1, write/return that slot row only.
        *slots* (list[int], engine decode): write each batch row to its slot
                 and return the per-slot cached history.
        Neither (ModelRunner path): write/return rows 0..B-1.
        """
        p = self.length
        if slots is not None:
            B = latent.shape[0]
            for i in range(B):
                self.kv[layer, slots[i], p : p + 1] = latent[i : i + 1]
            return torch.stack([self.kv[layer, s, : p + 1] for s in slots], dim=0)
        if slot is not None:
            self.kv[layer, slot, p : p + 1] = latent[0:1]
            return self.kv[layer, slot : slot + 1, : p + 1]
        B = latent.shape[0]
        self.kv[layer, :B, p : p + 1] = latent
        return self.kv[layer, :B, : p + 1]

    def advance(self, n: int = 1):
        self.length += n
