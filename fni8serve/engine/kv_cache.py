# SPDX-License-Identifier: MIT
"""BatchedKVCache — slot-based KV store for continuous batching.

`num_slots` concurrent sequences, each pinned to a slot with its own length. This
is still contiguous-per-slot (not paged): a paged block-table cache needs the fni8
decode kernel to accept block tables (a kernel TODO). The slot model is enough for
correct continuous batching today — the engine allocates/frees slots as requests
enter and finish, and decode reads each slot's own [Hkv, len, D] region.
"""
from __future__ import annotations

import torch


class BatchedKVCache:
    def __init__(self, num_layers, num_slots, num_kv_heads, max_len, head_dim,
                 *, device, dtype=torch.float16):
        shape = (num_layers, num_slots, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.num_slots = num_slots
        self.max_len = max_len
        self._free = list(range(num_slots))

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("BatchedKVCache: no free slots")
        return self._free.pop()

    def free(self, slot: int):
        self._free.append(slot)

    def write_prefill(self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot: int):
        """k, v: [1, Hkv, S, D] -> slot positions [0, S)."""
        s = k.shape[2]
        self.k[layer, slot, :, :s] = k[0]
        self.v[layer, slot, :, :s] = v[0]

    def append_read_slot(self, layer: int, slot: int, length: int,
                         k_row: torch.Tensor, v_row: torch.Tensor):
        """Append one token (k_row/v_row: [Hkv, 1, D]) at `length`, return the
        [1, Hkv, length+1, D] slice to attend against."""
        self.k[layer, slot, :, length:length + 1] = k_row
        self.v[layer, slot, :, length:length + 1] = v_row
        return (self.k[layer, slot, :, :length + 1].unsqueeze(0),
                self.v[layer, slot, :, :length + 1].unsqueeze(0))
