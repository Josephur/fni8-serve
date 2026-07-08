# SPDX-License-Identifier: MIT
"""PagedKVCache — block-table paged, int8 quantize-on-write KV store for
continuous batching.

Physical storage is one pool of `block_size`-token int8 blocks per layer, shared
across every sequence; each sequence gets a "slot" (a block-table row) grown one
block at a time as its length crosses a block boundary, so a short sequence only
ever pins the few blocks it actually uses instead of reserving a `max_len`-sized
region up front (the old `BatchedKVCache` this replaces). A finished sequence's
blocks return to the free pool immediately (`free`) for the next request to
reuse.

Every new token is quantized straight to int8 on write (`fni8.quantize_kv_write_paged`,
per-token RTN scale) — no fp16 K/V is ever kept resident. Decode reads the WHOLE
ragged running batch in one `fni8.attn_paged_decode_cached` launch (one block table
+ one context-lens tensor for every row), replacing the old per-slot Python loop
over `attn_int8_decode`.

Sliding-window layers (Gemma3) are the one gap `attn_paged_decode_cached` doesn't
cover (no window parameter yet): `read_dense` reconstructs a small dequantized fp16
window slice for the existing `attn_int8_decode` fallback in that case — see
`GQAAttention._decode_batched`.
"""
from __future__ import annotations

import torch

import fni8
from fni8.quant.rotation import rotate_last


class PagedKVCache:
    def __init__(self, num_layers, num_slots, num_kv_heads, max_len, head_dim,
                 *, device, block_size: int = 16, num_blocks: int | None = None):
        self.block_size = block_size
        self.max_blocks_per_seq = (max_len + block_size - 1) // block_size
        # Worst case (every slot at max_len) by default -- never starves the
        # scheduler's slot-based admission; pass num_blocks to over-subscribe
        # the pool for higher throughput once average length < max_len.
        self.num_blocks = num_blocks or num_slots * self.max_blocks_per_seq
        self.device = device
        shape = (num_layers, self.num_blocks, num_kv_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        scale_shape = (num_layers, self.num_blocks, num_kv_heads, block_size)
        self.k_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        self.v_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        self.num_slots = num_slots
        self.max_len = max_len
        self._free_blocks = list(range(self.num_blocks))
        self._free_slots = list(range(num_slots))
        self._slot_blocks: list[list[int]] = [[] for _ in range(num_slots)]

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("PagedKVCache: no free slots")
        return self._free_slots.pop()

    def free(self, slot: int):
        """Recycle a finished sequence's blocks back into the shared pool."""
        self._free_blocks.extend(self._slot_blocks[slot])
        self._slot_blocks[slot] = []
        self._free_slots.append(slot)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        """Grow each slot's block table so it can hold `lengths[i]` tokens,
        pulling new physical blocks from the shared pool one at a time."""
        for slot, n in zip(slots, lengths):
            blocks = self._slot_blocks[slot]
            need = (n + self.block_size - 1) // self.block_size
            while len(blocks) < need:
                if not self._free_blocks:
                    raise RuntimeError("PagedKVCache: no free blocks")
                blocks.append(self._free_blocks.pop())

    def _slot_mapping(self, slots: list[int], positions: list[int]) -> torch.Tensor:
        flat = [self._slot_blocks[s][p // self.block_size] * self.block_size + p % self.block_size
                for s, p in zip(slots, positions)]
        return torch.tensor(flat, dtype=torch.int32, device=self.device)

    # Public alias -- `engine/cuda_graph.py` builds this tensor itself (once per
    # step, reused across every layer) instead of going through `write_decode`.
    def slot_mapping_for(self, slots: list[int], positions: list[int]) -> torch.Tensor:
        return self._slot_mapping(slots, positions)

    def write_prefill(self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot: int):
        """k, v: [1, Hkv, S, D] fp16 -> quantize-on-write, one position at a time
        (the kernel's write granularity), into this one sequence's blocks."""
        s = k.shape[2]
        for t in range(s):
            mapping = self._slot_mapping([slot], [t])
            fni8.quantize_kv_write_paged(
                k[:, :, t, :].contiguous(), v[:, :, t, :].contiguous(),
                self.k_cache[layer], self.k_scale[layer],
                self.v_cache[layer], self.v_scale[layer], mapping,
            )

    def write_decode(self, layer: int, slots: list[int], positions: list[int],
                     k_new: torch.Tensor, v_new: torch.Tensor):
        """k_new, v_new: [B, Hkv, D] fp16 -- the newest token for every sequence in
        the batch, committed with ONE `quantize_kv_write_paged` call."""
        self.write_decode_static(layer, self._slot_mapping(slots, positions), k_new, v_new)

    def write_decode_static(self, layer: int, slot_mapping: torch.Tensor,
                            k_new: torch.Tensor, v_new: torch.Tensor):
        """Same as `write_decode`, but takes an already-built `slot_mapping` device
        tensor instead of python `slots`/`positions` lists -- the CUDA-graph decode
        path (engine/cuda_graph.py) calls this with a persistent buffer it refreshes
        via `copy_` before each replay, since a captured graph can only re-execute
        kernels against fixed memory, not rebuild tensors from python lists."""
        fni8.quantize_kv_write_paged(
            k_new.contiguous(), v_new.contiguous(),
            self.k_cache[layer], self.k_scale[layer],
            self.v_cache[layer], self.v_scale[layer], slot_mapping,
        )

    def block_table(self, slots: list[int]) -> torch.Tensor:
        """[B, max_blocks_per_seq] int32 physical block ids for this batch -- rows
        shorter than the widest sequence are zero-padded (never read: the kernel
        gates every read by `context_lens`)."""
        bt = torch.zeros(len(slots), self.max_blocks_per_seq, dtype=torch.int32, device=self.device)
        for i, s in enumerate(slots):
            blocks = self._slot_blocks[s]
            if blocks:
                bt[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
        return bt

    def decode_attn(self, layer: int, q: torch.Tensor, slots: list[int],
                    lengths: list[int], *, scale: float) -> torch.Tensor:
        """ONE batched paged-decode launch across the whole ragged running batch --
        `lengths[i]` is the write position of the token just committed by
        `write_decode`, so the valid context per row is `lengths[i] + 1`."""
        context_lens = torch.tensor([n + 1 for n in lengths], dtype=torch.int32, device=self.device)
        return self.decode_attn_static(layer, q, self.block_table(slots), context_lens,
                                       int(context_lens.max().item()), scale=scale)

    def decode_attn_static(self, layer: int, q: torch.Tensor, block_table: torch.Tensor,
                           context_lens: torch.Tensor, max_context_len: int, *,
                           scale: float) -> torch.Tensor:
        """Same as `decode_attn`, but takes precomputed `block_table`/`context_lens`
        device tensors and `max_context_len` as a plain python int instead of calling
        `context_lens.max().item()` -- that `.item()` is a device->host sync, which
        CUDA graph capture cannot contain. The CUDA-graph decode path passes a
        per-bucket compile-time upper bound here (safe: `attn_paged_decode_cached`
        only requires `max_context_len >= max(context_lens)`, since it just
        upper-bounds kernel split-sizing and every row is still gated by its own
        `context_lens` entry)."""
        return fni8.attn_paged_decode_cached(
            q, self.k_cache[layer], self.k_scale[layer],
            self.v_cache[layer], self.v_scale[layer],
            block_table, context_lens, self.block_size,
            max_context_len=max_context_len, scale=scale,
        )

    def read_dense(self, layer: int, slot: int, length: int, *, window: int | None = None):
        """Dequantize one sequence's cached K/V back to fp16 [1,Hkv,N,D] -- the
        sliding-window fallback (`attn_paged_decode_cached` has no window parameter
        yet). `window` caps N to the last `window` positions."""
        start = max(0, length - window) if window is not None else 0
        blocks = self._slot_blocks[slot]
        positions = list(range(start, length))
        blk = torch.tensor([blocks[p // self.block_size] for p in positions], device=self.device)
        off = torch.tensor([p % self.block_size for p in positions], device=self.device)
        k = (self.k_cache[layer, blk, :, off, :].float()
             * self.k_scale[layer, blk, :, off].unsqueeze(-1)).to(torch.float16)
        v = (self.v_cache[layer, blk, :, off, :].float()
             * self.v_scale[layer, blk, :, off].unsqueeze(-1)).to(torch.float16)
        k = rotate_last(k)   # undo the write-time Hadamard rotation (involution)
        return (k.permute(1, 0, 2).unsqueeze(0).contiguous(),
                v.permute(1, 0, 2).unsqueeze(0).contiguous())
