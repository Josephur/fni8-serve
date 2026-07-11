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
    def __init__(
        self,
        num_layers,
        num_slots,
        num_kv_heads,
        max_len,
        head_dim,
        *,
        device,
        block_size: int = 16,
        num_blocks: int | None = None,
    ):
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
        self._block_refcount: dict[int, int] = {}
        self._prefix_trie: dict = {}
        # Pinned host staging for the CUDA-graph decode hot path (issue #183): the
        # per-step slot-mapping / block-table used to be rebuilt with
        # `torch.tensor(list, device=cuda)` (a blocking pageable H2D) then D2D-copied
        # into the captured graph buffers. `fill_slot_mapping` / `fill_block_table`
        # fill these pinned buffers in place and issue ONE non_blocking copy into the
        # caller's persistent device tensor. Allocated lazily / grown on demand.
        self._pin = device != "cpu" and torch.cuda.is_available()
        self._slot_map_host: torch.Tensor | None = None
        self._block_table_host: torch.Tensor | None = None

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("PagedKVCache: no free slots")
        return self._free_slots.pop()

    def free(self, slot: int):
        """Recycle a finished sequence's blocks back into the shared pool.
        Reference-counted: blocks shared via prefix cache are only freed when
        the last reference is gone."""
        for blk in self._slot_blocks[slot]:
            c = self._block_refcount.get(blk, 1) - 1
            if c <= 0:
                self._block_refcount.pop(blk, None)
                self._free_blocks.append(blk)
            else:
                self._block_refcount[blk] = c
        self._slot_blocks[slot] = []
        self._free_slots.append(slot)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def has_free_block(self) -> bool:
        """Whether the shared block pool has at least one free physical block. The
        scheduler uses this to distinguish genuine KV-memory pressure (dry pool)
        from a mere count/slot cap: only the former justifies preemption."""
        return bool(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        """Physical blocks currently pinned (allocated) out of `num_blocks` --
        drives the KV-cache usage % in the telemetry heartbeat / TUI. Host-side
        int arithmetic only; no GPU sync."""
        return self.num_blocks - len(self._free_blocks)

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        """Grow each slot's block table so it can hold `lengths[i]` tokens,
        pulling new physical blocks from the shared pool one at a time."""
        for slot, n in zip(slots, lengths):
            blocks = self._slot_blocks[slot]
            need = (n + self.block_size - 1) // self.block_size
            while len(blocks) < need:
                if not self._free_blocks:
                    raise RuntimeError("PagedKVCache: no free blocks")
                blk = self._free_blocks.pop()
                blocks.append(blk)
                self._block_refcount[blk] = 1

    def share_blocks(self, slot: int, blocks: list[int]):
        """Point *slot* at existing physical *blocks* and increment their refcounts."""
        self._slot_blocks[slot] = list(blocks)
        for blk in blocks:
            self._block_refcount[blk] = self._block_refcount.get(blk, 1) + 1

    def store_prefix(self, token_ids: list[int], slot: int):
        """Store a completed prefix in the radix trie for future lookups.
        Stores entries at every block-aligned boundary so that shorter lookups
        that diverge at the suffix can still find the longest block-aligned match."""
        n = len(token_ids)
        num_shared = n // self.block_size
        if num_shared == 0:
            return
        full_blocks = list(self._slot_blocks[slot])
        node = self._prefix_trie
        for i, tid in enumerate(token_ids):
            node = node.setdefault(tid, {})
            pos = i + 1
            if pos % self.block_size == 0 and pos // self.block_size <= num_shared:
                key = "_"
                if key not in node:
                    nb = pos // self.block_size
                    node[key] = {"blocks": full_blocks[:nb], "num_tokens": pos, "slot": slot}
                    for blk in full_blocks[:nb]:
                        self._block_refcount[blk] = self._block_refcount.get(blk, 1) + 1

    def lookup_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Find the longest block-aligned matching prefix.
        Returns (matched_len, shared_blocks)."""
        node = self._prefix_trie
        best_len = 0
        best_blocks: list[int] = []
        for i, tid in enumerate(token_ids):
            if tid not in node:
                break
            node = node[tid]
            if "_" in node:
                entry = node["_"]
                stored = entry["num_tokens"]
                if i + 1 >= stored:
                    best_len = stored
                    best_blocks = list(entry["blocks"])
        return best_len, best_blocks

    def _slot_mapping(self, slots: list[int], positions: list[int]) -> torch.Tensor:
        flat = [
            self._slot_blocks[s][p // self.block_size] * self.block_size + p % self.block_size
            for s, p in zip(slots, positions)
        ]
        return torch.tensor(flat, dtype=torch.int32, device=self.device)

    # Public alias -- `engine/cuda_graph.py` builds this tensor itself (once per
    # step, reused across every layer) instead of going through `write_decode`.
    def slot_mapping_for(self, slots: list[int], positions: list[int]) -> torch.Tensor:
        return self._slot_mapping(slots, positions)

    def _ensure_decode_staging(self, batch_size: int):
        if self._slot_map_host is None or self._slot_map_host.numel() < batch_size:
            self._slot_map_host = torch.empty(batch_size, dtype=torch.int32, pin_memory=self._pin)
        if self._block_table_host is None or self._block_table_host.shape[0] < batch_size:
            self._block_table_host = torch.zeros(
                batch_size, self.max_blocks_per_seq, dtype=torch.int32, pin_memory=self._pin
            )

    def fill_slot_mapping(self, dst: torch.Tensor, slots: list[int], positions: list[int]):
        """Refresh the CUDA-graph decode slot-mapping buffer `dst` in place (same
        device pointer the graph captured) via a pinned host staging buffer + one
        non_blocking copy -- no per-step device allocation, no blocking H2D."""
        n = len(slots)
        self._ensure_decode_staging(n)
        bs = self.block_size
        flat = [self._slot_blocks[s][p // bs] * bs + p % bs for s, p in zip(slots, positions)]
        self._slot_map_host[:n].copy_(torch.tensor(flat, dtype=torch.int32))
        dst.copy_(self._slot_map_host[:n], non_blocking=self._pin)

    def fill_block_table(self, dst: torch.Tensor, slots: list[int]):
        """Refresh the CUDA-graph decode block-table buffer `dst` in place via a
        pinned host staging buffer + one non_blocking copy (companion to
        `fill_slot_mapping`). Rows shorter than the widest sequence are zero-padded
        (never read: the kernel gates every read by `context_lens`)."""
        n = len(slots)
        self._ensure_decode_staging(n)
        host = self._block_table_host[:n]
        host.zero_()
        for i, s in enumerate(slots):
            blocks = self._slot_blocks[s]
            if blocks:
                host[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        dst.copy_(host, non_blocking=self._pin)

    def write_prefill(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot: int, start: int = 0
    ):
        """k, v: [1, Hkv, S, D] fp16 -> quantize-on-write, one position at a time
        (the kernel's write granularity), into this one sequence's blocks.
        *start* skips the first *start* positions (shared-prefix reuse)."""
        s = k.shape[2]
        for t in range(start, s):
            mapping = self._slot_mapping([slot], [t])
            fni8.quantize_kv_write_paged(
                k[:, :, t, :].contiguous(),
                v[:, :, t, :].contiguous(),
                self.k_cache[layer],
                self.k_scale[layer],
                self.v_cache[layer],
                self.v_scale[layer],
                mapping,
            )

    def write_prefill_varlen(
        self, layer: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        """k, v: [total_tokens, Hkv, D] fp16. slot_mapping: [total_tokens] int32.
        Write every token's K/V to its correct page in ONE call — the varlen
        batched prefill path. Every token's slot_mapping entry points to the
        correct physical page + offset for its sequence and position, including
        tokens that were already filled by a shared prefix (overwrite is cheap
        and avoids per-sequence branching)."""
        fni8.quantize_kv_write_paged(
            k.contiguous(),
            v.contiguous(),
            self.k_cache[layer],
            self.k_scale[layer],
            self.v_cache[layer],
            self.v_scale[layer],
            slot_mapping,
        )

    def write_decode(
        self,
        layer: int,
        slots: list[int],
        positions: list[int],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        """k_new, v_new: [B, Hkv, D] fp16 -- the newest token for every sequence in
        the batch, committed with ONE `quantize_kv_write_paged` call."""
        self.write_decode_static(layer, self._slot_mapping(slots, positions), k_new, v_new)

    def write_decode_static(
        self, layer: int, slot_mapping: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor
    ):
        """Same as `write_decode`, but takes an already-built `slot_mapping` device
        tensor instead of python `slots`/`positions` lists -- the CUDA-graph decode
        path (engine/cuda_graph.py) calls this with a persistent buffer it refreshes
        via `copy_` before each replay, since a captured graph can only re-execute
        kernels against fixed memory, not rebuild tensors from python lists."""
        fni8.quantize_kv_write_paged(
            k_new.contiguous(),
            v_new.contiguous(),
            self.k_cache[layer],
            self.k_scale[layer],
            self.v_cache[layer],
            self.v_scale[layer],
            slot_mapping,
        )

    def block_table(self, slots: list[int]) -> torch.Tensor:
        """[B, max_blocks_per_seq] int32 physical block ids for this batch -- rows
        shorter than the widest sequence are zero-padded (never read: the kernel
        gates every read by `context_lens`)."""
        bt = torch.zeros(len(slots), self.max_blocks_per_seq, dtype=torch.int32, device=self.device)
        for i, s in enumerate(slots):
            blocks = self._slot_blocks[s]
            if blocks:
                bt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
        return bt

    def decode_attn(
        self, layer: int, q: torch.Tensor, slots: list[int], lengths: list[int], *, scale: float
    ) -> torch.Tensor:
        """ONE batched paged-decode launch across the whole ragged running batch --
        `lengths[i]` is the write position of the token just committed by
        `write_decode`, so the valid context per row is `lengths[i] + 1`."""
        context_lens = torch.tensor([n + 1 for n in lengths], dtype=torch.int32, device=self.device)
        return self.decode_attn_static(
            layer,
            q,
            self.block_table(slots),
            context_lens,
            int(context_lens.max().item()),
            scale=scale,
        )

    def decode_attn_static(
        self,
        layer: int,
        q: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Same as `decode_attn`, but takes precomputed `block_table`/`context_lens`
        device tensors and `max_context_len` as a plain python int instead of calling
        `context_lens.max().item()` -- that `.item()` is a device->host sync, which
        CUDA graph capture cannot contain. The CUDA-graph decode path passes a
        per-bucket compile-time upper bound here (safe: `attn_paged_decode_cached`
        only requires `max_context_len >= max(context_lens)`, since it just
        upper-bounds kernel split-sizing and every row is still gated by its own
        `context_lens` entry)."""
        return fni8.attn_paged_decode_cached(
            q,
            self.k_cache[layer],
            self.k_scale[layer],
            self.v_cache[layer],
            self.v_scale[layer],
            block_table,
            context_lens,
            self.block_size,
            max_context_len=max_context_len,
            scale=scale,
        )

    def build_verify_cache(
        self,
        layer: int,
        slots: list[int],
        lengths: list[int],
        k_draft: torch.Tensor,
        v_draft: torch.Tensor,
    ):
        """Build a contiguous int8 KV cache for the verify kernel: prefix K/V
        (read from the paged store and dequantised) + draft K/V concatenated.

        *k_draft*, *v_draft*: fp16 [B, Hkv, k, D]. Returns
        ``(k_i8, k_scale, v_i8, v_scale)`` shaped ``[B, Hkv, N, D]`` /
        ``[B, Hkv, N]`` float32 as required by :func:`fni8.attn_int8_verify`."""
        B = len(slots)
        max_prefix = max(lengths) if lengths else 0
        k_parts, v_parts = [], []
        for b in range(B):
            kp, vp = self.read_dense(layer, slots[b], lengths[b])
            if lengths[b] < max_prefix:
                pad_sz = max_prefix - lengths[b]
                pad = kp.new_zeros(1, kp.shape[1], pad_sz, kp.shape[3])
                kp = torch.cat([kp, pad], dim=2)
                vp = torch.cat([vp, pad], dim=2)
            k_parts.append(torch.cat([kp, k_draft[b : b + 1]], dim=2))
            v_parts.append(torch.cat([vp, v_draft[b : b + 1]], dim=2))
        full_k = torch.cat(k_parts, dim=0)  # [B, Hkv, N, D]
        full_v = torch.cat(v_parts, dim=0)
        return fni8.quantize_kv_cache(full_k, full_v)

    def read_dense(self, layer: int, slot: int, length: int, *, window: int | None = None):
        """Dequantize one sequence's cached K/V back to fp16 [1,Hkv,N,D] -- the
        sliding-window fallback (`attn_paged_decode_cached` has no window parameter
        yet). `window` caps N to the last `window` positions."""
        start = max(0, length - window) if window is not None else 0
        blocks = self._slot_blocks[slot]
        positions = list(range(start, length))
        blk = torch.tensor([blocks[p // self.block_size] for p in positions], device=self.device)
        off = torch.tensor([p % self.block_size for p in positions], device=self.device)
        k = (
            self.k_cache[layer, blk, :, off, :].float()
            * self.k_scale[layer, blk, :, off].unsqueeze(-1)
        ).to(torch.float16)
        v = (
            self.v_cache[layer, blk, :, off, :].float()
            * self.v_scale[layer, blk, :, off].unsqueeze(-1)
        ).to(torch.float16)
        k = rotate_last(k)  # undo the write-time Hadamard rotation (involution)
        return (
            k.permute(1, 0, 2).unsqueeze(0).contiguous(),
            v.permute(1, 0, 2).unsqueeze(0).contiguous(),
        )
