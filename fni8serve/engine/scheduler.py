# SPDX-License-Identifier: MIT
"""Scheduler — continuous batching over waiting/running sequences (nano-vllm shape).

Each `schedule()` returns either a batch of WAITING sequences to prefill (a slot is
allocated per sequence) or the full RUNNING set to decode one step. Prefill is
preferred while slots and the batch-token budget allow, so new requests join
quickly; otherwise the running set decodes. Finished sequences free their slot.
"""
from __future__ import annotations

from collections import deque

from .kv_cache import BatchedKVCache
from .sequence import Sequence, Status


class Scheduler:
    def __init__(self, cache: BatchedKVCache, *, max_num_seqs: int, max_batch_tokens: int,
                 eos_id: int | None):
        self.cache = cache
        self.max_num_seqs = max_num_seqs
        self.max_batch_tokens = max_batch_tokens
        self.eos_id = eos_id
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (batch, is_prefill)."""
        # Prefill newly-waiting sequences while we have slots + token budget.
        batch, tokens = [], 0
        while self.waiting and len(self.running) + len(batch) < self.max_num_seqs \
                and self.cache._free:
            seq = self.waiting[0]
            if batch and tokens + seq.num_prompt > self.max_batch_tokens:
                break
            self.waiting.popleft()
            seq.slot = self.cache.alloc()
            seq.status = Status.RUNNING
            tokens += seq.num_prompt
            batch.append(seq)
        if batch:
            return batch, True
        return list(self.running), False

    def postprocess(self, batch: list[Sequence], is_prefill: bool):
        if is_prefill:
            self.running.extend(batch)
        still = []
        for seq in self.running:
            if seq.is_finished(self.eos_id):
                seq.status = Status.FINISHED
                self.cache.free(seq.slot)
            else:
                still.append(seq)
        self.running = still
