# SPDX-License-Identifier: MIT
"""Scheduler — continuous batching over waiting/running sequences (nano-vllm shape).

Each `schedule()` returns either a batch of WAITING sequences to prefill (a slot is
allocated per sequence) or the full RUNNING set to decode one step. Prefill is
preferred while slots and the batch-token budget allow, so new requests join
quickly; otherwise the running set decodes. Finished sequences free their slot.

When the slot pool is full and sequences are waiting, the scheduler preempts the
lowest-priority running sequence (fewest generated tokens) to free a slot rather
than blocking indefinitely — the preempted sequence is returned to the waiting
queue and its full context (prompt + previously generated tokens) is recomputed
when it is re-admitted.
"""
from __future__ import annotations

from collections import deque

from .kv_cache import PagedKVCache
from .sequence import Sequence, Status


class Scheduler:
    def __init__(self, cache: PagedKVCache, *, max_num_seqs: int, max_batch_tokens: int,
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

    def _preempt_one(self):
        """Evict the running sequence with the fewest generated tokens and put it
        back in the waiting queue where its full context will be recomputed."""
        preempted = min(self.running, key=lambda s: len(s.output_ids))
        self.running.remove(preempted)
        self.cache.free(preempted.slot)
        if preempted.is_finished(self.eos_id):
            preempted.status = Status.FINISHED
            return
        # Extend prompt_ids so the model recomputes through the full context
        # (original prompt + previously generated tokens) on re-admission.
        # output_ids is NOT reset: it accumulates tokens across all admissions
        # so the caller sees the complete generation.
        preempted.prompt_ids = preempted.all_token_ids
        preempted.length = 0
        preempted.slot = -1
        preempted.status = Status.WAITING
        self.waiting.append(preempted)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (batch, is_prefill)."""
        # Prefill newly-waiting sequences while we have slots + token budget.
        batch, tokens = [], 0
        while True:
            while self.waiting and len(self.running) + len(batch) < self.max_num_seqs \
                    and self.cache.has_free_slot():
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
            # No room — preempt a running sequence to free a slot for waiters.
            if self.waiting and (len(self.running) >= self.max_num_seqs
                                 or not self.cache.has_free_slot()):
                self._preempt_one()
                continue
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
