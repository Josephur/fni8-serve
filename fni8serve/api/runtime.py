# SPDX-License-Identifier: MIT
"""EngineWorker — the API layer's single owner of an `LLMEngine`.

`LLMEngine.step()` and its `Scheduler` are plain Python state (a deque + a list),
not safe to call concurrently from multiple threads. One dedicated thread drains a
request queue and drives the scheduler, so concurrent HTTP requests still share the
engine's continuous-batching loop -- exactly what `LLMEngine.generate()` does for a
list of prompts, just fed incrementally over time (as HTTP requests arrive) instead
of all at once.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..engine.sequence import SamplingParams, Status


@dataclass
class Done:
    """Sentinel yielded (once, last) by `EngineWorker.stream` when a request finishes."""
    reason: str


@dataclass
class _PendingRequest:
    prompt_ids: list[int]
    params: SamplingParams
    out_queue: queue.Queue = field(default_factory=queue.Queue)
    seq_id: int | None = None
    sent: int = 0


class EngineWorker:
    def __init__(self, engine) -> None:
        self.engine = engine
        self._inbox: queue.Queue[_PendingRequest] = queue.Queue()
        self._pending: dict[int, _PendingRequest] = {}
        threading.Thread(target=self._run, daemon=True, name="fni8serve-engine").start()

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> queue.Queue:
        """Enqueue a generation request; returns the queue it will be streamed onto
        (token ids, terminated by a single `Done`)."""
        req = _PendingRequest(prompt_ids, params)
        self._inbox.put(req)
        return req.out_queue

    async def stream(
        self, prompt_ids: list[int], params: SamplingParams,
    ) -> AsyncIterator[int | Done]:
        out_q = self.submit(prompt_ids, params)
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, out_q.get)
            yield item
            if isinstance(item, Done):
                return

    def _run(self) -> None:
        while True:
            if not self._pending:
                self._register(self._inbox.get())    # idle: block for the next request
            while True:
                try:
                    self._register(self._inbox.get_nowait())
                except queue.Empty:
                    break
            self.engine.step()
            self._dispatch()

    def _register(self, req: _PendingRequest) -> None:
        req.seq_id = self.engine.add_request(req.prompt_ids, req.params)
        self._pending[req.seq_id] = req

    def _dispatch(self) -> None:
        done_ids = []
        for seq_id, req in self._pending.items():
            seq = self.engine.sequence(seq_id)
            while req.sent < len(seq.output_ids):
                req.out_queue.put(seq.output_ids[req.sent])
                req.sent += 1
            if seq.status is Status.FINISHED:
                req.out_queue.put(Done(seq.finish_reason(self.engine.eos_id) or "stop"))
                done_ids.append(seq_id)
        for seq_id in done_ids:
            del self._pending[seq_id]
            self.engine.forget(seq_id)
