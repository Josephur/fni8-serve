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

import torch

from ..engine.sequence import SamplingParams, Status


@dataclass
class Done:
    """Sentinel yielded (once, last) by `EngineWorker.stream` when a request finishes."""

    reason: str


@dataclass
class _EncodeRequest:
    """A pooled-embedding request routed through the worker so `engine.encode()`
    runs on the SAME thread as `engine.step()` -- the engine (scheduler + KV cache)
    is not safe to touch from the FastAPI handler thread concurrently with the
    worker's step loop. `out` carries back the embedding (or the raised exception)."""

    prompt_ids: list[int]
    out: queue.Queue = field(default_factory=queue.Queue)


@dataclass
class _PendingRequest:
    prompt_ids: list[int]
    params: SamplingParams
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    out_queue: queue.Queue = field(default_factory=queue.Queue)
    seq_id: int | None = None
    sent: int = 0


class EngineWorker:
    def __init__(self, engine, *, stats=None) -> None:
        self.engine = engine
        # Optional fni8serve.metrics.StatsCollector: the worker owns the per-request
        # lifecycle (arrival -> first token -> finish), so it records TTFT / ITL /
        # e2e-latency and the request counters here. All host-side timestamps.
        self.stats = stats
        self._inbox: queue.Queue[_PendingRequest] = queue.Queue()
        self._pending: dict[int, _PendingRequest] = {}
        threading.Thread(target=self._run, daemon=True, name="fni8serve-engine").start()

    def submit(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> queue.Queue:
        """Enqueue a generation request; returns the queue it will be streamed onto
        (token ids, terminated by a single `Done`)."""
        req = _PendingRequest(
            prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        self._inbox.put(req)
        return req.out_queue

    def encode(self, prompt_ids: list[int]) -> list[float]:
        """Encode a prompt and return the pooled embedding vector.

        Routed through the worker's inbox (like `submit`) so `engine.encode()` runs
        on the dedicated engine thread, serialized with `engine.step()` -- calling it
        straight from the FastAPI handler thread raced the step loop over the shared
        scheduler / KV cache (data race C1). Blocks the caller until the worker
        thread produces the embedding; the caller should run it off the event loop
        (the API handlers use `run_in_executor`)."""
        req = _EncodeRequest(prompt_ids)
        self._inbox.put(req)
        result = req.out.get()
        if isinstance(result, BaseException):
            raise result
        return result

    async def stream(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> AsyncIterator[int | Done]:
        out_q = self.submit(
            prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, out_q.get)
            yield item
            if isinstance(item, Done):
                return

    def _run(self) -> None:
        while True:
            if not self._pending:
                self._intake(self._inbox.get())  # idle: block for the next request
            while True:
                try:
                    self._intake(self._inbox.get_nowait())
                except queue.Empty:
                    break
            # Only step when a generation is in flight. An idle worker that only just
            # served an encode has nothing to decode, so skip the empty step and loop
            # back to block on the inbox.
            if self._pending:
                self.engine.step()
                self._dispatch()

    def _intake(self, item: _PendingRequest | _EncodeRequest) -> None:
        """Dispatch one inbox item on the worker thread: an encode is answered
        inline (serialized with `step`), a generation request is registered."""
        if isinstance(item, _EncodeRequest):
            try:
                item.out.put(self.engine.encode(item.prompt_ids))
            except BaseException as exc:  # noqa: BLE001 -- relayed to the caller thread
                item.out.put(exc)
        else:
            self._register(item)

    def _register(self, req: _PendingRequest) -> None:
        req.seq_id = self.engine.add_request(req.prompt_ids, req.params)
        seq = self.engine.sequence(req.seq_id)
        if req.pixel_values is not None:
            seq.pixel_values = req.pixel_values
            seq.image_grid_thw = req.image_grid_thw
        self._pending[req.seq_id] = req
        if self.stats is not None:
            p = req.params
            self.stats.record_request_start(
                req.seq_id,
                prompt_tokens=len(req.prompt_ids),
                sampling={
                    "temperature": p.temperature,
                    "top_p": p.top_p,
                    "max_tokens": p.max_tokens,
                },
            )

    def _dispatch(self) -> None:
        done_ids = []
        for seq_id, req in self._pending.items():
            seq = self.engine.sequence(seq_id)
            if self.stats is not None and req.sent == 0 and seq.output_ids:
                self.stats.record_first_token(seq_id)
            while req.sent < len(seq.output_ids):
                req.out_queue.put(seq.output_ids[req.sent])
                req.sent += 1
            if seq.status is Status.FINISHED:
                reason = seq.finish_reason(self.engine.eos_id) or "stop"
                # Record the finish BEFORE signalling Done so the metrics counters
                # are updated by the time the HTTP handler (woken by Done on the
                # queue) can observe them -- otherwise a fast client could GET
                # /metrics before this worker thread bumps `finished_requests`.
                if self.stats is not None:
                    self.stats.record_request_finish(
                        seq_id, output_tokens=len(seq.output_ids), finish_reason=reason
                    )
                req.out_queue.put(Done(reason))
                done_ids.append(seq_id)
        for seq_id in done_ids:
            del self._pending[seq_id]
            self.engine.forget(seq_id)
