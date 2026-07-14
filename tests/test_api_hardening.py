# SPDX-License-Identifier: MIT
"""Pre-auth DoS + encode-race hardening (C1/S1/S2/S3).

All against fake engine/tokenizer doubles (no CUDA/fni8/weights), mirroring
test_api.py / test_batches.py. Each test pins one confirmed vulnerability:

- C1: `/v1/embeddings` + `/v1/rerank` used to call `engine.encode()` straight from
  the FastAPI handler thread, concurrently with the EngineWorker thread's
  `engine.step()` (the engine is single-thread-only). encode must now run ON the
  worker thread.
- S1: `max_tokens` was never clamped to `max_len - prompt_len` -> block-table
  overflow / giant KV alloc. It must be truncated.
- S2: an over-length prompt was admitted unconditionally -> prefill OOM / KV OOB.
  It must be rejected cleanly (400 / per-line error).
- S3: `/v1/files`, `/v1/batches`, and data-URI images had no size caps.
"""

import base64
import itertools
import json
import threading

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from fni8serve.api.app import create_app  # noqa: E402
from fni8serve.api.runtime import EngineWorker  # noqa: E402
from fni8serve.engine.sequence import SamplingParams, Sequence, Status  # noqa: E402

EOS = 4


class CharTokenizer:
    """One token id per character, so a test can drive an exact prompt length."""

    eos_token_id = EOS

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, chat_template=None, tools=None
    ):
        return [ord(c) for c in messages[-1]["content"]]

    def encode(self, text, **kw):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        kept = [i for i in ids if not (skip_special_tokens and i == EOS)]
        return "".join(chr(i) for i in kept if 0 <= i < 0x110000)


class FakeEngine:
    """Deterministic 1-token-per-step engine that exposes `max_len` (so the API layer
    can read it for its clamp/guard) and records the thread each entrypoint runs on.

    It does NOT itself reject over-length prompts -- that lets the handler-level guard
    (S2) be tested in isolation: on unpatched code the request is admitted and decodes,
    on patched code the handler returns 400 before ever reaching the engine."""

    def __init__(self, max_len=None):
        self.eos_id = EOS
        self.max_len = max_len
        self._seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()
        self.step_threads: list[int] = []
        self.encode_threads: list[int] = []

    def add_request(self, prompt_ids, params: SamplingParams | None = None) -> int:
        seq_id = next(self._ids)
        seq = Sequence(seq_id, list(prompt_ids), params or SamplingParams())
        seq.max_len = self.max_len
        self._seqs[seq_id] = seq
        return seq_id

    def step(self) -> None:
        self.step_threads.append(threading.get_ident())
        for seq in self._seqs.values():
            if seq.status is Status.FINISHED:
                continue
            seq.output_ids.append((seq.last_token + 1) % 128)
            seq.status = Status.FINISHED if seq.is_finished(self.eos_id) else Status.RUNNING

    def encode(self, prompt_ids: list[int]) -> list[float]:
        self.encode_threads.append(threading.get_ident())
        return [0.5] * 8

    def sequence(self, seq_id: int) -> Sequence:
        return self._seqs[seq_id]

    def forget(self, seq_id: int) -> None:
        self._seqs.pop(seq_id, None)


# --- C1: encode runs on the worker thread, not the caller thread ------------


def test_encode_runs_on_worker_thread_not_caller():
    engine = FakeEngine()
    worker = EngineWorker(engine)
    caller_thread = threading.get_ident()

    out = worker.encode([1, 2, 3])

    assert out == [0.5] * 8
    assert engine.encode_threads, "encode never ran"
    (encode_thread,) = set(engine.encode_threads)
    assert encode_thread != caller_thread, "encode ran on the caller thread (race)"


def test_encode_and_step_share_one_thread():
    """encode and step must serialize on the SAME (worker) thread -- the whole point
    of routing encode through the worker instead of the handler thread."""
    engine = FakeEngine()
    worker = EngineWorker(engine)
    # Drive a generation so step() runs, plus an encode, and confirm both landed on
    # the single worker thread.
    q = worker.submit([1, 2], SamplingParams(max_tokens=2))
    worker.encode([7, 8])
    # drain the generation queue
    while True:
        item = q.get()
        if item.__class__.__name__ == "Done":
            break
    assert engine.step_threads and engine.encode_threads
    assert set(engine.step_threads) | set(engine.encode_threads) == {engine.step_threads[0]}


# --- app fixtures with a max_len-bearing engine -----------------------------


def _client(max_len=None):
    app = create_app(FakeEngine(max_len=max_len), CharTokenizer(), served_model_name="fake")
    return TestClient(app)


# --- S1: max_tokens clamped to remaining room -------------------------------


def test_completions_max_tokens_clamped_to_max_len():
    with _client(max_len=8) as c:
        # prompt is 5 tokens, max_len 8 -> at most 3 output tokens even though we ask 1000
        resp = c.post(
            "/v1/completions",
            json={
                "model": "fake",
                "prompt": "abcde",
                "max_tokens": 1000,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["usage"]["completion_tokens"] <= 3


# --- S2: over-length prompt rejected cleanly --------------------------------


def test_completions_over_length_prompt_rejected():
    with _client(max_len=8) as c:
        resp = c.post(
            "/v1/completions",
            json={
                "model": "fake",
                "prompt": "a" * 20,
                "max_tokens": 4,
            },
        )
        assert resp.status_code == 400, resp.text


def test_embeddings_over_length_prompt_rejected():
    with _client(max_len=8) as c:
        resp = c.post("/v1/embeddings", json={"model": "fake", "input": "a" * 20})
        assert resp.status_code == 400, resp.text


def test_llm_engine_add_request_rejects_over_length_prompt():
    """The real engine-level backstop for offline callers that never touch the API.
    Exercises `LLMEngine.add_request` directly (bypassing model construction) since the
    guard fires before any Sequence/scheduler/cache work."""
    from fni8serve.engine.llm_engine import LLMEngine

    eng = object.__new__(LLMEngine)
    eng.max_len = 4
    with pytest.raises(ValueError, match="max_len"):
        eng.add_request([1, 2, 3, 4, 5])


# --- S3: request-size caps --------------------------------------------------


def _jsonl(lines):
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode("utf-8")


def test_oversized_file_upload_rejected():
    from fni8serve.api.batches import _MAX_UPLOAD_BYTES

    with _client() as c:
        big = b"x" * (_MAX_UPLOAD_BYTES + 1024)
        resp = c.post(
            "/v1/files",
            files={"file": ("big.jsonl", big, "application/jsonl")},
            data={"purpose": "batch"},
        )
        assert resp.status_code == 413, resp.text


def test_oversized_batch_line_count_rejected():
    from fni8serve.api.batches import _MAX_BATCH_LINES

    with _client() as c:
        lines = [
            {
                "custom_id": str(i),
                "url": "/v1/completions",
                "body": {"model": "fake", "prompt": "a", "max_tokens": 1},
            }
            for i in range(_MAX_BATCH_LINES + 1)
        ]
        fid = c.post(
            "/v1/files",
            files={"file": ("b.jsonl", _jsonl(lines), "application/jsonl")},
            data={"purpose": "batch"},
        ).json()["id"]
        resp = c.post("/v1/batches", json={"input_file_id": fid, "endpoint": "/v1/completions"})
        assert resp.status_code == 413, resp.text


def test_data_uri_image_size_capped():
    pytest.importorskip("PIL")
    from fni8serve.multimodal import _MAX_IMAGE_BYTES, fetch_image

    # A base64 payload whose decoded length exceeds the cap must be rejected before PIL.
    huge = base64.b64encode(b"\x00" * (_MAX_IMAGE_BYTES + 1024)).decode()
    with pytest.raises(ValueError, match="cap"):
        fetch_image("data:image/png;base64," + huge)
