# SPDX-License-Identifier: MIT
"""API layer tests: HTTP routing, OpenAI response/SSE shapes, and chat-template
wiring, against a fake engine + fake tokenizer so this runs without CUDA/fni8/model
weights -- the engine's own correctness is covered in test_engine.py, and the
sampler's logit-processor hook in test_sampler.py."""
import itertools
import json

import pytest

pytest.importorskip("fastapi")
openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from fni8serve.api.app import create_app  # noqa: E402
from fni8serve.engine.sequence import SamplingParams, Sequence, Status  # noqa: E402

_VOCAB = {0: "Hello", 1: ",", 2: " world", 3: "!"}
EOS = 4


class FakeTokenizer:
    """A tiny stand-in for an HF `PreTrainedTokenizerBase`: fixed vocab, and an
    `apply_chat_template` that records exactly what the API layer passed it (the
    real Jinja rendering + sandboxing lives in `transformers` itself, upstream of
    this seam -- what we own, and what these tests check, is that our server calls
    it correctly, including threading through a caller-supplied template override)."""
    eos_token_id = EOS

    def __init__(self):
        self.chat_template_calls: list[str | None] = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            chat_template=None):
        assert messages[-1]["role"] == "user"
        self.chat_template_calls.append(chat_template)
        return [10, 11, 12]

    def encode(self, text, **kw):
        return [10, 11, 12]

    def decode(self, ids, skip_special_tokens=True):
        kept = [t for t in ids if not (skip_special_tokens and t == EOS)]
        return "".join(_VOCAB.get(t, "") for t in kept)


class FakeEngine:
    """Deterministic token stream, no CUDA/fni8 -- exercises `EngineWorker`'s
    continuous-batching loop and the HTTP layer without real weights."""

    def __init__(self):
        self.eos_id = EOS
        self.reply = [0, 1, 2, 3, EOS]
        self._seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()

    def add_request(self, prompt_ids, params: SamplingParams | None = None) -> int:
        seq_id = next(self._ids)
        self._seqs[seq_id] = Sequence(seq_id, list(prompt_ids), params or SamplingParams())
        return seq_id

    def step(self) -> None:
        for seq in self._seqs.values():
            if seq.status is Status.FINISHED:
                continue
            seq.output_ids.append(self.reply[len(seq.output_ids)])
            seq.status = Status.FINISHED if seq.is_finished(self.eos_id) else Status.RUNNING

    def sequence(self, seq_id: int) -> Sequence:
        return self._seqs[seq_id]

    def forget(self, seq_id: int) -> None:
        del self._seqs[seq_id]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def app(tokenizer):
    return create_app(FakeEngine(), tokenizer, served_model_name="fake-qwen3")


@pytest.fixture
def http_client(app):
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://testserver") as c:
        yield c


def test_list_models(http_client):
    resp = http_client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "fake-qwen3"


def test_chat_completions_non_streaming(http_client):
    resp = http_client.post("/v1/chat/completions", json={
        "model": "fake-qwen3",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello, world!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 5


def test_completions_non_streaming(http_client):
    resp = http_client.post("/v1/completions", json={"model": "fake-qwen3", "prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "Hello, world!"


def test_chat_completions_streaming(http_client):
    with http_client.stream("POST", "/v1/chat/completions", json={
        "model": "fake-qwen3",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: "):]) for line in lines[:-1]]
    text = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads)
    assert text == "Hello, world!"
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_completions_streaming(http_client):
    with http_client.stream("POST", "/v1/completions", json={
        "model": "fake-qwen3", "prompt": "hi", "stream": True,
    }) as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: "):]) for line in lines[:-1]]
    text = "".join(p["choices"][0]["text"] for p in payloads)
    assert text == "Hello, world!"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_via_openai_client(http_client):
    """The `openai` client itself hits our server -- proves the response shape is
    actually OpenAI-compatible, not just shaped like our own schemas."""
    client = openai.OpenAI(
        api_key="unused", base_url="http://testserver/v1", http_client=http_client)
    resp = client.chat.completions.create(
        model="fake-qwen3", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "Hello, world!"
    assert resp.choices[0].finish_reason == "stop"


def test_completions_via_openai_client(http_client):
    client = openai.OpenAI(
        api_key="unused", base_url="http://testserver/v1", http_client=http_client)
    resp = client.completions.create(model="fake-qwen3", prompt="hi")
    assert resp.choices[0].text == "Hello, world!"


def test_chat_uses_the_tokenizers_own_template_by_default(http_client, tokenizer):
    http_client.post("/v1/chat/completions", json={
        "model": "fake-qwen3", "messages": [{"role": "user", "content": "hi"}],
    })
    assert tokenizer.chat_template_calls == [None]


def test_chat_template_override_is_threaded_through(tokenizer):
    """`create_app(..., chat_template=...)` must reach `apply_chat_template` as the
    `chat_template` kwarg on every chat request -- the override seam `server.py`'s
    `--chat-template` flag uses, rendered through the same sandboxed Jinja
    environment `transformers` already applies to the model's own template."""
    custom = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    app = create_app(FakeEngine(), tokenizer, served_model_name="fake-qwen3",
                     chat_template=custom)
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://testserver") as client:
        client.post("/v1/chat/completions", json={
            "model": "fake-qwen3", "messages": [{"role": "user", "content": "hi"}],
        })
    assert tokenizer.chat_template_calls == [custom]
