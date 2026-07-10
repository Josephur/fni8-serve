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

from starlette.testclient import TestClient  # noqa: E402  (sync client for the ASGI app)

from fni8serve.api.app import create_app  # noqa: E402
from fni8serve.api.schemas import ChatCompletionRequest  # noqa: E402
from fni8serve.engine.sequence import SamplingParams, Sequence, Status  # noqa: E402
from fni8serve.structured import GrammarCompilerCache  # noqa: E402

_VOCAB = {0: "Hello", 1: ",", 2: " world", 3: "!"}
EOS = 4

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


class FakeTokenizer:
    """A tiny stand-in for an HF `PreTrainedTokenizerBase`: fixed vocab, and an
    `apply_chat_template` that records exactly what the API layer passed it (the
    real Jinja rendering + sandboxing lives in `transformers` itself, upstream of
    this seam -- what we own, and what these tests check, is that our server calls
    it correctly, including threading through a caller-supplied template override,
    and the `tools` list for issue #40's tool-calling support)."""

    eos_token_id = EOS

    def __init__(self):
        self.chat_template_calls: list[str | None] = []
        self.tools_calls: list[list[dict] | None] = []

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, chat_template=None, tools=None
    ):
        assert messages[-1]["role"] == "user"
        self.chat_template_calls.append(chat_template)
        self.tools_calls.append(tools)
        return [10, 11, 12]

    def encode(self, text, **kw):
        return [10, 11, 12]

    def decode(self, ids, skip_special_tokens=True):
        kept = [t for t in ids if not (skip_special_tokens and t == EOS)]
        return "".join(_VOCAB.get(t, "") for t in kept)


class FixedTextTokenizer(FakeTokenizer):
    """A `FakeTokenizer` that decodes any non-empty token id sequence to a fixed
    string, so a tool-call test can drive an exact model output (e.g. a
    `<tool_call>...</tool_call>` tag, or schema-constrained JSON) without modeling
    a real subword vocabulary."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text if ids else ""


class FakeEngine:
    """Deterministic token stream, no CUDA/fni8 -- exercises `EngineWorker`'s
    continuous-batching loop and the HTTP layer without real weights."""

    def __init__(self):
        self.eos_id = EOS
        self.reply = [0, 1, 2, 3, EOS]
        self._seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()

    def encode(self, prompt_ids: list[int]) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    # TestClient drives the async ASGI app synchronously (httpx.Client + ASGITransport
    # can't — ASGITransport is async-only). Same .get/.post/.stream interface.
    with TestClient(app) as c:
        yield c


def test_list_models(http_client):
    resp = http_client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "fake-qwen3"


def test_chat_completions_non_streaming(http_client):
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
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
    with http_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    text = "".join(p["choices"][0]["delta"].get("content") or "" for p in payloads)
    assert text == "Hello, world!"
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_completions_streaming(http_client):
    with http_client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": "fake-qwen3",
            "prompt": "hi",
            "stream": True,
        },
    ) as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    text = "".join(p["choices"][0]["text"] for p in payloads)
    assert text == "Hello, world!"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_via_openai_client(http_client):
    """The `openai` client itself hits our server -- proves the response shape is
    actually OpenAI-compatible, not just shaped like our own schemas."""
    client = openai.OpenAI(
        api_key="unused", base_url="http://testserver/v1", http_client=http_client
    )
    resp = client.chat.completions.create(
        model="fake-qwen3", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "Hello, world!"
    assert resp.choices[0].finish_reason == "stop"


def test_completions_via_openai_client(http_client):
    client = openai.OpenAI(
        api_key="unused", base_url="http://testserver/v1", http_client=http_client
    )
    resp = client.completions.create(model="fake-qwen3", prompt="hi")
    assert resp.choices[0].text == "Hello, world!"


def test_chat_uses_the_tokenizers_own_template_by_default(http_client, tokenizer):
    http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert tokenizer.chat_template_calls == [None]


def test_response_format_json_schema_parses_openais_shape():
    """The `response_format` field (issue #39) must accept OpenAI's own
    `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}` shape --
    `schema` is a reserved BaseModel attribute name, so it's aliased to `schema_`."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}},
            },
        }
    )
    assert req.response_format.type == "json_schema"
    assert req.response_format.json_schema.name == "answer"
    assert req.response_format.json_schema.schema_ == {"type": "object"}


def test_grammar_extension_field_parses():
    req = ChatCompletionRequest.model_validate(
        {
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": 'root ::= "yes" | "no"',
        }
    )
    assert req.grammar == 'root ::= "yes" | "no"'


def test_tools_are_formatted_into_the_prompt(http_client, tokenizer):
    """`tools` (issue #40) must reach `apply_chat_template` as the `tools` kwarg,
    the same seam `chat_template` already uses -- HF's own tools-aware templates
    (Qwen, etc.) render it from there, so fni8-serve just needs to pass it through."""
    http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "tools": [WEATHER_TOOL],
        },
    )
    assert tokenizer.tools_calls == [[WEATHER_TOOL]]


def test_tool_choice_none_omits_tools_from_the_prompt(http_client, tokenizer):
    http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        },
    )
    assert tokenizer.tools_calls == [None]


def test_tools_present_but_model_replies_with_plain_text(http_client):
    """`tool_choice="auto"` (the default once `tools` is set): if the model just
    answers in plain text, the response must look exactly like the no-tools case
    -- no `tool_calls`, `finish_reason="stop"`."""
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEATHER_TOOL],
        },
    )
    message = resp.json()["choices"][0]["message"]
    assert message["content"] == "Hello, world!"
    assert message.get("tool_calls") is None
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


def test_tool_call_round_trip_via_hermes_parser():
    """The weather-tool round trip (issue #40): the model's raw text emits a
    Qwen-style `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` tag
    (the `hermes` parser, the default `--tool-parser`); the parsed call in the
    response must match it exactly."""
    tool_call_text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"location": "San Francisco"}}\n'
        "</tool_call>"
    )
    tokenizer = FixedTextTokenizer(tool_call_text)
    engine = FakeEngine()
    engine.reply = [0, EOS]
    app = create_app(engine, tokenizer, served_model_name="fake-qwen3")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-qwen3",
                "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
                "tools": [WEATHER_TOOL],
            },
        )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "San Francisco"}


def test_forced_tool_choice_builds_a_schema_and_parses_the_result(monkeypatch):
    """`tool_choice="required"` (issue #40) must drive generation through the
    existing structured-output backend (issue #39), constrained to the named
    tool's `parameters` schema, then parse the (guaranteed schema-valid) JSON
    straight into `tool_calls` -- no text parser involved. `for_json_schema` is
    mocked out here (its own correctness -- that XGrammar actually constrains
    generation to a schema -- is `test_structured.py`'s job); this test only
    proves the API layer wires tool_choice -> schema -> parsed result correctly,
    without requiring `xgrammar` to be installed."""
    built_schemas = []

    def fake_for_json_schema(self, tok, schema):
        built_schemas.append(schema)
        return lambda input_ids, logits: logits

    monkeypatch.setattr(GrammarCompilerCache, "for_json_schema", fake_for_json_schema)

    tokenizer = FixedTextTokenizer(
        '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'
    )
    engine = FakeEngine()
    engine.reply = [0, EOS]
    app = create_app(engine, tokenizer, served_model_name="fake-qwen3")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-qwen3",
                "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": "required",
            },
        )

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "San Francisco"}
    assert built_schemas[0]["oneOf"][0]["properties"]["name"]["const"] == "get_weather"


def test_named_tool_choice_forces_exactly_that_function(monkeypatch):
    monkeypatch.setattr(
        GrammarCompilerCache,
        "for_json_schema",
        lambda self, tok, schema: lambda input_ids, logits: logits,
    )

    tokenizer = FixedTextTokenizer('{"name": "get_weather", "arguments": {"location": "Berlin"}}')
    engine = FakeEngine()
    engine.reply = [0, EOS]
    app = create_app(engine, tokenizer, served_model_name="fake-qwen3")
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-qwen3",
                "messages": [{"role": "user", "content": "weather in Berlin?"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            },
        )

    assert resp.status_code == 200
    call = resp.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"


def test_embeddings_single_input(http_client):
    resp = http_client.post(
        "/v1/embeddings",
        json={
            "model": "fake-qwen3",
            "input": "Hello world",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert isinstance(body["data"][0]["embedding"], list)
    assert len(body["data"][0]["embedding"]) == 10
    assert body["model"] == "fake-qwen3"
    assert "usage" in body
    assert "prompt_tokens" in body["usage"]


def test_embeddings_batch_input(http_client):
    resp = http_client.post(
        "/v1/embeddings",
        json={
            "model": "fake-qwen3",
            "input": ["Hello world", "Goodbye"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["index"] == 0
    assert body["data"][1]["index"] == 1
    assert len(body["data"][0]["embedding"]) == 10
    assert len(body["data"][1]["embedding"]) == 10
    assert body["usage"]["prompt_tokens"] == 6  # 3 tokens per input


def test_embeddings_via_openai_client(http_client):
    client = openai.OpenAI(
        api_key="unused", base_url="http://testserver/v1", http_client=http_client
    )
    resp = client.embeddings.create(model="fake-qwen3", input="Hello world")
    assert len(resp.data) == 1
    assert len(resp.data[0].embedding) == 10
    assert resp.data[0].index == 0


def test_rerank(http_client):
    resp = http_client.post(
        "/v1/rerank",
        json={
            "model": "fake-qwen3",
            "query": "capital of France",
            "documents": ["Paris is the capital.", "London is the capital."],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    for r in body["data"]:
        assert "index" in r
        assert "relevance_score" in r
        assert isinstance(r["relevance_score"], float)
        assert "document" in r
        assert "text" in r["document"]
    assert body["model"] == "fake-qwen3"
    assert "usage" in body


def test_rerank_with_top_n(http_client):
    resp = http_client.post(
        "/v1/rerank",
        json={
            "model": "fake-qwen3",
            "query": "capital of France",
            "documents": [
                "Paris is the capital.",
                "London is the capital.",
                "Berlin is the capital.",
            ],
            "top_n": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 2


def test_rerank_sorted_by_score_descending(http_client):
    resp = http_client.post(
        "/v1/rerank",
        json={
            "model": "fake-qwen3",
            "query": "test",
            "documents": ["doc a", "doc b", "doc c"],
        },
    )
    scores = [r["relevance_score"] for r in resp.json()["data"]]
    assert scores == sorted(scores, reverse=True)


def test_image_url_content_part_parses():
    """``ContentPart`` with ``type="image_url"`` validates from the OpenAI wire
    shape (issue #151)."""
    from fni8serve.api.schemas import ChatMessage, ContentPart, ImageURL

    msg = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
    )
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "What's in this image?"
    assert msg.content[1].type == "image_url"
    assert msg.content[1].image_url is not None
    assert msg.content[1].image_url.url == "https://example.com/img.png"


def test_image_url_base64_data_uri_parses():
    """Base64 data URI is accepted as a valid ``image_url.url`` (issue #151)."""
    from fni8serve.api.schemas import ChatMessage

    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    msg = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    )
    assert isinstance(msg.content, list)
    assert msg.content[1].type == "image_url"
    assert msg.content[1].image_url is not None
    assert msg.content[1].image_url.url == data_uri


def test_message_dict_with_content_parts():
    """``message_dict`` must serialise a ``ChatMessage`` whose ``content`` is a
    list of ``ContentPart`` back into the OpenAI wire shape (list of dicts) for
    the chat template (issue #151)."""
    from fni8serve.api.request_helpers import message_dict
    from fni8serve.api.schemas import ChatMessage

    msg = ChatMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
    )
    d = message_dict(msg)
    assert d["role"] == "user"
    assert isinstance(d["content"], list)
    assert d["content"][0] == {"type": "text", "text": "What's in this image?"}
    assert d["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/img.png"},
    }


def test_fetch_image_base64_data_uri():
    """``fetch_image`` decodes a base64 data URI to a PIL Image (issue #151)."""
    from fni8serve.multimodal import fetch_image
    from PIL import Image

    # A 1×1 red pixel PNG
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    img = fetch_image(data_uri)
    assert isinstance(img, Image.Image)
    assert img.size == (1, 1)
    assert img.mode == "RGB"


def test_extract_images_from_messages_with_data_uri():
    """``extract_images_from_messages`` preprocesses a base64 data-URI image
    through ``preprocess_qwen2_5_vl`` and returns pixel_values + metadata
    (issue #151)."""
    from fni8serve.api.request_helpers import extract_images_from_messages
    from fni8serve.api.schemas import ChatMessage

    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    messages = [
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ),
    ]
    results = extract_images_from_messages(messages)
    assert len(results) == 1
    assert "pixel_values" in results[0]
    assert "image_grid_thw" in results[0]
    # pixel_values shape: [num_patches, C * temporal_patch_size * patch_size ** 2]
    assert results[0]["pixel_values"].ndim == 2
    # image_grid_thw is [1, 3] long tensor
    assert results[0]["image_grid_thw"].shape == (1, 3)


def test_chat_completions_with_image_url_via_openai_client(http_client):
    """A chat completion request with ``image_url`` content part is accepted and
    returns a valid response (issue #151). The image is a 1×1 PNG via base64 data
    URI. The model itself does not yet process images (no VLM integration); this
    test verifies the API layer doesn't reject image_url content."""
    import openai

    client = openai.OpenAI(
        api_key="unused", base_url="http://testserver/v1", http_client=http_client
    )

    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    resp = client.chat.completions.create(
        model="fake-qwen3",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    )
    assert resp.choices[0].message.content == "Hello, world!"


def test_chat_completions_with_image_url_non_streaming(http_client):
    """Non-streaming POST with image_url content parts (issue #151)."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello, world!"


def test_chat_completions_with_image_url_pixel_values_flow():
    """When a message contains an ``image_url`` content part, the processed
    ``pixel_values`` tensor must reach ``Sequence.pixel_values`` (issue #151).
    This test intercepts ``FakeEngine.forget`` to capture the sequence before
    it is cleaned up by the background engine thread."""
    import torch
    from fni8serve.api.app import create_app
    from starlette.testclient import TestClient

    captured_seqs = []
    orig_forget = FakeEngine.forget

    def tracking_forget(self, seq_id):
        seq = self._seqs.get(seq_id)
        if seq is not None and seq.pixel_values is not None:
            captured_seqs.append(seq)
        return orig_forget(self, seq_id)

    FakeEngine.forget = tracking_forget
    try:
        engine = FakeEngine()
        tokenizer = FakeTokenizer()
        app = create_app(engine, tokenizer, served_model_name="fake-qwen3")

        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake-qwen3",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image"},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        },
                    ],
                },
            )
        assert resp.status_code == 200
        assert len(captured_seqs) == 1, "Expected 1 sequence with pixel_values, got %d" % len(
            captured_seqs
        )
        assert captured_seqs[0].pixel_values is not None
        assert isinstance(captured_seqs[0].pixel_values, torch.Tensor)
    finally:
        FakeEngine.forget = orig_forget


def test_chat_completions_with_only_text_still_works(http_client):
    """A request with only text (no image parts) still works after the
    image_url changes (regression guard, issue #151)."""
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen3",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello, world!"


def test_chat_template_override_is_threaded_through(tokenizer):
    """`create_app(..., chat_template=...)` must reach `apply_chat_template` as the
    `chat_template` kwarg on every chat request -- the override seam `server.py`'s
    `--chat-template` flag uses, rendered through the same sandboxed Jinja
    environment `transformers` already applies to the model's own template."""
    custom = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    app = create_app(FakeEngine(), tokenizer, served_model_name="fake-qwen3", chat_template=custom)
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-qwen3",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert tokenizer.chat_template_calls == [custom]
