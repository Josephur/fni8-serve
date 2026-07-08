# SPDX-License-Identifier: MIT
"""FastAPI app exposing an OpenAI-compatible surface over an `LLMEngine`.

`/v1/chat/completions` and `/v1/completions` support both streaming (SSE, the
`text/event-stream` framing the `openai` client expects) and non-streaming
responses. Chat formatting goes through the tokenizer's own `apply_chat_template`
(Jinja2, sandboxed by `transformers` itself via
`jinja2.sandbox.ImmutableSandboxedEnvironment`), not a hardcoded template, so
tool/template formatting comes free per model. `chat_template` optionally overrides
the model's embedded template (a Jinja source string) with a caller-supplied one --
same mechanism vLLM's `--chat-template` uses, and the same sandboxed environment, so
a custom template gets no more code-execution power than the model's own.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ..engine.sequence import SamplingParams
from ..structured import GrammarCompilerCache
from .runtime import Done, EngineWorker
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    ModelCard,
    ModelList,
    ResponseFormat,
    UsageInfo,
)


def create_app(engine, tokenizer, *, served_model_name: str,
               chat_template: str | None = None) -> FastAPI:
    """`engine` needs `.add_request`, `.step`, `.sequence`, `.forget`, `.eos_id`
    (an `LLMEngine`, or a test double with the same seam). `tokenizer` needs
    `.apply_chat_template`, `.encode`, `.decode` (an HF `PreTrainedTokenizerBase`).
    `chat_template`, if given, overrides the tokenizer's own embedded chat template
    (e.g. loaded from a file by the `server.py` CLI's `--chat-template` flag)."""
    app = FastAPI(title="fni8-serve", version="0.0.1")
    worker = EngineWorker(engine)
    grammars = GrammarCompilerCache()

    def _prompt_ids(messages: list[ChatMessage]) -> list[int]:
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        return tokenizer.apply_chat_template(
            [m.model_dump(exclude_none=True) for m in messages], **kwargs)

    def _logit_processors(
        response_format: ResponseFormat | None, grammar: str | None,
    ) -> list | None:
        """Builds this request's structured-output `LogitsProcessor` (issue #39), if
        any -- `grammar` wins over `response_format` when both are given."""
        if grammar is not None:
            return [grammars.for_grammar(tokenizer, grammar)]
        if response_format is None or response_format.type == "text":
            return None
        if response_format.type == "json_object":
            return [grammars.for_json_object(tokenizer)]
        schema = response_format.json_schema.schema_ if response_format.json_schema else None
        return [grammars.for_json_schema(tokenizer, schema or {})]

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=served_model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        prompt_ids = _prompt_ids(req.messages)
        params = _sampling_params(req.temperature, req.top_p, req.max_tokens,
                                   _logit_processors(req.response_format, req.grammar))

        if req.stream:
            return StreamingResponse(
                _chat_stream(worker, tokenizer, prompt_ids, params, req.model),
                media_type="text/event-stream",
            )

        output_ids, reason = await _generate(worker, prompt_ids, params)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return ChatCompletionResponse(
            model=req.model,
            choices=[ChatCompletionResponseChoice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason=reason,
            )],
            usage=UsageInfo(prompt_tokens=len(prompt_ids), completion_tokens=len(output_ids),
                            total_tokens=len(prompt_ids) + len(output_ids)),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        prompt_ids = tokenizer.encode(prompt)
        params = _sampling_params(req.temperature, req.top_p, req.max_tokens,
                                   _logit_processors(req.response_format, req.grammar))

        if req.stream:
            return StreamingResponse(
                _completion_stream(worker, tokenizer, prompt_ids, params, req.model),
                media_type="text/event-stream",
            )

        output_ids, reason = await _generate(worker, prompt_ids, params)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return CompletionResponse(
            model=req.model,
            choices=[CompletionResponseChoice(text=text, finish_reason=reason)],
            usage=UsageInfo(prompt_tokens=len(prompt_ids), completion_tokens=len(output_ids),
                            total_tokens=len(prompt_ids) + len(output_ids)),
        )

    return app


def _sampling_params(
    temperature: float, top_p: float, max_tokens: int | None, logit_processors: list | None = None,
) -> SamplingParams:
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens or 16,
                           logit_processors=logit_processors)


async def _generate(
    worker: EngineWorker, prompt_ids: list[int], params: SamplingParams,
) -> tuple[list[int], str]:
    ids: list[int] = []
    reason = "stop"
    async for item in worker.stream(prompt_ids, params):
        if isinstance(item, Done):
            reason = item.reason
        else:
            ids.append(item)
    return ids, reason


async def _chat_stream(worker, tokenizer, prompt_ids, params, model) -> AsyncIterator[str]:
    first, prev_text, ids = True, "", []
    async for item in worker.stream(prompt_ids, params):
        if isinstance(item, Done):
            choice = ChatCompletionChunkChoice(delta=DeltaMessage(), finish_reason=item.reason)
            chunk = ChatCompletionChunk(model=model, choices=[choice])
            yield f"data: {chunk.model_dump_json()}\n\n"
            break
        ids.append(item)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        delta = text[len(prev_text):]
        prev_text = text
        if delta:
            chunk = ChatCompletionChunk(
                model=model,
                choices=[ChatCompletionChunkChoice(
                    delta=DeltaMessage(role="assistant" if first else None, content=delta))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            first = False
    yield "data: [DONE]\n\n"


async def _completion_stream(worker, tokenizer, prompt_ids, params, model) -> AsyncIterator[str]:
    prev_text, ids = "", []
    async for item in worker.stream(prompt_ids, params):
        if isinstance(item, Done):
            chunk = CompletionChunk(
                model=model, choices=[CompletionChunkChoice(text="", finish_reason=item.reason)])
            yield f"data: {chunk.model_dump_json()}\n\n"
            break
        ids.append(item)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        delta = text[len(prev_text):]
        prev_text = text
        if delta:
            chunk = CompletionChunk(model=model, choices=[CompletionChunkChoice(text=delta)])
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
