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

`/v1/files` + `/v1/batches` (issue #41) mount alongside these on the same
`EngineWorker`, so offline batch traffic and live HTTP traffic share one
continuous-batching loop instead of contending for the engine.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ..structured import GrammarCompilerCache
from ..tool_calls import forced_tool_schema
from .batches import register_batch_routes
from .request_helpers import (
    build_logit_processors,
    chat_prompt_ids,
    forced_tool_message,
    parsed_tool_message,
    sampling_params,
    tool_choice_forced,
)
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
    NamedToolChoice,
    UsageInfo,
)


def create_app(engine, tokenizer, *, served_model_name: str,
               chat_template: str | None = None, tool_parser: str = "hermes") -> FastAPI:
    """`engine` needs `.add_request`, `.step`, `.sequence`, `.forget`, `.eos_id`
    (an `LLMEngine`, or a test double with the same seam). `tokenizer` needs
    `.apply_chat_template`, `.encode`, `.decode` (an HF `PreTrainedTokenizerBase`).
    `chat_template`, if given, overrides the tokenizer's own embedded chat template
    (e.g. loaded from a file by the `server.py` CLI's `--chat-template` flag).
    `tool_parser` (issue #40) names the `fni8serve.tool_calls` parser used to
    extract `tool_calls` from free-form generation when `tool_choice` is `"auto"`
    (the default once `tools` is set) -- irrelevant for a forced `tool_choice`,
    which is schema-constrained instead (see `request_helpers.forced_tool_message`)."""
    app = FastAPI(title="fni8-serve", version="0.0.1")
    worker = EngineWorker(engine)
    grammars = GrammarCompilerCache()

    def _logit_processors(response_format, grammar):
        return build_logit_processors(grammars, tokenizer, response_format, grammar)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=served_model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        wants_tools = bool(req.tools) and req.tool_choice != "none"
        tools_wire = [t.model_dump() for t in req.tools] if wants_tools else None

        logit_processors = _logit_processors(req.response_format, req.grammar)
        forced = wants_tools and tool_choice_forced(req.tool_choice)
        if forced:
            tool_choice = (req.tool_choice.model_dump()
                          if isinstance(req.tool_choice, NamedToolChoice) else req.tool_choice)
            _, schema = forced_tool_schema(tools_wire, tool_choice)
            logit_processors = [grammars.for_json_schema(tokenizer, schema)]

        prompt_ids = chat_prompt_ids(tokenizer, req.messages, chat_template, tools=tools_wire)
        params = sampling_params(req.temperature, req.top_p, req.max_tokens, logit_processors)

        if req.stream:
            return StreamingResponse(
                _chat_stream(worker, tokenizer, prompt_ids, params, req.model),
                media_type="text/event-stream",
            )

        output_ids, reason = await _generate(worker, prompt_ids, params)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)

        if forced:
            message, reason = forced_tool_message(text)
        elif wants_tools:
            message, reason = parsed_tool_message(text, reason, tool_parser)
        else:
            message = ChatMessage(role="assistant", content=text)

        return ChatCompletionResponse(
            model=req.model,
            choices=[ChatCompletionResponseChoice(message=message, finish_reason=reason)],
            usage=UsageInfo(prompt_tokens=len(prompt_ids), completion_tokens=len(output_ids),
                            total_tokens=len(prompt_ids) + len(output_ids)),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        prompt_ids = tokenizer.encode(prompt)
        params = sampling_params(req.temperature, req.top_p, req.max_tokens,
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

    register_batch_routes(app, worker, tokenizer, grammars, chat_template=chat_template,
                          tool_parser=tool_parser)

    return app


async def _generate(
    worker: EngineWorker, prompt_ids: list[int], params,
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
