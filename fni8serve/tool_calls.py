# SPDX-License-Identifier: MIT
"""OpenAI `tools`/`tool_choice` support (issue #40).

Two paths, matching the request's `tool_choice`:

- `"auto"` (the default once `tools` is set) or unset: `tools` is formatted into
  the prompt via the tokenizer's own `apply_chat_template(tools=...)` (HF's
  tools-aware templates, e.g. Qwen's, already render these; nothing
  fni8-serve-specific is needed there). The model free-generates, and a
  per-model `ToolCallParser` (registered in `TOOL_CALL_PARSERS`) extracts any
  `tool_calls` from the raw text afterwards.
- `"required"` or a named tool_choice: no parser needed, because generation
  itself is constrained. `forced_tool_schema` builds a JSON schema of
  `{"name": ..., "arguments": {...}}` (narrowed to the eligible tool(s)) and the
  caller compiles it through the existing XGrammar structured-output backend
  (`fni8serve.structured`, issue #39) the same way `response_format=
  {"type": "json_schema"}` does. The resulting text is guaranteed schema-valid
  JSON, so `parse_forced_tool_call` just does `json.loads`.

`HermesToolCallParser` implements the `<tool_call>{"name": ..., "arguments":
{...}}</tool_call>` convention Qwen 2.5/3 (and most Hermes-templated instruct
models) use for native tool calling. Its shape was studied from vLLM's
Apache-2.0-licensed `hermes_tool_parser.py` (see NOTICE) -- this is an
independent reimplementation of the same tag convention, not a port of that
file's code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict


@dataclass
class ExtractedToolCalls:
    content: str | None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


class ToolCallParser:
    """Extracts OpenAI-shape tool calls from a model's raw text completion."""

    def extract_tool_calls(self, text: str) -> ExtractedToolCalls:
        raise NotImplementedError


class HermesToolCallParser(ToolCallParser):
    """`<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>` -- one tag
    per call. Text before the first tag is kept as `content` (e.g. a model's
    preamble before it decides to call a tool). A call that isn't a JSON object,
    or lacks a `name` / has non-dict `arguments`, is dropped rather than raising
    -- one malformed call shouldn't sink an otherwise-valid response."""

    _CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def extract_tool_calls(self, text: str) -> ExtractedToolCalls:
        matches = list(self._CALL_RE.finditer(text))
        if not matches:
            return ExtractedToolCalls(content=text or None)

        calls = []
        for m in matches:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "name" not in obj:
                continue
            arguments = obj.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            calls.append(ParsedToolCall(name=obj["name"], arguments=arguments))

        content = text[:matches[0].start()].strip() or None
        return ExtractedToolCalls(content=content, tool_calls=calls)


TOOL_CALL_PARSERS: dict[str, type[ToolCallParser]] = {
    "hermes": HermesToolCallParser,
}


def get_tool_call_parser(name: str) -> ToolCallParser:
    try:
        return TOOL_CALL_PARSERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown tool_parser {name!r}; available: {sorted(TOOL_CALL_PARSERS)}"
        ) from None


def forced_tool_schema(tools: list[dict], tool_choice: str | dict) -> tuple[list[dict], dict]:
    """Builds the JSON schema that constrains generation to a single
    `{"name": ..., "arguments": {...}}` object, for `tool_choice="required"`
    (any of `tools`) or a named `tool_choice` (exactly that one). `tools` and
    `tool_choice` are plain dicts in the OpenAI wire shape (`ChatCompletionRequest`
    already validates that shape; this stays decoupled from the pydantic models).
    Returns the narrowed candidate tool list alongside the schema -- the caller
    doesn't need it to interpret the result (the generated JSON already names the
    tool), but it's the natural place to raise on an unknown `tool_choice` name.
    """
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        candidates = [t for t in tools if t["function"]["name"] == name]
        if not candidates:
            raise ValueError(f"tool_choice names unknown function {name!r}")
    else:
        candidates = tools

    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"const": t["function"]["name"]},
                    "arguments": t["function"].get("parameters") or {"type": "object"},
                },
                "required": ["name", "arguments"],
            }
            for t in candidates
        ]
    }
    return candidates, schema


def parse_forced_tool_call(text: str) -> ParsedToolCall:
    obj = json.loads(text)
    return ParsedToolCall(name=obj["name"], arguments=obj.get("arguments", {}))
