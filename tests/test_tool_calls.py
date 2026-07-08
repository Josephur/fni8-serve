# SPDX-License-Identifier: MIT
"""`fni8serve.tool_calls` unit tests (issue #40): the Hermes-style
`<tool_call>...</tool_call>` text parser used for `tool_choice="auto"`, and the
JSON-schema builder that drives `tool_choice="required"` / a named tool_choice
through the existing structured-output backend (issue #39). The API-layer
round trip (through `/v1/chat/completions`) lives in `tests/test_api.py`; these
tests exercise the parsing/schema logic directly, without an app or engine."""
from __future__ import annotations

import json

import pytest

from fni8serve.tool_calls import (
    HermesToolCallParser,
    ParsedToolCall,
    forced_tool_schema,
    get_tool_call_parser,
    parse_forced_tool_call,
)

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

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}


def test_hermes_parser_extracts_a_single_call():
    text = ('<tool_call>\n'
           '{"name": "get_weather", "arguments": {"location": "San Francisco"}}\n'
           '</tool_call>')
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert extracted.content is None
    assert extracted.tool_calls == [
        ParsedToolCall(name="get_weather", arguments={"location": "San Francisco"})]


def test_hermes_parser_extracts_multiple_calls_in_order():
    text = (
        '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>\n'
        '<tool_call>{"name": "get_time", "arguments": {"timezone": "PT"}}</tool_call>'
    )
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert [c.name for c in extracted.tool_calls] == ["get_weather", "get_time"]
    assert extracted.tool_calls[1].arguments == {"timezone": "PT"}


def test_hermes_parser_keeps_leading_text_as_content():
    text = ('Let me check that for you.\n'
           '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>')
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert extracted.content == "Let me check that for you."
    assert extracted.tool_calls[0].name == "get_weather"


def test_hermes_parser_with_no_tags_returns_content_only():
    extracted = HermesToolCallParser().extract_tool_calls("just a plain reply")
    assert extracted.content == "just a plain reply"
    assert extracted.tool_calls == []


def test_hermes_parser_empty_text_has_no_content():
    extracted = HermesToolCallParser().extract_tool_calls("")
    assert extracted.content is None
    assert extracted.tool_calls == []


def test_hermes_parser_drops_malformed_json_but_keeps_valid_calls():
    text = (
        '<tool_call>{not valid json}</tool_call>'
        '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>'
    )
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].name == "get_weather"


def test_hermes_parser_drops_call_missing_a_name():
    text = '<tool_call>{"arguments": {"location": "SF"}}</tool_call>'
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == []


def test_get_tool_call_parser_returns_hermes_by_default():
    assert isinstance(get_tool_call_parser("hermes"), HermesToolCallParser)


def test_get_tool_call_parser_raises_on_unknown_name():
    with pytest.raises(ValueError, match="unknown tool_parser"):
        get_tool_call_parser("not-a-real-parser")


def test_forced_tool_schema_required_covers_every_tool():
    candidates, schema = forced_tool_schema([WEATHER_TOOL, TIME_TOOL], "required")
    assert candidates == [WEATHER_TOOL, TIME_TOOL]
    names = {branch["properties"]["name"]["const"] for branch in schema["oneOf"]}
    assert names == {"get_weather", "get_time"}


def test_forced_tool_schema_named_choice_narrows_to_one():
    tool_choice = {"type": "function", "function": {"name": "get_time"}}
    candidates, schema = forced_tool_schema([WEATHER_TOOL, TIME_TOOL], tool_choice)
    assert candidates == [TIME_TOOL]
    assert len(schema["oneOf"]) == 1
    assert schema["oneOf"][0]["properties"]["name"]["const"] == "get_time"
    assert schema["oneOf"][0]["properties"]["arguments"] == TIME_TOOL["function"]["parameters"]


def test_forced_tool_schema_unknown_named_choice_raises():
    tool_choice = {"type": "function", "function": {"name": "not_a_tool"}}
    with pytest.raises(ValueError, match="not_a_tool"):
        forced_tool_schema([WEATHER_TOOL], tool_choice)


def test_forced_tool_schema_validates_a_matching_call():
    jsonschema = pytest.importorskip("jsonschema")
    _, schema = forced_tool_schema([WEATHER_TOOL, TIME_TOOL], "required")
    jsonschema.validate({"name": "get_weather", "arguments": {"location": "SF"}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"name": "get_weather", "arguments": {"location": 5}}, schema)


def test_parse_forced_tool_call():
    parsed = parse_forced_tool_call(json.dumps({"name": "get_weather",
                                                "arguments": {"location": "SF"}}))
    assert parsed == ParsedToolCall(name="get_weather", arguments={"location": "SF"})
