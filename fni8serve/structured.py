# SPDX-License-Identifier: MIT
"""Constrained decoding (issue #39): XGrammar as the backend behind the sampler's
per-step logit-processor hook (`fni8serve.layers.sampler`, issue #38). Compiles an
OpenAI `response_format={"type": "json_schema", ...}` schema, or a raw grammar
(GBNF/EBNF), into an XGrammar token-mask matcher, and exposes it as a
`LogitsProcessor`: `(input_ids, logits) -> logits`.

XGrammar's own `GrammarCompiler` already caches compiled grammars by schema/grammar
string; `GrammarCompilerCache` here caches the (expensive to build) `GrammarCompiler`
itself, one per tokenizer vocab. A `GrammarMatcher` carries per-sequence progress
through the grammar though, so each in-flight request gets its own
`GrammarLogitsProcessor` instance -- never share one across requests or sequences.
"""
from __future__ import annotations

import json

import torch

try:
    import xgrammar
except ImportError:  # pragma: no cover -- exercised via the missing-dependency path
    xgrammar = None


def xgrammar_available() -> bool:
    return xgrammar is not None


def _require_xgrammar() -> None:
    if xgrammar is None:
        raise RuntimeError(
            "response_format=json_schema / grammar needs the `xgrammar` package "
            "(pip install xgrammar, or `pip install -e '.[structured]'`)."
        )


class GrammarLogitsProcessor:
    """A `LogitsProcessor` backed by one `xgrammar.GrammarMatcher`. Stateful and
    scoped to a single request: construct a fresh instance per generation (via
    `GrammarCompilerCache.for_json_schema` / `.for_grammar`), never reuse across
    sequences -- the matcher's progress through the grammar IS that sequence's
    generation state.
    """

    def __init__(self, compiled_grammar) -> None:
        self._matcher = xgrammar.GrammarMatcher(compiled_grammar)
        self._seen: int | None = None
        vocab_size = compiled_grammar.tokenizer_info.vocab_size
        self._bitmask = xgrammar.allocate_token_bitmask(1, vocab_size)

    def __call__(self, input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        # `input_ids` is this sequence's full history so far (prompt + generated).
        # The grammar only constrains the generated continuation, so the first call
        # (at the prompt's last position, before any token is generated) just
        # anchors `_seen`; every later call advances the matcher by the one token
        # sampled since the previous step.
        if self._seen is None:
            self._seen = len(input_ids)
        else:
            for token_id in input_ids[self._seen:]:
                self._matcher.accept_token(token_id)
            self._seen = len(input_ids)

        self._matcher.fill_next_token_bitmask(self._bitmask)
        batched = logits.unsqueeze(0).clone()
        xgrammar.apply_token_bitmask_inplace(batched, self._bitmask)
        return batched.squeeze(0)


class GrammarCompilerCache:
    """One `xgrammar.GrammarCompiler` per tokenizer (building it walks the whole
    vocab, so it's cached); hands out a fresh per-request `GrammarLogitsProcessor`
    for a JSON schema or a raw grammar string."""

    def __init__(self) -> None:
        self._compilers: dict[int, "xgrammar.GrammarCompiler"] = {}

    def _compiler(self, tokenizer) -> "xgrammar.GrammarCompiler":
        _require_xgrammar()
        key = id(tokenizer)
        compiler = self._compilers.get(key)
        if compiler is None:
            tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
            compiler = xgrammar.GrammarCompiler(tokenizer_info)
            self._compilers[key] = compiler
        return compiler

    def for_json_schema(self, tokenizer, schema: str | dict) -> GrammarLogitsProcessor:
        if isinstance(schema, dict):
            schema = json.dumps(schema)
        compiled = self._compiler(tokenizer).compile_json_schema(schema)
        return GrammarLogitsProcessor(compiled)

    def for_json_object(self, tokenizer) -> GrammarLogitsProcessor:
        compiled = self._compiler(tokenizer).compile_builtin_json_grammar()
        return GrammarLogitsProcessor(compiled)

    def for_grammar(self, tokenizer, grammar: str) -> GrammarLogitsProcessor:
        compiled = self._compiler(tokenizer).compile_grammar(grammar)
        return GrammarLogitsProcessor(compiled)
