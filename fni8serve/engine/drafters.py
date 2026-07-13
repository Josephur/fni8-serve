# SPDX-License-Identifier: MIT
"""Speculative-decode drafters — cheap proposers that feed the shared MTP verify
path. Two are cascaded (n-gram first, MTP fallback) by the engine runner.

The verify forward is a single weight-stream over ``k+1`` tokens; acceptance-length
(how many drafts survive) is what divides that stream's cost across emitted tokens.
A drafter's only job is to make the accepted prefix as long as possible for free
(no full-model forward):

  * :class:`NgramDrafter` — prompt-lookup / n-gram. Matches the last ``n`` tokens
    against everything generated so far and proposes the continuation that followed
    the most recent earlier match. On structured / repetitive spans (code, tool-call
    JSON, RAG quotes, edits) this proposes a long correct run at zero model cost, so
    acceptance-length climbs well past MTP's depth-1 (~2). A pure table lookup — no
    parameters, no device work.

  * The MTP head (``model.mtp``) is the fallback: on non-repetitive prose the n-gram
    lookup misses and the learned depth-1 head still lands ~90%. The engine runner
    cascades them: try n-gram, fall back to MTP on a miss (:func:`cascade_draft`).
"""

from __future__ import annotations

import os


class NgramDrafter:
    """Prompt-lookup n-gram drafter (Saxena 2023, "prompt lookup decoding").

    ``propose(tokens, k)`` returns up to ``k`` continuation tokens by finding the
    most recent earlier occurrence of the last ``n`` tokens (``n`` swept high→low so
    the longest, most specific match wins) and returning what followed it. Empty
    list on a miss. The match is searched over the sequence's OWN tokens so far
    (prompt + generated) — the span that actually repeats in code/JSON/editing.
    """

    def __init__(self, min_n: int = 2, max_n: int = 3, max_k: int = 8):
        # min_n: shortest pattern we trust (n==1 matches far too loosely and drafts
        #   noise); max_n: longest pattern we bother trying (diminishing returns).
        # max_k: cap on proposed continuation length (the verify cost ceiling).
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)
        self.max_k = max_k

    def propose(self, tokens: list[int], k: int) -> list[int]:
        """Return up to ``min(k, max_k)`` draft tokens continuing ``tokens``.

        Sweeps pattern length ``n`` from ``max_n`` down to ``min_n``; for each, finds
        the LAST index ``i`` (most recent) with ``tokens[i:i+n] == tokens[-n:]`` and
        ``i+n < len(tokens)`` (so a continuation exists) and returns
        ``tokens[i+n : i+n+k]``. Longest match first = most context = best drafts."""
        k = min(k, self.max_k)
        if k <= 0:
            return []
        L = len(tokens)
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            pattern = tokens[-n:]
            # Scan backward for the most recent earlier occurrence (skip the trailing
            # pattern itself: search window ends at L-n-1's start, i.e. i <= L-n-1).
            for i in range(L - n - 1, -1, -1):
                if tokens[i : i + n] == pattern:
                    cont = tokens[i + n : i + n + k]
                    if cont:
                        return cont
                    break  # match with no continuation room; try a shorter n
        return []


def cascade_draft(ngram, mtp_fallback, tokens, k):
    """Cascade: n-gram first (free, long on structured spans), else the MTP head.

    ``ngram`` is a :class:`NgramDrafter` (or None to skip straight to MTP);
    ``mtp_fallback`` is a zero-arg callable returning the MTP head's draft list
    (evaluated lazily — only on an n-gram miss, so the head's forward is skipped
    whenever the free lookup hits). Returns the draft token list (possibly empty)."""
    if ngram is not None:
        drafts = ngram.propose(tokens, k)
        if drafts:
            return drafts
    return mtp_fallback() if mtp_fallback is not None else []


def drafter_config():
    """Read spec-decode drafter config from the environment (engine kwargs override).

    FNI8SERVE_SPEC_DRAFTER: ``cascade`` (default) | ``ngram`` | ``mtp``.
    FNI8SERVE_SPEC_K:       max draft length (default 4). n-gram uses up to this;
                            the MTP head is depth-1 so it caps the effective k at 1
                            on prose. Larger k helps structured workloads, costs a
                            few wasted verify slots on prose.
    FNI8SERVE_SPEC_NGRAM:   ``min_n,max_n`` (default ``2,3``).
    """
    mode = os.environ.get("FNI8SERVE_SPEC_DRAFTER", "cascade").lower()
    k = int(os.environ.get("FNI8SERVE_SPEC_K", "4"))
    ng = os.environ.get("FNI8SERVE_SPEC_NGRAM", "2,3")
    try:
        min_n, max_n = (int(x) for x in ng.split(","))
    except ValueError:
        min_n, max_n = 2, 3
    return mode, max(1, k), min_n, max_n
