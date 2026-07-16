# SPDX-License-Identifier: MIT
"""Shared, format-neutral generation benchmark accounting."""

from __future__ import annotations

import statistics


def summarize_generation_steps(
    step_times_s: list[float], *, warmup_decode_steps: int = 1
) -> dict[str, object]:
    """Separate prefill, graph-capture, steady decode, and end-to-end rates.

    An engine generation emits its first token during the first (prefill) step.
    The first decode step commonly performs lazy CUDA-graph capture, so it must not
    be reported as steady decode.  End-to-end throughput remains available under an
    explicit name and includes every step.
    """
    if not step_times_s:
        raise ValueError("step_times_s must contain at least the prefill step")
    if warmup_decode_steps < 0:
        raise ValueError("warmup_decode_steps must be non-negative")

    prefill_s = step_times_s[0]
    capture = step_times_s[1 : 1 + warmup_decode_steps]
    steady = step_times_s[1 + warmup_decode_steps :]
    median_decode_s = statistics.median(steady) if steady else 0.0
    total_s = sum(step_times_s)
    return {
        "prefill_s": prefill_s,
        "graph_capture_s": sum(capture),
        "steady_decode_s": steady,
        "steady_decode_tok_s": (1.0 / median_decode_s) if median_decode_s else 0.0,
        "end_to_end_tok_s": len(step_times_s) / total_s if total_s else 0.0,
        "total_s": total_s,
    }
