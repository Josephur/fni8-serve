# SPDX-License-Identifier: MIT
"""Regression tests for honest prefill/decode benchmark accounting."""

from bench.metrics import summarize_generation_steps


def test_generation_summary_does_not_call_prefill_decode_throughput():
    summary = summarize_generation_steps([4.0, 0.2, 0.020, 0.018, 0.019], warmup_decode_steps=1)

    assert summary["prefill_s"] == 4.0
    assert summary["graph_capture_s"] == 0.2
    assert summary["steady_decode_s"] == [0.020, 0.018, 0.019]
    assert abs(summary["steady_decode_tok_s"] - (1.0 / 0.019)) < 1e-9
    assert summary["end_to_end_tok_s"] == 5 / sum([4.0, 0.2, 0.020, 0.018, 0.019])


def test_generation_summary_handles_short_runs():
    summary = summarize_generation_steps([1.0], warmup_decode_steps=2)
    assert summary["prefill_s"] == 1.0
    assert summary["graph_capture_s"] == 0.0
    assert summary["steady_decode_s"] == []
    assert summary["steady_decode_tok_s"] == 0.0
