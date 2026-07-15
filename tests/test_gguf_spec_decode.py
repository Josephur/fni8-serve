# SPDX-License-Identifier: MIT
"""Speculative decode on the GGUF-native path (n-gram cascade + MTP-head detection).

Two layers:

  * CPU unit tests — the model-agnostic gate: n-gram spec-decode must engage on a
    model with NO MTP head (the common quantized-GGUF case, ``model.mtp is None``).
    The pre-fix gate refused all spec-decode unless an MTP head existed, which
    blocked the free n-gram lookup on every stripped GGUF. These construct an
    ``EngineRunner`` shell (``__new__`` + the handful of attrs the gate reads) so
    they run with no model / no CUDA.

  * GPU e2e tests (perf-marked, real Qwen3-8B-Q4_K_M) — the load-bearing proof:
    n-gram spec-decode produces BIT-IDENTICAL output to plain greedy decode, plus
    the accept-rate and the spec-vs-non-spec net tok/s. Skips without the GGUF/CUDA.
"""

from __future__ import annotations

import os

import pytest
import torch

from fni8serve.engine.drafters import NgramDrafter
from fni8serve.engine.model_runner import EngineRunner
from fni8serve.engine.sequence import SamplingParams, Sequence

_QWEN3_8B = (
    "/mnt/24tb/containers-archive/comfy/storage-models/models/"
    "text_encoders/Qwen 3/Qwen3-8B-Q4_K_M.gguf"
)
requires_qwen3 = pytest.mark.skipif(not os.path.exists(_QWEN3_8B), reason=f"missing {_QWEN3_8B}")


def _gate_runner(*, ngram, has_recurrent=False, spec_recurrent_ok=True):
    """An ``EngineRunner`` shell carrying only the attributes ``_spec_decode_allowed``
    reads — no model build, no CUDA. Mirrors the real runner's fields exactly."""
    r = EngineRunner.__new__(EngineRunner)
    r._ngram = ngram
    r.has_recurrent = has_recurrent
    r._spec_recurrent_ok = spec_recurrent_ok
    return r


def _greedy_seq():
    return Sequence(0, [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=8))


# ── gate: n-gram spec-decode must engage with NO MTP head ─────────────────────
def test_gate_allows_ngram_spec_without_mtp_head():
    """The core GGUF-native enablement: an MTP-less model (``mtp is None``) with an
    active n-gram drafter is a VALID spec-decode target — the lookup needs no head."""
    r = _gate_runner(ngram=NgramDrafter())
    assert r._spec_decode_allowed([_greedy_seq()], mtp=None) is True


def test_gate_blocks_when_no_drafter_at_all():
    """No MTP head AND no n-gram drafter (``FNI8SERVE_SPEC_DRAFTER=mtp`` on a model
    without a head) = nothing to propose → spec-decode must NOT engage (it would just
    verify base_tok every step, a pointless net slowdown)."""
    r = _gate_runner(ngram=None)
    assert r._spec_decode_allowed([_greedy_seq()], mtp=None) is False


def test_gate_still_blocks_non_greedy():
    """The accept-longest-greedy-prefix rule is only bit-identical under greedy; a
    sampled (temp>0) sequence must fall back to plain decode even with a drafter."""
    r = _gate_runner(ngram=NgramDrafter())
    hot = Sequence(0, [1, 2, 3], SamplingParams(temperature=0.7, max_tokens=8))
    assert r._spec_decode_allowed([hot], mtp=None) is False


def test_gate_blocks_image_sequence():
    """A text-only spec path must never verify over spliced image embeds."""
    r = _gate_runner(ngram=NgramDrafter())
    seq = _greedy_seq()
    seq.pixel_values = torch.zeros(1, 3, 8, 8)
    assert r._spec_decode_allowed([seq], mtp=None) is False


# ── MTP-head detection on the real GGUF (config level, no GPU) ────────────────
@pytest.mark.correctness
@requires_qwen3
def test_qwen3_8b_reports_mtp_absent():
    """Qwen3-8B-Q4_K_M ships no ``nextn.*`` (num_mtp_layers==0) → the loader must
    detect MTP absent and the drafter cascade falls back to n-gram."""
    from fni8serve.gguf_native import gguf_config

    cfg = gguf_config(_QWEN3_8B)
    assert cfg.num_mtp_layers == 0


# ── e2e: n-gram spec-decode is bit-identical to greedy + accept-rate + tok/s ──
@pytest.mark.perf
@requires_qwen3
def test_gguf_ngram_spec_bit_identical_and_accept_rate():
    """The deliverable: on a REAL GGUF LLM loaded natively, n-gram spec-decode
    (a) emits BYTE-IDENTICAL tokens to plain greedy decode and (b) reports a nonzero
    accept-rate + the net decode tok/s (spec vs non-spec). Needs CUDA + the 8B GGUF."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import time

    from fni8serve.gguf_native import load_gguf_engine

    # A repetitive prompt so the n-gram lookup has structured spans to ride.
    prompt = [
        3838, 374, 279, 6722, 315, 9625, 30,  # "What is the capital of France?"
        3838, 374, 279, 6722, 315, 9625, 30,
    ]
    n = 48
    params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    # -- baseline: plain greedy (spec OFF) --
    eng0 = load_gguf_engine(_QWEN3_8B, device="cuda", max_num_seqs=2, max_len=256)
    t0 = time.perf_counter()
    base = eng0.generate([list(prompt)], params)[0]
    dt0 = time.perf_counter() - t0
    del eng0
    torch.cuda.empty_cache()

    # -- spec ON (n-gram cascade; no MTP head on this GGUF) --
    eng1 = load_gguf_engine(
        _QWEN3_8B, device="cuda", max_num_seqs=2, max_len=256, spec_decode=True
    )
    assert getattr(eng1.model, "mtp", None) is None, "Qwen3-8B GGUF must have no MTP head"
    assert eng1.runner._ngram is not None, "n-gram drafter must be active"
    t1 = time.perf_counter()
    spec = eng1.generate([list(prompt)], params)[0]
    dt1 = time.perf_counter() - t1
    st = eng1.runner.spec_stats

    assert spec == base, (
        f"spec-decode output diverged from greedy:\n base[{len(base)}]={base}\n spec[{len(spec)}]={spec}"
    )
    accept_rate = st["accepts"] / st["drafts"] if st["drafts"] else 0.0
    assert st["steps"] > 0, "spec path never ran"
    assert st["drafts"] > 0, "n-gram never proposed a draft on a repetitive prompt"
    print(
        f"\n[gguf-spec] bit-identical={spec == base} "
        f"steps={st['steps']} drafts={st['drafts']} accepts={st['accepts']} "
        f"accept_rate={accept_rate:.3f}\n"
        f"[gguf-spec] tok/s incl. prefill: non-spec={n / dt0:.2f}  spec={n / dt1:.2f}  "
        f"(spec/non-spec={dt0 / dt1:.2f}x)"
    )
