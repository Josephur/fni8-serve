# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP speculative-decode: head wiring + the safety guard.

Two tiers:
  * CPU-only: the `EngineRunner._spec_decode_allowed` guard predicate. This is the
    load-bearing safety contract — spec-decode MUST be refused for a hybrid
    (recurrent DeltaNet) model and for any image-carrying batch, because the
    multi-token verify forward is unimplemented for the gated/linear mixers and
    would corrupt DeltaNet recurrent state (and mixing it with vision is the exact
    interaction that broke in llama.cpp).
  * GPU + checkpoint (skipped otherwise): the real Qwen3.5-0.8B `.fni8` — the MTP
    head is wired and present, guarded text decode equals plain decode, and an
    image request stays correct (red -> "red") with the MTP head present.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fni8")

import torch
import torch.nn as nn

from fni8serve.engine.model_runner import EngineRunner
from fni8serve.engine.sequence import SamplingParams, Sequence

CUDA = torch.cuda.is_available()
FNI8 = os.environ.get("QWEN35_VL_FNI8", "/models/Qwen3.5-0.8B-fni8/Qwen__Qwen3.5-0.8B.b8.fni8")
TOK = os.environ.get("QWEN35_VL_TOK", "/models/Qwen3.5-0.8B-tok")


# ── CPU-only: the guard predicate ────────────────────────────────────────────


class _PlainModel(nn.Module):
    """A non-recurrent model (plain attention) — spec-decode is safe here."""


class _RecurrentInner(nn.Module):
    is_recurrent = True


class _HybridModel(nn.Module):
    """A model carrying a recurrent (DeltaNet-style) mixer — spec-decode unsafe."""

    def __init__(self):
        super().__init__()
        self.mixer = _RecurrentInner()


def _runner(model) -> EngineRunner:
    # device="cpu", no CUDA graph -> __init__ builds no GPU state.
    return EngineRunner(model, cache=object(), device="cpu", enable_cuda_graph=False)


def _seq(pixel_values=None, temperature=0.0) -> Sequence:
    s = Sequence(0, [1, 2, 3], SamplingParams(temperature=temperature, max_tokens=8))
    s.pixel_values = pixel_values
    return s


_MTP = object()  # sentinel: the guard only checks `mtp is None`


def test_guard_allows_plain_greedy_text():
    r = _runner(_PlainModel())
    assert r.has_recurrent is False
    assert r._spec_decode_allowed([_seq()], _MTP) is True


def test_guard_refuses_when_no_mtp_head():
    r = _runner(_PlainModel())
    assert r._spec_decode_allowed([_seq()], None) is False


def test_guard_refuses_nonzero_temperature():
    r = _runner(_PlainModel())
    assert r._spec_decode_allowed([_seq(temperature=0.7)], _MTP) is False


def test_guard_refuses_recurrent_hybrid_model():
    """Qwen3.5 is a DeltaNet+gated hybrid: the verify forward advances recurrent
    state by every drafted token with no rollback, so spec-decode must be refused."""
    r = _runner(_HybridModel())
    assert r.has_recurrent is True
    assert r._spec_decode_allowed([_seq()], _MTP) is False


def test_guard_refuses_image_batch():
    """MTP x vision guard: any sequence carrying pixel_values disables spec-decode,
    even on an otherwise-safe non-recurrent model."""
    r = _runner(_PlainModel())
    img = torch.zeros(1, 3, 16, 16)
    assert r._spec_decode_allowed([_seq(pixel_values=img)], _MTP) is False
    # ragged batch: one plain seq + one image seq -> still refused
    assert r._spec_decode_allowed([_seq(), _seq(pixel_values=img)], _MTP) is False


# ── GPU + checkpoint: real Qwen3.5-0.8B end-to-end ───────────────────────────

_have_ckpt = os.path.exists(FNI8) and os.path.exists(f"{TOK}/config.json")
gpu_ckpt = pytest.mark.skipif(
    not (CUDA and _have_ckpt), reason="needs CUDA + Qwen3.5-0.8B .fni8 + tokenizer dir"
)


@pytest.fixture(scope="module")
def qwen35_vl_engine():
    import json

    from fni8serve.engine.llm_engine import LLMEngine
    from fni8serve.loader import checkpoint_info, load_fni8_state_dict
    from fni8serve.models.config import ModelConfig

    meta_cfg = dict(checkpoint_info(FNI8)["meta"]["config"])
    hf_cfg = json.load(open(f"{TOK}/config.json"))
    meta_cfg["vision_config"] = hf_cfg["vision_config"]
    meta_cfg["image_token_id"] = hf_cfg["image_token_id"]
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5_vl")
    weights = load_fni8_state_dict(FNI8, device="cuda")
    eng = LLMEngine(
        cfg, weights, device="cuda", max_num_seqs=2, max_len=512, enable_cuda_graph=False
    )
    return eng, cfg


@gpu_ckpt
def test_mtp_head_present(qwen35_vl_engine):
    """The MTP head is wired onto the VLM wrapper and recovered from the weights
    even though the shipped .fni8 meta baked in num_mtp_layers=0."""
    eng, _ = qwen35_vl_engine
    mtp = getattr(eng.runner.model, "mtp", None)
    assert mtp is not None
    assert mtp.num_depths() == 1
    # This is a hybrid model, so the guard must keep spec-decode OFF.
    assert eng.runner.has_recurrent is True


@gpu_ckpt
def test_spec_decode_guarded_equals_plain_decode(qwen35_vl_engine):
    """With the MTP head present but guarded off (hybrid), greedy decode must be
    bit-identical to decode with the head removed — i.e. the guard truly falls back
    to plain decode, never engaging the (unsafe) verify path."""
    from transformers import AutoTokenizer

    eng, _ = qwen35_vl_engine
    tok = AutoTokenizer.from_pretrained(TOK)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Count from one to five in words."}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    prompt = [int(x) for x in ids]
    params = SamplingParams(temperature=0.0, max_tokens=24)

    out_mtp = eng.generate([prompt], params)[0]
    eng.runner.model.mtp = None
    out_none = eng.generate([prompt], params)[0]
    eng.runner.model.mtp = eng.runner.model.lm.mtp  # restore
    assert out_mtp == out_none, "guarded MTP decode diverged from plain decode"


@gpu_ckpt
def test_image_with_mtp_stays_correct(qwen35_vl_engine):
    """CRITICAL (llama.cpp regression): a red image through the engine, greedy, with
    the MTP head present, must still answer 'red' — and the guard must refuse
    spec-decode for the image-carrying sequence."""
    pytest.importorskip("torchvision")  # HF Qwen3.5 image processor needs it
    from transformers import AutoProcessor
    from PIL import Image

    eng, _ = qwen35_vl_engine
    proc = AutoProcessor.from_pretrained(TOK)
    tok = proc.tokenizer
    img = Image.new("RGB", (64, 64), "red")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What color is this image? Answer in one word."},
            ],
        }
    ]
    inp = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    img_ids = inp["input_ids"][0].tolist()

    assert getattr(eng.runner.model, "mtp", None) is not None
    sid = eng.add_request(img_ids, SamplingParams(temperature=0.0, max_tokens=8))
    seq = eng.sequence(sid)
    seq.pixel_values = inp["pixel_values"].cuda()
    seq.image_grid_thw = inp["image_grid_thw"].cuda()

    # The guard must refuse spec-decode for this image sequence even if the model
    # were (hypothetically) non-recurrent — pixel_values alone disables it.
    saved = eng.runner.has_recurrent
    eng.runner.has_recurrent = False
    assert eng.runner._spec_decode_allowed([seq], eng.runner.model.mtp) is False
    eng.runner.has_recurrent = saved

    while eng.scheduler.has_work():
        eng.step()
    ans = tok.decode(eng._out[sid].output_ids, skip_special_tokens=True).lower()
    eng.forget(sid)
    assert "red" in ans, f"image+MTP answer should contain 'red', got {ans!r}"
