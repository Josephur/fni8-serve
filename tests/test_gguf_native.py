# SPDX-License-Identifier: MIT
"""Native-GGUF loader tests (P1 of the GGUF-native migration).

Header-only, no GPU: read a REAL GGUF's KV metadata via `gguf.GGUFReader` and
assert `gguf_native.gguf_config` reproduces the same `ModelConfig` that
`ModelConfig.from_hf` builds from the model's HF `config.json` — dims, rope,
heads, the hybrid `full_attention_interval`, and the divergent SSM / MoE fields.

The reference numbers below are the AUTHORITATIVE HF `config.json` values for the
two on-disk GGUFs (verified against the HF configs at test-authoring time):

  * Qwen3.5-9B (hybrid gated-DeltaNet + full attn, DENSE) — the rich case that
    exercises the SSM/hybrid/partial-rope mapping.
  * Qwen3-8B (standard dense GQA) — the clean e2e target used by PR-3.

If the GGUFs are not present the model tests skip (they live outside the repo).
"""

from __future__ import annotations

import os

import pytest
import torch

from fni8serve.gguf_native import gguf_config, gguf_dit_config

# ── Real on-disk GGUFs (outside the repo; skip if absent) ────────────────────
_QWEN35_9B = "/mnt/24tb/containers-archive/qwen9b/Qwen3.5-9B-UD-Q6_K_XL.gguf"
_QWEN3_8B = (
    "/mnt/24tb/containers-archive/comfy/storage-models/models/"
    "text_encoders/Qwen 3/Qwen3-8B-Q4_K_M.gguf"
)
_LTX_DIT = "/mnt/24tb/containers-archive/wan2gp/ckpts/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"

_HAVE_GGUF = pytest.importorskip("gguf", reason="gguf lib required for native loader")

requires_qwen35 = pytest.mark.skipif(not os.path.exists(_QWEN35_9B), reason=f"missing {_QWEN35_9B}")
requires_qwen3 = pytest.mark.skipif(not os.path.exists(_QWEN3_8B), reason=f"missing {_QWEN3_8B}")
requires_ltx = pytest.mark.skipif(not os.path.exists(_LTX_DIT), reason=f"missing {_LTX_DIT}")


def test_gguf_engine_consumes_owned_weights(monkeypatch):
    """The one-shot GGUF load must free source rows while merging a card-filling model."""
    from types import SimpleNamespace

    import fni8serve.engine
    import fni8serve.gguf_native as native

    cfg = SimpleNamespace(
        arch="qwen3",
        hidden_size=4,
        linear_attention=False,
        qk_norm=False,
        num_mtp_layers=0,
    )
    weights = {"model.embed_tokens.weight": torch.zeros(8, 4)}
    seen = {}

    class FakeEngine:
        def __init__(self, passed_cfg, passed_weights, **kwargs):
            seen.update(cfg=passed_cfg, weights=passed_weights, kwargs=kwargs)

    monkeypatch.setattr(native, "_open_gguf", lambda _path: object())
    monkeypatch.setattr(native, "gguf_config", lambda _path, **_kwargs: cfg)
    monkeypatch.setattr(native, "gguf_state_dict", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(native, "_native_kquant_types", lambda: set())
    monkeypatch.setattr(native, "_gguf_eos_id", lambda _path, **_kwargs: 1)
    monkeypatch.setattr(fni8serve.engine, "LLMEngine", FakeEngine)

    native.load_gguf_engine("model.gguf", device="cpu")
    assert seen["kwargs"]["consume_weights"] is True


def test_gguf_engine_reuses_one_reader(monkeypatch):
    """A cold load must not parse the multi-GB GGUF header three separate times."""
    from types import SimpleNamespace

    import fni8serve.engine
    import fni8serve.gguf_native as native

    reader = object()
    opened = []
    observed = []
    cfg = SimpleNamespace(
        arch="qwen3",
        hidden_size=4,
        linear_attention=False,
        qk_norm=False,
        num_mtp_layers=0,
    )
    monkeypatch.setattr(native, "_open_gguf", lambda path: opened.append(path) or reader)
    monkeypatch.setattr(
        native,
        "gguf_config",
        lambda path, *, _reader=None: observed.append(("config", _reader)) or cfg,
    )
    monkeypatch.setattr(
        native,
        "gguf_state_dict",
        lambda path, *, device, native_types, _reader=None: (
            observed.append(("weights", _reader))
            or {"model.embed_tokens.weight": torch.zeros(8, 4)}
        ),
    )
    monkeypatch.setattr(
        native,
        "_gguf_eos_id",
        lambda path, *, _reader=None: observed.append(("eos", _reader)) or 1,
    )
    monkeypatch.setattr(native, "_native_kquant_types", lambda: ())
    monkeypatch.setattr(fni8serve.engine, "LLMEngine", lambda *args, **kwargs: object())

    native.load_gguf_engine("model.gguf", device="cpu")

    assert opened == ["model.gguf"]
    assert observed == [("config", reader), ("weights", reader), ("eos", reader)]


@pytest.mark.correctness
def test_qwen35_gguf_transforms_are_restored_to_hf_semantics():
    """llama.cpp rewrites Qwen3.5 tensors; the HF-layout builder needs the inverse."""
    from types import SimpleNamespace

    from fni8serve.gguf_native import _restore_hf_qwen35_weights

    cfg = SimpleNamespace(
        linear_attention=True,
        extra={"linear_num_key_heads": 2, "linear_num_value_heads": 4},
    )
    sd = {
        "model.layers.0.input_layernorm.weight": torch.tensor([1.25]),
        "model.layers.0.self_attn.q_norm.weight": torch.tensor([0.75]),
        "model.layers.0.linear_attn.norm.weight": torch.tensor([0.875]),
        "model.layers.0.linear_attn.A_log": torch.tensor([-torch.exp(torch.tensor(2.0))]),
    }

    restored, tiled = _restore_hf_qwen35_weights(sd, cfg)

    assert torch.equal(restored["model.layers.0.input_layernorm.weight"], torch.tensor([0.25]))
    assert torch.equal(restored["model.layers.0.self_attn.q_norm.weight"], torch.tensor([-0.25]))
    # The gated DeltaNet output norm is not zero-centered and is not shifted by the converter.
    assert torch.equal(restored["model.layers.0.linear_attn.norm.weight"], torch.tensor([0.875]))
    assert torch.allclose(restored["model.layers.0.linear_attn.A_log"], torch.tensor([2.0]))
    assert tiled is True


@pytest.mark.correctness
def test_native_kquant_capabilities_cover_every_installed_kernel(monkeypatch):
    """The loader must not silently transcode a format whose fused kernel exists."""
    import fni8

    from fni8serve.gguf_native import _native_kquant_types

    expected = {
        gguf_type
        for gguf_type, op in {
            "Q2_K": "linear_q2k",
            "Q3_K": "linear_q3k",
            "Q4_K": "linear_q4k",
            "Q5_K": "linear_q5k",
            "Q6_K": "linear_q6k",
        }.items()
        if callable(getattr(fni8, op, None))
    }
    assert set(_native_kquant_types()) == expected

    monkeypatch.setattr(fni8, "linear_q6k", None, raising=False)
    assert "Q6_K" not in _native_kquant_types()


@pytest.mark.correctness
@requires_qwen35
def test_qwen35_9b_config_matches_hf():
    """Qwen3.5-9B hybrid: every dim/rope/head/hybrid field maps from GGUF-KV, and
    the divergent SSM params round-trip into `extra` for the qwen3_next builder."""
    cfg = gguf_config(_QWEN35_9B)

    # arch: GGUF `qwen35` must map onto our registered builder key `qwen3_next`
    # (registry._resolve does NOT know `qwen35`; the GGUF-arch alias table does).
    from fni8serve.models.registry import _resolve

    assert _resolve(cfg.arch) == "qwen3_next", f"arch {cfg.arch!r} did not resolve to a builder"

    # ── dims (qwen35.* KV → ModelConfig) ──
    assert cfg.vocab_size == 248320  # len(tokenizer.ggml.tokens)
    assert cfg.hidden_size == 4096  # qwen35.embedding_length
    assert cfg.num_hidden_layers == 32  # qwen35.block_count
    assert cfg.num_attention_heads == 16  # qwen35.attention.head_count
    assert cfg.num_key_value_heads == 4  # qwen35.attention.head_count_kv
    assert cfg.intermediate_size == 12288  # qwen35.feed_forward_length
    assert cfg.resolved_head_dim() == 256  # qwen35.attention.key_length
    assert cfg.max_position_embeddings == 262144  # qwen35.context_length
    assert cfg.tie_word_embeddings is False  # output.weight present → untied

    # ── rope (qwen35.rope.* KV) ──
    assert cfg.rope_theta == 10000000.0  # qwen35.rope.freq_base
    # partial_rotary = rope.dimension_count / head_dim = 64 / 256 = 0.25
    assert abs(cfg.partial_rotary_factor - 0.25) < 1e-9
    assert cfg.rotary_dim() == 64
    assert abs(cfg.rms_norm_eps - 1e-6) < 1e-8  # qwen35.attention.layer_norm_rms_epsilon

    # ── hybrid layer schedule (qwen35.full_attention_interval + ssm.* present) ──
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    # every 4th layer full, rest linear (matches HF layer_types)
    assert cfg.attention_kind(0) == "linear"
    assert cfg.attention_kind(3) == "full"
    assert cfg.attention_kind(7) == "full"
    assert cfg.attention_kind(31) == "full"

    # ── divergent SSM / gated-DeltaNet params (qwen35.ssm.* → extra) ──
    # Sources cross-checked vs research/llamacpp-hybrid-linattn.md §1.5 (HF→GGUF):
    #   linear_key_head_dim   = ssm.state_size       (128)
    #   linear_num_key_heads  = ssm.group_count      (16)
    #   linear_num_value_heads= ssm.time_step_rank   (32)
    #   linear_conv_kernel_dim= ssm.conv_kernel      (4)
    #   linear_value_head_dim = ssm.inner_size / ssm.time_step_rank = 4096/32 = 128
    x = cfg.extra
    assert x["linear_key_head_dim"] == 128
    assert x["linear_num_key_heads"] == 16
    assert x["linear_num_value_heads"] == 32
    assert x["linear_conv_kernel_dim"] == 4
    assert x["linear_value_head_dim"] == 128

    # this GGUF dropped the MTP/nextn head (no qwen35.nextn_predict_layers) → 0,
    # a documented divergence from the HF config (§5c of MIGRATION).
    assert cfg.num_mtp_layers == 0


@pytest.mark.correctness
@requires_qwen3
def test_qwen3_8b_config_matches_hf():
    """Standard dense Qwen3 (the PR-3 e2e model): plain GQA, full rotary, no SSM."""
    cfg = gguf_config(_QWEN3_8B)
    from fni8serve.models.registry import _resolve

    assert _resolve(cfg.arch) == "qwen3"
    assert cfg.hidden_size == 4096  # qwen3.embedding_length
    assert cfg.num_hidden_layers == 36  # qwen3.block_count
    assert cfg.num_attention_heads == 32  # qwen3.attention.head_count
    assert cfg.num_key_value_heads == 8  # qwen3.attention.head_count_kv
    assert cfg.intermediate_size == 12288  # qwen3.feed_forward_length
    assert cfg.resolved_head_dim() == 128  # qwen3.attention.key_length
    assert cfg.rope_theta == 1000000.0  # qwen3.rope.freq_base
    # no qwen3.rope.dimension_count → full rotary
    assert abs(cfg.partial_rotary_factor - 1.0) < 1e-9
    assert cfg.linear_attention is False  # dense, no hybrid
    assert cfg.num_experts == 0  # dense, no MoE
    assert cfg.vocab_size > 150000  # from tokenizer.ggml.tokens


@pytest.mark.correctness
@requires_qwen35
def test_gguf_config_moe_gating_asserted_not_defaulted():
    """A silent softmax default for a sigmoid group-routing MoE = wrong experts =
    garbage output (MIGRATION §5b). Assert the resolver NEVER leaves a MoE config
    with an unresolved gate. (Qwen3.5-9B is dense → num_experts==0 → vacuously ok;
    this guards the invariant so a future MoE GGUF cannot slip through defaulted.)"""
    cfg = gguf_config(_QWEN35_9B)
    if cfg.num_experts > 0:
        # if experts exist, the gate func MUST be explicitly resolved into extra
        assert "expert_gating_func" in cfg.extra, "MoE config left gating func defaulted"


@pytest.mark.correctness
@requires_qwen3
def test_gguf_state_dict_schemes_native():
    """PR-2: k-quant weights land RESIDENT as native `gguf_kquant` (raw bytes, no
    transcode); norms/embeddings stay fp16 Tensors — the shapes the builders expect."""
    from fni8 import QTensor

    from fni8serve.gguf_native import gguf_state_dict

    sd = gguf_state_dict(_QWEN3_8B, device="cpu")
    # a Q4_K linear (o_proj) → native gguf_kquant raw bytes
    o = sd["model.layers.0.self_attn.o_proj.weight"]
    assert isinstance(o, QTensor) and o.scheme == "gguf_kquant" and o.codebook == "q4_k"
    assert o.scale is None and o.data.dtype == torch.uint8
    # QTensor invariant: [out, n_superblocks*144] (Q4_K type_size=144), no paired scale
    assert o.data.dim() == 2 and o.data.shape[1] % 144 == 0
    # a Q6_K linear (ffn_down) → native gguf_kquant q6_k
    d = sd["model.layers.0.mlp.down_proj.weight"]
    assert isinstance(d, QTensor) and d.scheme == "gguf_kquant" and d.codebook == "q6_k"
    # norms + embedding stay fp16 Tensors (not QTensors)
    assert not isinstance(sd["model.layers.0.input_layernorm.weight"], QTensor)
    assert sd["model.embed_tokens.weight"].dtype == torch.float16


@pytest.mark.perf
@requires_qwen3
def test_linear_gguf_kquant_matches_dequant():
    """PR-2 proof: LinearW8A8 over a native `gguf_kquant` weight matches a full-
    precision dequant reference at cos≥0.99 — the Q4_K fused dp4a path AND the
    Q5_K/Q6_K dequant fallback. Needs a CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from fni8 import QTensor

    from fni8serve.gguf_native import dequant_kquant, gguf_state_dict
    from fni8serve.layers.linear import LinearW8A8

    # load on CPU (the full 8B doesn't fit alongside other jobs), move only the two
    # tested weights to CUDA — keeps this a targeted per-linear correctness probe.
    sd = gguf_state_dict(_QWEN3_8B, device="cpu")
    torch.manual_seed(0)
    for name in ("model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight"):
        src = sd[name]
        qt = QTensor(
            src.data.cuda(), None, scheme="gguf_kquant", codebook=src.codebook, group_size=256
        )
        lin = LinearW8A8(qt).cuda()
        x = torch.randn(4, lin.in_features, device="cuda", dtype=torch.float16)
        y = lin(x)
        ref = torch.nn.functional.linear(x, dequant_kquant(qt).to(x.dtype))
        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        assert cos >= 0.99, f"{name} ({qt.codebook}) cos={cos:.4f} < 0.99"


@pytest.mark.perf
@requires_qwen3
def test_gguf_e2e_greedy_decode_zero_fni8():
    """PR-3 / the P1 milestone: greedily decode from a REAL GGUF LLM through the full
    fni8-serve engine (prefill + paged-KV decode) with ZERO `.fni8` — no .fni8 file,
    no .fni8 loader touched. Assert it runs end-to-end, the output is deterministic,
    and every emitted id is in-vocab. Needs a CUDA device with room for an 8B."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import time

    from fni8serve.engine.sequence import SamplingParams
    from fni8serve.gguf_native import load_gguf_engine

    eng = load_gguf_engine(_QWEN3_8B, device="cuda", max_num_seqs=2, max_len=256)
    assert eng.cfg.vocab_size > 150000
    prompt = [3838, 374, 279, 6722, 315, 9625, 30]  # arbitrary in-vocab Qwen3 ids
    n = 16
    params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)  # greedy

    t0 = time.perf_counter()
    out1 = eng.generate([prompt], params)[0]
    dt = time.perf_counter() - t0
    out2 = eng.generate([prompt], params)[0]

    assert len(out1) == n, f"expected {n} decoded tokens, got {len(out1)}"
    assert out1 == out2, "greedy decode must be deterministic across runs"
    assert all(0 <= t < eng.cfg.vocab_size for t in out1), "decoded ids out of vocab range"
    print(f"\n[e2e] Qwen3-8B GGUF greedy decode: {out1}  ({n / dt:.1f} tok/s incl. prefill)")


@pytest.mark.correctness
@requires_ltx
def test_dit_config_blob_parses():
    """DiT GGUFs (ComfyUI-GGUF convention) carry the diffusers config as one opaque
    JSON blob, not <arch>.* KV (MIGRATION §2c). `gguf_dit_config` must json-parse it."""
    blob = gguf_dit_config(_LTX_DIT)
    assert isinstance(blob, dict)
    # LTX-2 diffusers config nests the transformer knobs
    assert "transformer" in blob or "_class_name" in blob
    # and gguf_config must route DiT arches to that helper, not misparse as an LLM
    with pytest.raises(Exception):
        gguf_config(_LTX_DIT)  # LLM path can't build a DiT; must point to gguf_dit_config
