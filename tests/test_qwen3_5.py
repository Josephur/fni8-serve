# SPDX-License-Identifier: MIT
"""Qwen3.5-9B (real `Qwen/Qwen3.5-9B`) — DENSE hybrid (Gated-DeltaNet linear + GATED
full attention).

Covers the three deliverable stages:
  * config: `from_hf` derives the right hybrid layer pattern / head dims / gate flag
    from the REAL Qwen3.5-9B config (verifies #204 for this checkpoint);
  * convert: the fused linear-attn remap fires for the `qwen3_5` arch;
  * build + prefill + full autoregressive decode through the standalone ModelRunner,
    where decode must match a single teacher-forced forward.

The full-attention layers exercise the `attn_output_gate` path (query|gate split,
`sigmoid(gate)` before o_proj). head_dim is 256 (the real value) to prove the int8
prefill/decode kernels handle it end-to-end.
"""

import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.convert import _remap_qwen3_next
from fni8serve.models import ModelRunner, build_model, is_supported
from fni8serve.models.base import ForwardContext
from fni8serve.models.cache import KVCache
from fni8serve.models.config import ModelConfig as _MC

CUDA = torch.cuda.is_available()


# The real Qwen/Qwen3.5-9B config (text_config trimmed to the load-bearing axes).
_REAL_HF = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "vocab_size": 248320,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {"depth": 27, "hidden_size": 1152},
}


def test_qwen3_5_registered():
    assert is_supported("qwen3_5")
    assert is_supported("Qwen3_5ForConditionalGeneration")
    assert is_supported("Qwen3_5ForCausalLM")


def test_qwen3_5_from_hf_real_config():
    """The real Qwen3.5-9B config must derive: hybrid 3:1 linear/full pattern,
    dense MLP (no experts), partial-rotary 0.25 over head_dim 256, rope_theta 1e7,
    the output-gate flag, and the linear head dims (#204 + this PR's mtp key)."""
    c = _MC.from_hf(_REAL_HF)
    assert c.arch == "qwen3_5"
    assert c.linear_attention and c.full_attention_interval == 4
    assert [c.attention_kind(i) for i in range(4)] == ["linear", "linear", "linear", "full"]
    assert c.resolved_head_dim() == 256 and c.rotary_dim() == 64
    assert c.partial_rotary_factor == 0.25 and c.rope_theta == 10000000
    assert c.num_experts == 0 and not c.is_moe()  # DENSE — unlike Qwen3-Next MoE
    assert c.intermediate_size == 12288
    assert not c.qkv_bias
    assert c.extra["attn_output_gate"] is True
    assert c.extra["linear_num_key_heads"] == 16 and c.extra["linear_num_value_heads"] == 32
    assert c.extra["linear_key_head_dim"] == 128 and c.extra["linear_value_head_dim"] == 128
    assert c.num_mtp_layers == 1  # read from mtp_num_hidden_layers


def test_qwen3_5_convert_remaps_fused_linear_attn():
    """The converter's fused in_proj remap must fire for the qwen3_5 arch, splitting
    HF's in_proj_qkvz -> qkv_proj|z_proj and in_proj_ba -> beta_proj|dt_proj."""
    nk, nv, kd, vd = 2, 4, 16, 16
    cfg = _MC(
        arch="qwen3_5",
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        linear_attention=True,
        full_attention_interval=2,
        extra=dict(
            linear_num_key_heads=nk,
            linear_num_value_heads=nv,
            linear_key_head_dim=kd,
            linear_value_head_dim=vd,
        ),
    )
    H = 64
    qk, vdim = nk * kd, nv * vd
    qkvz = qk + qk + vdim
    sd = {
        "model.layers.0.linear_attn.in_proj_qkvz.weight": torch.randn(qkvz + vdim, H),
        "model.layers.0.linear_attn.in_proj_ba.weight": torch.randn(2 * nv, H),
        "model.layers.0.linear_attn.conv1d.weight": torch.randn(qkvz + vdim, 4),
    }
    out = _remap_qwen3_next(sd, cfg)
    assert out["model.layers.0.linear_attn.qkv_proj.weight"].shape[0] == qkvz
    assert out["model.layers.0.linear_attn.z_proj.weight"].shape[0] == vdim
    assert out["model.layers.0.linear_attn.beta_proj.weight"].shape[0] == nv
    assert out["model.layers.0.linear_attn.dt_proj.weight"].shape[0] == nv
    assert out["model.layers.0.linear_attn.conv_weight"].shape[0] == qkvz


# ── synthetic DENSE hybrid checkpoint (gated full attn + DeltaNet linear) ──
_H, _NH, _NKV, _HD = 128, 2, 1, 256  # head_dim 256 == real Qwen3.5-9B
_NK, _NV, _KD, _VD = 1, 2, 16, 16  # linear-attn head dims (kept tiny)
_INTER = 64


def _cfg():
    return _MC(
        arch="qwen3_5",
        vocab_size=64,
        hidden_size=_H,
        num_hidden_layers=2,
        num_attention_heads=_NH,
        num_key_value_heads=_NKV,
        intermediate_size=_INTER,
        max_position_embeddings=64,
        head_dim=_HD,
        qk_norm=True,
        tie_word_embeddings=False,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        linear_attention=True,
        full_attention_interval=2,
        extra=dict(
            linear_num_key_heads=_NK,
            linear_num_value_heads=_NV,
            linear_key_head_dim=_KD,
            linear_value_head_dim=_VD,
            linear_conv_kernel_dim=4,
            attn_output_gate=True,
        ),
    )


def _sd(cfg, device="cpu"):
    def r(*s):
        return torch.randn(*s, device=device, dtype=torch.float16) * 0.05

    qkv_lin = 2 * _NK * _KD + _NV * _VD
    sd = {
        "model.embed_tokens.weight": r(cfg.vocab_size, _H),
        "model.norm.weight": r(_H),
        "lm_head.weight": r(cfg.vocab_size, _H),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(_H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(_H)
        if cfg.attention_kind(i) == "full":
            # q_proj emits query|gate -> 2*nh*hd rows (attn_output_gate)
            sd[f"{p}.self_attn.q_proj.weight"] = r(2 * _NH * _HD, _H)
            sd[f"{p}.self_attn.k_proj.weight"] = r(_NKV * _HD, _H)
            sd[f"{p}.self_attn.v_proj.weight"] = r(_NKV * _HD, _H)
            sd[f"{p}.self_attn.o_proj.weight"] = r(_H, _NH * _HD)
            sd[f"{p}.self_attn.q_norm.weight"] = r(_HD)
            sd[f"{p}.self_attn.k_norm.weight"] = r(_HD)
        else:
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = r(qkv_lin, _H)
            sd[f"{la}.z_proj.weight"] = r(_NV * _VD, _H)
            sd[f"{la}.out_proj.weight"] = r(_H, _NV * _VD)
            sd[f"{la}.conv_weight"] = r(qkv_lin, 4)
            sd[f"{la}.A_log"] = r(_NV).float()
            sd[f"{la}.dt_bias"] = r(_NV).float()
            sd[f"{la}.beta_proj.weight"] = r(_NV, _H)
            sd[f"{la}.dt_proj.weight"] = r(_NV, _H)
            sd[f"{la}.norm.weight"] = r(_NV * _VD)
        # dense SwiGLU MLP (no experts)
        sd[f"{p}.mlp.gate_proj.weight"] = r(_INTER, _H)
        sd[f"{p}.mlp.up_proj.weight"] = r(_INTER, _H)
        sd[f"{p}.mlp.down_proj.weight"] = r(_H, _INTER)
    return sd


@pytest.mark.skipif(not CUDA, reason="prefill needs the dp4a kernels")
def test_qwen3_5_hybrid_prefill():
    cfg = _cfg()
    model = build_model(cfg, _sd(cfg, "cuda")).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        16,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    from fni8serve.models.cache import RecurrentStateCache

    ctx = ForwardContext(is_prefill=True, kv_cache=cache, lin_cache=RecurrentStateCache())
    h = model(ids, pos, ctx)
    logits = model.compute_logits(h[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_qwen3_5_hybrid_decode_matches_teacher_forced():
    """Full generate (prefill + N decode steps) must match a single teacher-forced
    forward over the whole (prompt + generated) sequence — the property the recurrent
    state (linear layers) and per-layer KV cache (gated full layers) preserve."""
    cfg = _cfg()
    model = build_model(cfg, _sd(cfg, "cuda")).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    gen = runner.generate_greedy(prompt, max_new_tokens=4)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    from fni8serve.models.cache import RecurrentStateCache

    ref_cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        32,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    ctx = ForwardContext(is_prefill=True, kv_cache=ref_cache, lin_cache=RecurrentStateCache())
    hidden = model(full_ids, pos, ctx)
    logits = model.compute_logits(hidden)
    ref_tokens = logits[:, prompt.shape[1] - 1 : -1].argmax(-1)
    assert torch.equal(ref_tokens, gen)


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_qwen3_5_offline_fni8_roundtrip(tmp_path):
    """Convert path: quantize -> .fni8 -> load -> build must byte-match a model built
    directly from the same quantized dict (proves the dense-hybrid .fni8 round-trip)."""
    from fni8 import QTensor, save_fni8

    from fni8serve.convert import quantize_state_dict
    from fni8serve.loader import load_fni8_state_dict

    cfg = _cfg()
    qsd = quantize_state_dict(_sd(cfg, "cpu"), weight_bits=8)
    path = str(tmp_path / "q35.fni8")
    save_fni8(path, qsd)

    def _to_cuda(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, QTensor):
                out[k] = QTensor(
                    v.data.cuda(),
                    v.scale.cuda() if v.scale is not None else None,
                    scheme=v.scheme,
                    group_size=v.group_size,
                    codebook=v.codebook,
                )
            else:
                out[k] = v.cuda()
        return out

    direct = {k: (v.data if v.scheme == "raw" else v) for k, v in qsd.items()}
    m_direct = build_model(cfg, _to_cuda(direct)).cuda().eval()
    m_off = build_model(cfg, load_fni8_state_dict(path, device="cuda")).cuda().eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    r_d = ModelRunner(m_direct, cfg, max_batch=1, max_len=32, device="cuda")
    r_o = ModelRunner(m_off, cfg, max_batch=1, max_len=32, device="cuda")
    torch.testing.assert_close(r_o.prefill(prompt), r_d.prefill(prompt), rtol=0, atol=0)
