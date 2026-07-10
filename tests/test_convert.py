# SPDX-License-Identifier: MIT
"""HF -> .fni8 conversion: quantize a state dict, save, reload, build, and confirm
the offline path is numerically identical to runtime quantization (per-row/per-group
int8 scale depends only on its own row, so merge-then-quant == quant-then-merge).
Needs CUDA for the forward."""

import json
import os

import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.convert import (
    _remap_qwen3_next,
    convert_hf_to_fni8,
    is_quantizable_linear,
    quantize_state_dict,
)
from fni8serve.loader import checkpoint_info, load_fni8_state_dict
from fni8serve.models import ModelConfig, ModelRunner, build_model

CUDA = torch.cuda.is_available()


def _cfg():
    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=False,
    )


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, dtype=torch.float16) * 0.1

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    sd = {
        "model.embed_tokens.weight": r(cfg.vocab_size, H),
        "model.norm.weight": r(H),
        "lm_head.weight": r(cfg.vocab_size, H),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        sd[f"{p}.self_attn.q_proj.weight"] = r(nh * hd, H)
        sd[f"{p}.self_attn.k_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.v_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.o_proj.weight"] = r(H, nh * hd)
        sd[f"{p}.self_attn.q_norm.weight"] = r(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = r(hd)
        sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_is_quantizable_linear_excludes_router_and_norms():
    assert is_quantizable_linear("model.layers.0.self_attn.q_proj.weight")
    assert is_quantizable_linear("model.layers.0.mlp.experts.3.down_proj.weight")
    assert is_quantizable_linear("lm_head.weight")
    assert not is_quantizable_linear("model.layers.0.mlp.gate.weight")  # MoE router
    assert not is_quantizable_linear("model.layers.0.input_layernorm.weight")
    assert not is_quantizable_linear("model.embed_tokens.weight")
    # Non-standard MLP names (LFM2 feed_forward.w1/2/3, Mistral-style) must be
    # quantized too -- the old name-allowlist silently left them fp16.
    assert is_quantizable_linear("model.layers.0.feed_forward.w1.weight")
    assert is_quantizable_linear("model.layers.0.feed_forward.w2.weight")
    assert is_quantizable_linear("model.layers.0.feed_forward.w3.weight")
    assert is_quantizable_linear("model.layers.0.self_attn.out_proj.weight")
    # ...but a router by any common name still stays fp (routing is sensitive).
    assert not is_quantizable_linear("model.layers.0.block_sparse_moe.gate.weight")
    assert not is_quantizable_linear("model.layers.0.mlp.router.weight")


def test_quantize_state_dict_schemes():
    cfg = _cfg()
    q8 = quantize_state_dict(_sd(cfg), weight_bits=8)
    assert q8["model.layers.0.self_attn.q_proj.weight"].scheme == "per_row_i8"
    assert q8["model.norm.weight"].scheme == "raw"
    assert q8["model.layers.0.mlp.gate_proj.weight"].scheme == "per_row_i8"
    q4 = quantize_state_dict(_sd(cfg), weight_bits=4, group_size=32)
    qt = q4["model.layers.0.mlp.down_proj.weight"]
    assert qt.scheme == "per_group_i4" and qt.codebook == "int4" and qt.group_size == 32


def _to_cuda(qsd):
    from fni8 import QTensor

    out = {}
    for k, v in qsd.items():
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


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_fni8_roundtrip_is_exact(tmp_path):
    """save_fni8 -> load must be byte-exact: the offline-loaded model equals one built
    directly from the same quantized dict, bitwise (isolates the container round-trip
    from any CPU/GPU quant ULP)."""
    from fni8 import save_fni8

    cfg = _cfg()
    qsd = quantize_state_dict(_sd(cfg), weight_bits=8)  # CPU QTensors

    path = str(tmp_path / "m.fni8")
    save_fni8(path, qsd)
    # unwrap raw QTensors -> tensors for the direct build (mirrors load_fni8_state_dict)
    direct = {k: (v.data if v.scheme == "raw" else v) for k, v in qsd.items()}
    model_direct = build_model(cfg, _to_cuda(direct)).cuda().eval()
    model_off = build_model(cfg, load_fni8_state_dict(path, device="cuda")).cuda().eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    r_d = ModelRunner(model_direct, cfg, max_batch=1, max_len=32, device="cuda")
    r_o = ModelRunner(model_off, cfg, max_batch=1, max_len=32, device="cuda")
    torch.testing.assert_close(r_o.prefill(prompt), r_d.prefill(prompt), rtol=0, atol=0)


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_fni8_model_close_to_runtime_quant(tmp_path):
    """Offline (.fni8, CPU-quant) vs runtime (GPU-quant) differ only by quant ULPs."""
    from fni8 import save_fni8

    cfg = _cfg()
    sd = _sd(cfg)
    path = str(tmp_path / "m.fni8")
    save_fni8(path, quantize_state_dict(sd, weight_bits=8))
    model_off = build_model(cfg, load_fni8_state_dict(path, device="cuda")).cuda().eval()
    model_rt = build_model(cfg, {k: v.cuda() for k, v in sd.items()}).cuda().eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    a = ModelRunner(model_off, cfg, max_batch=1, max_len=32, device="cuda").prefill(prompt)
    b = ModelRunner(model_rt, cfg, max_batch=1, max_len=32, device="cuda").prefill(prompt)
    cos = torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)
    assert cos.item() >= 0.9999


def _hf_config_dict(cfg):
    return {
        "model_type": cfg.arch,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "intermediate_size": cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "head_dim": cfg.head_dim,
    }


def test_convert_hf_to_fni8_streams_shards_one_at_a_time(tmp_path, monkeypatch):
    """Regression for the 220GB-in-RAM OOM (GLM-4.5-Air): convert_hf_to_fni8 must
    quantize each safetensors shard as it's loaded rather than merging the whole
    checkpoint into one dict first. A synthetic 2-shard checkpoint stands in for the
    real multi-hundred-shard ones — the behavior under test (one quantize call per
    shard, never the full state dict at once) is shard-count-independent."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    import fni8serve.convert as convert_mod

    cfg = _cfg()
    sd = _sd(cfg)
    keys = list(sd)
    mid = len(keys) // 2
    save_file({k: sd[k] for k in keys[:mid]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({k: sd[k] for k in keys[mid:]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(cfg)))

    seen_shard_sizes = []
    orig_quantize = convert_mod.quantize_state_dict

    def spy(sd_arg, **kw):
        seen_shard_sizes.append(len(sd_arg))
        return orig_quantize(sd_arg, **kw)

    monkeypatch.setattr(convert_mod, "quantize_state_dict", spy)

    out = tmp_path / "m.fni8"
    convert_mod.convert_hf_to_fni8(str(tmp_path), str(out), weight_bits=8)

    assert seen_shard_sizes == [mid, len(keys) - mid]  # one call per shard...
    assert all(n < len(keys) for n in seen_shard_sizes)  # ...never the full state dict
    assert checkpoint_info(str(out))["num_tensors"] == len(keys)


def test_convert_hf_to_fni8_deletes_source_shards_as_processed(tmp_path, monkeypatch):
    """Regression for the 850GB+ disk-capacity failure (MiniMax-M3 / issue #69):
    convert_hf_to_fni8 must delete each source safetensors shard immediately after its
    tensors are quantized and written, so peak disk is max(source, consumed+output)
    rather than source+output."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    import fni8serve.convert as convert_mod

    cfg = _cfg()
    sd = _sd(cfg)
    keys = list(sd)
    mid = len(keys) // 2
    save_file({k: sd[k] for k in keys[:mid]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({k: sd[k] for k in keys[mid:]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(cfg)))

    # Track which safetensors files get deleted
    removed_safetensors = []
    orig_remove = os.remove

    def spy_remove(path):
        if path.endswith(".safetensors"):
            removed_safetensors.append(path)
        return orig_remove(path)

    monkeypatch.setattr(os, "remove", spy_remove)

    # Shards exist before conversion
    assert (tmp_path / "model-00001-of-00002.safetensors").exists()
    assert (tmp_path / "model-00002-of-00002.safetensors").exists()
    shard_count_before = len(list(tmp_path.glob("*.safetensors")))

    out = tmp_path / "m.fni8"
    convert_mod.convert_hf_to_fni8(str(tmp_path), str(out), weight_bits=8)

    # Every source shard was deleted via os.remove during conversion
    assert len(removed_safetensors) == 2
    assert shard_count_before == 2
    assert not (tmp_path / "model-00001-of-00002.safetensors").exists()
    assert not (tmp_path / "model-00002-of-00002.safetensors").exists()
    # config.json and the output file must survive
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "m.fni8").exists()
    assert checkpoint_info(str(out))["num_tensors"] == len(keys)


def test_convert_hf_to_fni8_multi_shard_output_matches_single_shard(tmp_path):
    """However the checkpoint is split into shards, the quantized output must be
    identical (per-row/per-group scales depend only on a tensor's own values)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    cfg = _cfg()
    sd = _sd(cfg)
    cfg_json = json.dumps(_hf_config_dict(cfg))

    one_shard_dir = tmp_path / "one"
    one_shard_dir.mkdir()
    save_file(sd, str(one_shard_dir / "model.safetensors"))
    (one_shard_dir / "config.json").write_text(cfg_json)
    out_one = one_shard_dir / "m.fni8"
    convert_hf_to_fni8(str(one_shard_dir), str(out_one), weight_bits=8)

    many_shard_dir = tmp_path / "many"
    many_shard_dir.mkdir()
    for i, k in enumerate(sd):
        save_file({k: sd[k]}, str(many_shard_dir / f"model-{i:05d}.safetensors"))
    (many_shard_dir / "config.json").write_text(cfg_json)
    out_many = many_shard_dir / "m.fni8"
    convert_hf_to_fni8(str(many_shard_dir), str(out_many), weight_bits=8)

    info_one, info_many = checkpoint_info(str(out_one)), checkpoint_info(str(out_many))
    assert info_one["num_tensors"] == info_many["num_tensors"] == len(sd)
    # header "arch" is the hardware target (sm70); the model arch lives in meta.
    assert info_one["arch"] == info_many["arch"]
    assert info_one["meta"]["arch"] == info_many["meta"]["arch"] == cfg.arch


def _qwen3_next_cfg():
    return ModelConfig(
        arch="qwen3_next",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        linear_attention=True,
        full_attention_interval=1,
        extra=dict(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        ),
    )


def _hf_fused_sd(cfg):
    """Qwen3-Next state dict using the fused HF tensor names."""
    H, nk, nv, kd, vd = (
        cfg.hidden_size,
        cfg.extra["linear_num_key_heads"],
        cfg.extra["linear_num_value_heads"],
        cfg.extra["linear_key_head_dim"],
        cfg.extra["linear_value_head_dim"],
    )
    qk = nk * kd
    v_dim = nv * vd
    sd = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, H, dtype=torch.float16),
        "model.norm.weight": torch.randn(H, dtype=torch.float16),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(H, dtype=torch.float16)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.float16)
        la = f"{p}.linear_attn"
        # Fused HF names
        sd[f"{la}.in_proj_qkvz.weight"] = torch.randn(
            qk + qk + v_dim + v_dim, H, dtype=torch.float16
        )
        sd[f"{la}.in_proj_ba.weight"] = torch.randn(2 * nv, H, dtype=torch.float16)
        sd[f"{la}.conv1d.weight"] = torch.randn(qk + qk + v_dim, 4, dtype=torch.float16)
        sd[f"{la}.out_proj.weight"] = torch.randn(H, v_dim, dtype=torch.float16)
        sd[f"{la}.A_log"] = torch.randn(nv).float()
        sd[f"{la}.dt_bias"] = torch.randn(nv).float()
        sd[f"{la}.norm.weight"] = torch.randn(v_dim, dtype=torch.float16)
    return sd


def test_qwen3_next_hf_fused_names_remap():
    """_remap_qwen3_next must split fused HF names into the per-name tensors the
    builder expects (qkv_proj, z_proj, beta_proj, dt_proj, conv_weight)."""
    cfg = _qwen3_next_cfg()
    hf_sd = _hf_fused_sd(cfg)
    mapped = _remap_qwen3_next(hf_sd, cfg)

    H = cfg.hidden_size
    nk = cfg.extra["linear_num_key_heads"]
    nv = cfg.extra["linear_num_value_heads"]
    kd = cfg.extra["linear_key_head_dim"]
    vd = cfg.extra["linear_value_head_dim"]
    qk = nk * kd
    v_dim = nv * vd

    # Fused names must be gone
    assert not any("in_proj_qkvz" in k for k in mapped)
    assert not any("in_proj_ba" in k for k in mapped)
    assert not any("conv1d" in k for k in mapped)

    # Builder-expected names must exist
    la = "model.layers.0.linear_attn"
    assert f"{la}.qkv_proj.weight" in mapped
    assert f"{la}.z_proj.weight" in mapped
    assert f"{la}.beta_proj.weight" in mapped
    assert f"{la}.dt_proj.weight" in mapped
    assert f"{la}.conv_weight" in mapped

    # Shape checks
    assert mapped[f"{la}.qkv_proj.weight"].shape == (qk + qk + v_dim, H)
    assert mapped[f"{la}.z_proj.weight"].shape == (v_dim, H)
    assert mapped[f"{la}.beta_proj.weight"].shape == (nv, H)
    assert mapped[f"{la}.dt_proj.weight"].shape == (nv, H)
    assert mapped[f"{la}.conv_weight"].shape == (qk + qk + v_dim, 4)

    # Non-linear-attn tensors pass through unchanged
    assert "model.embed_tokens.weight" in mapped
    assert "model.norm.weight" in mapped
    assert "model.layers.0.input_layernorm.weight" in mapped


def test_qwen3_next_conv1d_trim_4part():
    """conv1d.weight with 4 parts (including Z rows) is trimmed to 3 parts."""
    cfg = _qwen3_next_cfg()
    hf_sd = _hf_fused_sd(cfg)
    H = cfg.hidden_size
    nk = cfg.extra["linear_num_key_heads"]
    nv = cfg.extra["linear_num_value_heads"]
    kd = cfg.extra["linear_key_head_dim"]
    vd = cfg.extra["linear_value_head_dim"]
    qk = nk * kd
    v_dim = nv * vd
    qkv_dim = qk + qk + v_dim
    # Replace with 4-part conv weight
    hf_sd["model.layers.0.linear_attn.conv1d.weight"] = torch.randn(
        qkv_dim + v_dim, 4, dtype=torch.float16
    )
    mapped = _remap_qwen3_next(hf_sd, cfg)
    conv_w = mapped["model.layers.0.linear_attn.conv_weight"]
    assert conv_w.shape == (qkv_dim, 4), f"expected ({qkv_dim}, 4) got {conv_w.shape}"


def test_qwen3_next_remap_skips_other_archs():
    """_remap_qwen3_next is a no-op for non-qwen3_next architectures."""
    cfg = _cfg()
    sd = _sd(cfg)
    mapped = _remap_qwen3_next(sd, cfg)
    # same keys, same tensors
    assert set(mapped.keys()) == set(sd.keys())
    for k in sd:
        assert mapped[k] is sd[k]
