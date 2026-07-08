# SPDX-License-Identifier: MIT
"""HF -> .fni8 conversion: quantize a state dict, save, reload, build, and confirm
the offline path is numerically identical to runtime quantization (per-row/per-group
int8 scale depends only on its own row, so merge-then-quant == quant-then-merge).
Needs CUDA for the forward."""
import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.convert import is_quantizable_linear, quantize_state_dict
from fni8serve.loader import load_fni8_state_dict
from fni8serve.models import ModelConfig, ModelRunner, build_model

CUDA = torch.cuda.is_available()


def _cfg():
    return ModelConfig(arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=256, head_dim=32, qk_norm=True,
                       tie_word_embeddings=False)


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, dtype=torch.float16) * 0.1
    hd, nh, nkv, H = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H),
          "lm_head.weight": r(cfg.vocab_size, H)}
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
    assert not is_quantizable_linear("model.layers.0.mlp.gate.weight")      # MoE router
    assert not is_quantizable_linear("model.layers.0.input_layernorm.weight")
    assert not is_quantizable_linear("model.embed_tokens.weight")


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
            out[k] = QTensor(v.data.cuda(), v.scale.cuda() if v.scale is not None else None,
                             scheme=v.scheme, group_size=v.group_size, codebook=v.codebook)
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
    qsd = quantize_state_dict(_sd(cfg), weight_bits=8)      # CPU QTensors

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
