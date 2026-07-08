# SPDX-License-Identifier: MIT
"""Registration + prefill for the additional families (qwen3_next hybrid, LFM2,
GLM, Hunyuan, MiniMax). Each builds through the registry and runs a prefill forward
with random weights — proving the modular assembly. Full autoregressive decode for
the linear/conv/lightning families needs recurrent-state caching (Track 1.5)."""
import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.models import ModelConfig, build_model, is_supported
from fni8serve.models.base import ForwardContext
from fni8serve.models.cache import KVCache

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="projections use the dp4a GEMM")


def _r(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05


def _prefill(cfg, sd):
    model = build_model(cfg, sd).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = KVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 16,
                    cfg.resolved_head_dim(), device="cuda")
    h = model(ids, pos, ForwardContext(is_prefill=True, kv_cache=cache))
    logits = model.compute_logits(h[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()


def _moe_sd(sd, p, H, E, mi, shared_name=None):
    sd[f"{p}.mlp.gate.weight"] = _r(E, H)
    for e in range(E):
        sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = _r(H, mi)
    if shared_name:
        sd[f"{p}.mlp.{shared_name}.gate_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.{shared_name}.up_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.{shared_name}.down_proj.weight"] = _r(H, mi)


def test_qwen3_next_hybrid_prefill():
    x = dict(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
             linear_value_head_dim=16, linear_conv_kernel_dim=4)
    cfg = ModelConfig(arch="qwen3_next", vocab_size=64, hidden_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                      max_position_embeddings=64, head_dim=32, qk_norm=True,
                      tie_word_embeddings=True, num_experts=2, num_experts_per_tok=2,
                      moe_intermediate_size=64, linear_attention=True, full_attention_interval=2,
                      shared_expert_intermediate_size=64, extra=x)
    assert is_supported("qwen3_next")
    H, nh, nkv, hd = 128, 4, 2, 32
    nk, nv, kd, vd = 2, 4, 16, 16
    qkv_lin = 2 * nk * kd + nv * vd
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        if (i + 1) % 2 == 0:      # full attn
            sd[f"{p}.self_attn.q_proj.weight"] = _r(nh * hd, H)
            sd[f"{p}.self_attn.k_proj.weight"] = _r(nkv * hd, H)
            sd[f"{p}.self_attn.v_proj.weight"] = _r(nkv * hd, H)
            sd[f"{p}.self_attn.o_proj.weight"] = _r(H, nh * hd)
            sd[f"{p}.self_attn.q_norm.weight"] = _r(hd)
            sd[f"{p}.self_attn.k_norm.weight"] = _r(hd)
        else:                     # linear (DeltaNet)
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = _r(qkv_lin, H)
            sd[f"{la}.out_proj.weight"] = _r(H, nv * vd)
            sd[f"{la}.conv_weight"] = _r(qkv_lin, 4)
            sd[f"{la}.A_log"] = _r(nv).float()
            sd[f"{la}.dt_bias"] = _r(nv).float()
            sd[f"{la}.beta_proj.weight"] = _r(nv, H)
            sd[f"{la}.dt_proj.weight"] = _r(nv, H)
            sd[f"{la}.norm.weight"] = _r(nv * vd)
        _moe_sd(sd, p, H, cfg.num_experts, cfg.moe_intermediate_size, shared_name="shared_expert")
    _prefill(cfg, sd)


def _attn_sd(sd, a, cfg, qk=None, bias=False):
    hd, nh, nkv, H = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
    o_name = "out_proj" if a.endswith("self_attn") and cfg.arch in ("lfm2",) else "o_proj"
    sd[f"{a}.q_proj.weight"] = _r(nh * hd, H)
    sd[f"{a}.k_proj.weight"] = _r(nkv * hd, H)
    sd[f"{a}.v_proj.weight"] = _r(nkv * hd, H)
    sd[f"{a}.{o_name}.weight"] = _r(H, nh * hd)
    if bias:
        for x in "qkv":
            sd[f"{a}.{x}_proj.bias"] = _r(nh * hd if x == "q" else nkv * hd)
    if qk:
        sd[f"{a}.{qk[0]}.weight"] = _r(hd)
        sd[f"{a}.{qk[1]}.weight"] = _r(hd)


def test_lfm2_prefill():
    cfg = ModelConfig(arch="lfm2", vocab_size=64, hidden_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                      max_position_embeddings=64, head_dim=32, qk_norm=True, tie_word_embeddings=True,
                      rms_norm_eps=1e-5, extra=dict(full_attn_idxs=[1], conv_L_cache=3))
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = _r(H)
        sd[f"{p}.ffn_norm.weight"] = _r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = _r(d, H if w != "w2" else cfg.intermediate_size)
        if i == 1:  # attention layer
            _attn_sd(sd, f"{p}.self_attn", cfg, qk=("q_layernorm", "k_layernorm"))
        else:       # conv layer
            sd[f"{p}.conv.in_proj.weight"] = _r(3 * H, H)
            sd[f"{p}.conv.out_proj.weight"] = _r(H, H)
            sd[f"{p}.conv.conv.weight"] = _r(H, 1, 3)
    _prefill(cfg, sd)


def test_glm_prefill():
    cfg = ModelConfig(arch="glm", vocab_size=64, hidden_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                      max_position_embeddings=64, head_dim=32, qk_norm=True, partial_rotary_factor=0.5,
                      tie_word_embeddings=False, rms_norm_eps=1e-5, num_experts=4, num_experts_per_tok=2,
                      moe_intermediate_size=64, extra=dict(first_k_dense_replace=1, routed_scaling_factor=2.5))
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H),
          "lm_head.weight": _r(cfg.vocab_size, H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        _attn_sd(sd, f"{p}.self_attn", cfg, qk=("q_norm", "k_norm"), bias=True)
        if i >= 1:
            _moe_sd(sd, p, H, cfg.num_experts, cfg.moe_intermediate_size, shared_name="shared_experts")
            sd[f"{p}.mlp.gate.e_score_correction_bias"] = _r(cfg.num_experts).float()
        else:
            for w, d in (("gate_proj", cfg.intermediate_size), ("up_proj", cfg.intermediate_size), ("down_proj", H)):
                sd[f"{p}.mlp.{w}.weight"] = _r(d, H if w != "down_proj" else cfg.intermediate_size)
    _prefill(cfg, sd)


def test_hunyuan_prefill():
    cfg = ModelConfig(arch="hunyuan", vocab_size=64, hidden_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                      max_position_embeddings=64, head_dim=32, qk_norm=True, tie_word_embeddings=True,
                      rms_norm_eps=1e-5, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64)
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        _attn_sd(sd, f"{p}.self_attn", cfg, qk=("query_layernorm", "key_layernorm"))
        _moe_sd(sd, p, H, cfg.num_experts, cfg.moe_intermediate_size, shared_name="shared_mlp")
        sd[f"{p}.mlp.gate.wg.weight"] = sd.pop(f"{p}.mlp.gate.weight")
    _prefill(cfg, sd)


def test_minimax_prefill():
    cfg = ModelConfig(arch="minimax", vocab_size=64, hidden_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, intermediate_size=256,
                      max_position_embeddings=64, head_dim=32, tie_word_embeddings=False,
                      rms_norm_eps=1e-5, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
                      rope_theta=1e7, partial_rotary_factor=0.5,
                      extra=dict(attn_type_list=[0, 1], layernorm_full_attention_alpha=1.0, layernorm_mlp_beta=1.0))
    H, hd, nh = cfg.hidden_size, cfg.resolved_head_dim(), cfg.num_attention_heads
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H),
          "lm_head.weight": _r(cfg.vocab_size, H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        a = f"{p}.self_attn"
        if i == 0:  # lightning
            sd[f"{a}.qkv_proj.weight"] = _r(3 * nh * hd, H)
            sd[f"{a}.out_proj.weight"] = _r(H, nh * hd)
            sd[f"{a}.output_gate.weight"] = _r(nh * hd, H)
            sd[f"{a}.norm.weight"] = _r(nh * hd)
        else:       # softmax
            _attn_sd(sd, a, cfg)
        m = f"{p}.block_sparse_moe"
        sd[f"{m}.gate.weight"] = _r(cfg.num_experts, H)
        for e in range(cfg.num_experts):
            sd[f"{m}.experts.{e}.w1.weight"] = _r(cfg.moe_intermediate_size, H)
            sd[f"{m}.experts.{e}.w3.weight"] = _r(cfg.moe_intermediate_size, H)
            sd[f"{m}.experts.{e}.w2.weight"] = _r(H, cfg.moe_intermediate_size)
    _prefill(cfg, sd)
