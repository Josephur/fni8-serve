# SPDX-License-Identifier: MIT
"""DeepSeek (MLA + fine-grained MoE + shared expert) registration + prefill.

Builds a small random DeepSeek through the registry and runs a prefill forward
(MLA decompress path + first-dense-then-MoE layers). MLA decode needs latent-KV
caching (the Track-2 absorb kernel's home), so this covers build + prefill."""
import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.models import ModelConfig, build_model, is_supported

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="MLA projections use the dp4a GEMM")


def _cfg():
    return ModelConfig(
        arch="deepseek", vocab_size=128, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, intermediate_size=256,
        max_position_embeddings=64, head_dim=48, tie_word_embeddings=True,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        extra=dict(q_lora_rank=96, kv_lora_rank=64, qk_nope_head_dim=32,
                   qk_rope_head_dim=16, v_head_dim=32, first_k_dense_replace=1),
    )


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
    x = cfg.extra
    H, nh = cfg.hidden_size, cfg.num_attention_heads
    qk = x["qk_nope_head_dim"] + x["qk_rope_head_dim"]
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        a = f"{p}.self_attn"
        sd[f"{a}.q_a_proj.weight"] = r(x["q_lora_rank"], H)
        sd[f"{a}.q_a_layernorm.weight"] = r(x["q_lora_rank"])
        sd[f"{a}.q_b_proj.weight"] = r(nh * qk, x["q_lora_rank"])
        sd[f"{a}.kv_a_proj_with_mqa.weight"] = r(x["kv_lora_rank"] + x["qk_rope_head_dim"], H)
        sd[f"{a}.kv_a_layernorm.weight"] = r(x["kv_lora_rank"])
        sd[f"{a}.kv_b_proj.weight"] = r(nh * (x["qk_nope_head_dim"] + x["v_head_dim"]), x["kv_lora_rank"])
        sd[f"{a}.o_proj.weight"] = r(H, nh * x["v_head_dim"])
        if i >= x["first_k_dense_replace"]:
            sd[f"{p}.mlp.gate.weight"] = r(cfg.num_experts, H)
            for e in range(cfg.num_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(cfg.moe_intermediate_size, H)
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(cfg.moe_intermediate_size, H)
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
            for n in ("gate_proj", "up_proj"):
                sd[f"{p}.mlp.shared_experts.{n}.weight"] = r(cfg.moe_intermediate_size, H)
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
        else:
            sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
            sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
            sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_deepseek_registered():
    assert is_supported("deepseek") and is_supported("DeepseekV3ForCausalLM")


def test_deepseek_prefill():
    from fni8serve.models.base import ForwardContext
    from fni8serve.models.cache import KVCache
    cfg = _cfg()
    model = build_model(cfg, _sd(cfg)).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = KVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 16,
                    cfg.resolved_head_dim(), device="cuda")
    hidden = model(ids, pos, ForwardContext(is_prefill=True, kv_cache=cache))
    logits = model.compute_logits(hidden[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()
