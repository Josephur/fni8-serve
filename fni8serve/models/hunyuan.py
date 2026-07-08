# SPDX-License-Identifier: MIT
"""Tencent Hunyuan (hunyuan_v1_moe, e.g. Hunyuan-A13B) — GQA + QK-norm + softmax MoE
with a shared expert. Registered `hunyuan`.

Per HF: QK-norm named query_layernorm/key_layernorm (over head_dim, pre-RoPE), no
attention bias, rope_theta 1e4; MoE router `mlp.gate.wg` fp32 softmax -> top-k ->
renorm, shared expert `mlp.shared_mlp`, all layers MoE (A13B). Plain RMSNorm eps
1e-5, tied embeddings. Cross-layer attention (Hunyuan-Large only) not modeled here.
"""
from __future__ import annotations

import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gqa_attention import GQAAttention
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE
from .registry import register_model
from .weights import gate_up_weight, merge_qtensor, to_qtensor


class HunyuanDecoderLayer(nn.Module):
    def __init__(self, cfg, i, sd, rope):
        super().__init__()
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        hd = cfg.resolved_head_dim()
        self.self_attn = GQAAttention(
            num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads, head_dim=hd,
            qkv_proj=merge_qtensor([sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"], sd[f"{a}.v_proj.weight"]]),
            o_proj=to_qtensor(sd[f"{a}.o_proj.weight"]), scale=hd ** -0.5, rope=rope,
            q_norm=sd.get(f"{a}.query_layernorm.weight"), k_norm=sd.get(f"{a}.key_layernorm.weight"),
            rms_norm_eps=cfg.rms_norm_eps)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                sd[f"{p}.post_attention_layernorm.weight"])
        experts = [(gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
                    to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"])) for e in range(cfg.num_experts)]
        shared = None
        if f"{p}.mlp.shared_mlp.gate_proj.weight" in sd:
            shared = (gate_up_weight(sd, f"{p}.mlp.shared_mlp"),
                      to_qtensor(sd[f"{p}.mlp.shared_mlp.down_proj.weight"]))
        gate_w = sd.get(f"{p}.mlp.gate.wg.weight", sd.get(f"{p}.mlp.gate.weight"))
        self.mlp = SparseMoE(gate=gate_w, experts=experts, top_k=cfg.num_experts_per_tok,
                             norm_topk_prob=cfg.norm_topk_prob, act=cfg.hidden_act,
                             shared_expert=shared, scoring_func="softmax")

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.self_attn(h, positions, ctx, 0)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class HunyuanForCausalLM(nn.Module):
    def __init__(self, cfg, sd):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"])
        rope = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings, base=cfg.rope_theta)
        self.layers = nn.ModuleList([HunyuanDecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"])
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        self.lm_head = LMHead(lm_w if cfg.tie_word_embeddings else to_qtensor(lm_w))

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.norm(h, residual)
        return h

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model("hunyuan", "hunyuan_v1_moe", "HunYuanMoEV1ForCausalLM")
def build_hunyuan(cfg: ModelConfig, weights: dict) -> HunyuanForCausalLM:
    if cfg.rope_theta == 1e6:
        cfg.rope_theta = 1e4     # Hunyuan default
    return HunyuanForCausalLM(cfg, weights)
