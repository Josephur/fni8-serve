# SPDX-License-Identifier: MIT
"""Qwen3.5 (real `Qwen/Qwen3.5-9B`) — HYBRID Gated-DeltaNet (linear) + GATED full
attention, with a DENSE SwiGLU MLP.

Distinct from `qwen3_next` (the 80B-A3B MoE backbone) on two axes that matter for
the 9B checkpoint:

  1. **Dense MLP.** Qwen3.5-9B has no experts (`num_experts == 0`,
     `intermediate_size == 12288`); every layer is a plain SwiGLU `GatedMLP`. The
     builder still routes to `SparseMoE` when a config DOES carry experts, so the
     larger MoE Qwen3.5/3.6 variants reuse this same assembly.
  2. **Gated full attention** (`attn_output_gate == True`). The full-attention
     layers emit a per-head output gate (q_proj -> query|gate) and multiply the
     attention output by `sigmoid(gate)` before o_proj — see
     `layers/gated_gqa_attention.py`. The linear (DeltaNet) layers are identical to
     `qwen3_next`.

`attention_kind(i)` picks the backend per layer from `layer_types` (3 linear : 1
full, `full_attention_interval == 4`). RoPE is partial-rotary 0.25 over head_dim 256,
theta 1e7. Registered under the HF `model_type`/arch of the (VLM-wrapped) release;
this builder serves the TEXT backbone (vision tower is out of scope here).
"""

from __future__ import annotations

import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gated_gqa_attention import GatedGQAAttention
from ..layers.linear_attn import GatedDeltaNetAttention
from ..layers.mlp import GatedMLP
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE
from .registry import register_model
from .weights import gate_up_weight, qkv_weight, to_qtensor


def _gated_full_attn(cfg, sd, p, rope):
    """Full-attention layer WITH output gate. HF stores the fused query+gate in
    `self_attn.q_proj.weight` ([2*nh*hd, H]); merging q|k|v yields the
    [2*nh*hd | nkv*hd | nkv*hd] projection GatedGQAAttention expects."""
    hd = cfg.resolved_head_dim()
    return GatedGQAAttention(
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd,
        qkv_gate_proj=qkv_weight(sd, f"{p}.self_attn"),
        o_proj=to_qtensor(sd[f"{p}.self_attn.o_proj.weight"]),
        scale=hd**-0.5,
        rope=rope,
        q_norm=sd.get(f"{p}.self_attn.q_norm.weight"),
        k_norm=sd.get(f"{p}.self_attn.k_norm.weight"),
        rms_norm_eps=cfg.rms_norm_eps,
    )


def _linear_attn(cfg, sd, p):
    x = cfg.extra
    la = f"{p}.linear_attn"
    return GatedDeltaNetAttention(
        cfg,
        qkv_proj=to_qtensor(sd[f"{la}.qkv_proj.weight"]),
        out_proj=to_qtensor(sd[f"{la}.out_proj.weight"]),
        conv_weight=sd[f"{la}.conv_weight"],
        a_log=sd[f"{la}.A_log"],
        dt_bias=sd[f"{la}.dt_bias"],
        beta_proj=to_qtensor(sd[f"{la}.beta_proj.weight"]),
        gate_proj=to_qtensor(sd[f"{la}.dt_proj.weight"]),
        z_proj=to_qtensor(sd[f"{la}.z_proj.weight"]),
        norm_gain=sd[f"{la}.norm.weight"],
        num_k_heads=x["linear_num_key_heads"],
        num_v_heads=x["linear_num_value_heads"],
        key_dim=x["linear_key_head_dim"],
        value_dim=x["linear_value_head_dim"],
        conv_kernel=x.get("linear_conv_kernel_dim", 4),
    )


def _build_mlp(cfg, sd, p):
    """Dense SwiGLU (Qwen3.5-9B) or ultra-sparse MoE + shared expert (larger MoE
    variants). Selected purely by whether the config carries experts."""
    if not cfg.is_moe():
        return GatedMLP(
            gate_up_weight(sd, f"{p}.mlp"),
            to_qtensor(sd[f"{p}.mlp.down_proj.weight"]),
            act=cfg.hidden_act,
        )
    experts = [
        (
            gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
            to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"]),
        )
        for e in range(cfg.num_experts)
    ]
    shared = None
    if f"{p}.mlp.shared_expert.gate_proj.weight" in sd:
        shared = (
            gate_up_weight(sd, f"{p}.mlp.shared_expert"),
            to_qtensor(sd[f"{p}.mlp.shared_expert.down_proj.weight"]),
        )
    return SparseMoE(
        gate=sd[f"{p}.mlp.gate.weight"],
        experts=experts,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
        act=cfg.hidden_act,
        shared_expert=shared,
        shared_expert_gate=sd.get(f"{p}.mlp.shared_expert_gate.weight"),
    )


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int, sd: dict, rope: RotaryEmbedding):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        self.kind = cfg.attention_kind(i)
        self.attn = (
            _gated_full_attn(cfg, sd, p, rope) if self.kind == "full" else _linear_attn(cfg, sd, p)
        )
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"]
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.post_attention_layernorm.weight"]
        )
        self.mlp = _build_mlp(cfg, sd, p)

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"])
        rope = RotaryEmbedding(
            cfg.resolved_head_dim(),
            cfg.max_position_embeddings,
            base=cfg.rope_theta,
            rotary_dim=cfg.rotary_dim(),
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"])
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        self.lm_head = LMHead(to_qtensor(lm_w))

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.norm(h, residual)
        return h

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model("qwen3_5", "qwen3.5", "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
def build_qwen3_5(cfg: ModelConfig, weights: dict) -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM(cfg, weights)
