# SPDX-License-Identifier: MIT
"""ModelConfig — the union of architectural axes across supported families.

One flat dataclass whose fields are a superset of the HF `config.json` knobs we
care about, so a new model family is expressed as *values*, not code. A concrete
model's `build()` reads only the fields it needs; unused fields stay at their
defaults. `from_hf(dict)` maps a HuggingFace text config (or the `text_config`
sub-dict of a multimodal one) onto this schema.

Axes captured: GQA dims, norm style (plain vs Gemma (1+w)), activation, RoPE
(single- or dual-theta for Gemma local/global), sliding-window pattern, QK-norm,
attention/query scaling + soft-caps, MoE routing, MTP depth, and the decode
strategy (autoregressive vs diffusion).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    arch: str                                   # registry key, e.g. "qwen3"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int = 32768
    head_dim: int | None = None                 # explicit; else hidden // heads
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6                      # global-layer theta
    rope_local_theta: float | None = None       # Gemma local-layer theta (dual-RoPE)
    tie_word_embeddings: bool = False

    # norm / embedding style
    norm_add_unit_offset: bool = False          # Gemma (1+w) RMSNorm
    embed_scale: float | None = None            # Gemma sqrt(hidden) embedding scale
    qk_norm: bool = False                        # per-head RMSNorm on Q,K (Qwen3/Gemma3)
    qkv_bias: bool = False                       # Qwen2 had it; Qwen3/Gemma3 do not

    # attention scaling / caps
    query_pre_attn_scalar: float | None = None  # Gemma: scale = scalar**-0.5
    attn_logit_softcap: float | None = None      # Gemma2 (removed in Gemma3)
    final_logit_softcap: float | None = None

    # partial rotary (GLM, Qwen3-Next gated attn): rotate only factor*head_dim dims
    partial_rotary_factor: float = 1.0

    # sliding window (Gemma3 local layers, Mistral). pattern P => global every Pth.
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None    # e.g. 6 => layers where (i+1)%6==0 are global

    # hybrid linear-attention backbone (Qwen3-Next / Qwen3.5/3.6, MiniMax lightning):
    # a subset of layers use linear attention; the rest are full softmax attention.
    linear_attention: bool = False               # this family has linear-attn layers
    full_attention_interval: int = 0             # full-attn every Nth layer (0 => none/all)

    # multi-head latent attention (DeepSeek-V2/V3/V4): compressed latent KV
    latent_attention: bool = False

    # MoE (Qwen3-MoE)
    num_experts: int = 0                         # 0 => dense
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    shared_expert_intermediate_size: int = 0     # 0 => no shared expert
    decoder_sparse_step: int = 1                 # every Nth layer is MoE (else dense)
    mlp_only_layers: tuple[int, ...] = ()        # layer indices forced dense

    # MTP (multi-token prediction; Qwen3-Next / DeepSeek-V3 style)
    num_mtp_layers: int = 0

    # decode strategy
    decode_strategy: str = "autoregressive"      # or "diffusion"

    # quantization intent (how weights were stored in the .fni8)
    weight_bits: int = 8

    extra: dict = field(default_factory=dict)    # arch-specific overflow

    def resolved_head_dim(self) -> int:
        return self.head_dim or (self.hidden_size // self.num_attention_heads)

    def is_moe(self) -> bool:
        return self.num_experts > 0

    def layer_is_global(self, layer_idx: int) -> bool:
        """Full-attention layer? True for dense models; for Gemma3 sliding-window
        models, only every `sliding_window_pattern`-th layer is global."""
        if not self.sliding_window or not self.sliding_window_pattern:
            return True
        return (layer_idx + 1) % self.sliding_window_pattern == 0

    def attention_kind(self, layer_idx: int) -> str:
        """Per-layer attention backend selector — the axis that makes hybrids work.
        'linear'  : Gated-DeltaNet / lightning linear attention (Qwen3-Next, MiniMax)
        'latent'  : MLA compressed-KV (DeepSeek)
        'sliding' : local windowed softmax attention (Gemma3 local layers)
        'full'    : standard GQA softmax attention (the fni8 dp4a default)"""
        if self.latent_attention:
            return "latent"
        if self.linear_attention and self.full_attention_interval:
            # full attn every Nth layer, linear (DeltaNet) on the rest
            return "full" if (layer_idx + 1) % self.full_attention_interval == 0 else "linear"
        if self.sliding_window and self.sliding_window_pattern:
            return "full" if self.layer_is_global(layer_idx) else "sliding"
        return "full"

    def rotary_dim(self) -> int:
        return int(self.resolved_head_dim() * self.partial_rotary_factor)

    def layer_is_moe(self, layer_idx: int) -> bool:
        if not self.is_moe() or layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    @classmethod
    def from_hf(cls, hf: dict, *, arch: str | None = None) -> "ModelConfig":
        """Map a HuggingFace text config dict onto ModelConfig. Accepts either a
        top-level text config or one nested under `text_config`."""
        c = dict(hf)
        if "text_config" in c and "hidden_size" not in c:
            c = dict(c["text_config"])
        archs = c.get("architectures") or []
        model_type = c.get("model_type", "")
        resolved_arch = arch or model_type or (archs[0] if archs else "unknown")
        n_heads = c["num_attention_heads"]
        return cls(
            arch=resolved_arch,
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=c.get("num_key_value_heads", n_heads),
            intermediate_size=c.get("intermediate_size", 0),
            max_position_embeddings=c.get("max_position_embeddings", 32768),
            head_dim=c.get("head_dim"),
            hidden_act=c.get("hidden_act", "silu"),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=c.get("rope_theta", 1e6),
            rope_local_theta=c.get("rope_local_base_freq"),
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            qkv_bias=c.get("attention_bias", False),
            query_pre_attn_scalar=c.get("query_pre_attn_scalar"),
            attn_logit_softcap=c.get("attn_logit_softcapping"),
            final_logit_softcap=c.get("final_logit_softcapping"),
            sliding_window=c.get("sliding_window"),
            sliding_window_pattern=c.get("sliding_window_pattern"),
            num_experts=c.get("num_experts", 0),
            num_experts_per_tok=c.get("num_experts_per_tok", 0),
            moe_intermediate_size=c.get("moe_intermediate_size", 0),
            norm_topk_prob=c.get("norm_topk_prob", True),
            shared_expert_intermediate_size=c.get("shared_expert_intermediate_size", 0),
            decoder_sparse_step=c.get("decoder_sparse_step", 1),
            mlp_only_layers=tuple(c.get("mlp_only_layers", []) or []),
            num_mtp_layers=c.get("num_nextn_predict_layers", 0),
            extra={k: v for k, v in c.items() if k not in _KNOWN_HF_KEYS},
        )


_KNOWN_HF_KEYS = {
    "text_config", "architectures", "model_type", "vocab_size", "hidden_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "max_position_embeddings", "head_dim", "hidden_act",
    "rms_norm_eps", "rope_theta", "rope_local_base_freq", "tie_word_embeddings",
    "attention_bias", "query_pre_attn_scalar", "attn_logit_softcapping",
    "final_logit_softcapping", "sliding_window", "sliding_window_pattern",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size", "norm_topk_prob",
    "shared_expert_intermediate_size", "decoder_sparse_step", "mlp_only_layers",
    "num_nextn_predict_layers",
}
