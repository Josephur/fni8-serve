# SPDX-License-Identifier: MIT
"""Native GGUF loader — build the engine straight from a `.gguf`, no `.fni8`.

This is the P1 seam of the GGUF-native migration (see the sister repo's
`MIGRATION.md` §2/§3/§7). It mirrors the existing `.fni8` load path
(`loader.load_fni8_state_dict` + `api.server.load_engine`) but sources everything
from GGUF-KV metadata + raw k-quant tensor bytes instead of the `.fni8` container:

    gguf_config(path)            -> ModelConfig   (GGUF-KV → the same schema
                                                   `ModelConfig.from_hf` produces)
    gguf_state_dict(path, dev)   -> {name: QTensor|Tensor}   (resident k-quant,
                                                   NO dequant→requant transcode)
    load_gguf_engine(path, ...)  -> LLMEngine

It removes NO `.fni8` code (that is P5); the loader simply branches on file suffix.

**KV-key provenance.** Every mapping below cites its source: the standard llama.cpp
`<arch>.*` schema (`gguf-py/gguf/constants.py` `Keys`, written by
`convert_hf_to_gguf.py`) and the four research digests in the sister repo's
`research/` (`llamacpp-hybrid-linattn.md`, `llamacpp-moe.md`, `llamacpp-mtp-spec.md`).
`gguf` (llama.cpp's own python reader) is the only new dependency.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from fni8 import QTensor

from .models.config import ModelConfig
from .models.registry import _resolve

# ── GGUF k-quant type tag ↔ our `gguf_kquant` codebook + block byte size ──────
# type_size (bytes per 256-value super-block) is fixed by the GGML layout; the raw
# GGUF bytes are already stored [out, n_superblocks*type_size], exactly the shape a
# `gguf_kquant` QTensor wants (verified against gguf.constants.GGML_QUANT_SIZES).
_KQUANT = {
    "Q4_K": ("q4_k", 144),
    "Q5_K": ("q5_k", 176),
    "Q6_K": ("q6_k", 210),
}
_FLOAT_TYPES = {"F32", "F16", "BF16"}

# ── GGUF `general.architecture` string → our registry builder key ────────────
# The GGUF arch strings (llama.cpp `LLM_ARCH_*` names, `gguf-py/gguf/constants.py`
# `MODEL_ARCH_NAMES`) do NOT all equal our registry keys, and `registry._resolve`
# only knows OUR keys + HF class-name aliases — it cannot map `qwen35`→`qwen3_next`.
# This table bridges the gap; anything not listed falls through to `_resolve`
# (which handles the cases where the GGUF name already equals a registered key,
# e.g. `qwen3`, `gemma3`, `minimax`, `glm4moe`).
#
# Sources: research/llamacpp-hybrid-linattn.md §1 (qwen35/qwen3next), §5a;
#          research/llamacpp-moe.md §1.1 (arch registrations).
_GGUF_ARCH_ALIASES = {
    "qwen35": "qwen3_next",  # Qwen3.5 hybrid = our qwen3_next backbone (§5a)
    "qwen35moe": "qwen3_next",  # MoE variant of the same backbone
    "qwen3next": "qwen3_next",
    "qwen3moe": "qwen3",  # our qwen3 builder registers qwen3_moe/qwen3moe
    "deepseek2": "deepseek",
    "glm4moe": "glm",
    "hunyuan": "hunyuan",
    "lfm2moe": "lfm2",
    "gemma3": "gemma3",
    "diffusion-gemma": "diffusion_gemma",
}


def _resolve_arch(gguf_arch: str) -> str:
    """GGUF arch string → a ModelConfig.arch that `registry._resolve` accepts."""
    mapped = _GGUF_ARCH_ALIASES.get(gguf_arch.lower())
    if mapped is not None:
        return mapped
    # already one of our keys / an HF class-name alias?
    return gguf_arch if _resolve(gguf_arch) else gguf_arch


# ── robust GGUF KV reader (works across `gguf` lib versions) ──────────────────
def _field_value(field):
    """Extract a python value from a `gguf.ReaderField`, tolerant of lib version.

    Newer `gguf` exposes `ReaderField.contents()`; older releases require manual
    part indexing. We try the modern API first, then fall back to the canonical
    `parts[data[...]]` idiom (a string is one byte-array part; a scalar is the
    last part's [0]; an array is each element's part)."""
    if hasattr(field, "contents"):
        try:
            return field.contents()
        except Exception:  # pragma: no cover - version drift
            pass
    from gguf.constants import GGUFValueType

    if not field.types:
        return None
    vtype = field.types[0]
    if vtype == GGUFValueType.STRING:
        return str(bytes(field.parts[field.data[0]]), encoding="utf-8")
    if vtype == GGUFValueType.ARRAY:
        elem = field.types[1] if len(field.types) > 1 else None
        if elem == GGUFValueType.STRING:
            return [str(bytes(field.parts[i]), encoding="utf-8") for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    return field.parts[field.data[-1]].tolist()[0]


class _KV:
    """Thin accessor over `GGUFReader.fields` with an `<arch>.` prefix helper."""

    def __init__(self, fields, arch: str):
        self._f = fields
        self._arch = arch

    def get(self, key: str, default=None):
        f = self._f.get(key)
        return default if f is None else _field_value(f)

    def a(self, suffix: str, default=None):
        """Read an arch-scoped key, `<arch>.<suffix>`."""
        return self.get(f"{self._arch}.{suffix}", default)

    def has(self, key: str) -> bool:
        return key in self._f


def gguf_config(path: str) -> ModelConfig:
    """Read a GGUF's KV metadata and build the same `ModelConfig` that
    `ModelConfig.from_hf` produces for the model — sourced from GGUF-KV, no `.fni8`.

    LLMs: read the standard `<arch>.*` keys and synthesize an HF-shaped config dict,
    then hand it to `ModelConfig.from_hf` so ALL of from_hf's derivations (hybrid
    layer schedule, MoE routing, VLM detection) are reused verbatim (DRY, no
    second copy of the schema). DiT GGUFs carry no `<arch>.*` KV — only an opaque
    diffusers `config` JSON blob — so they route to `gguf_dit_config` (§2c)."""
    from gguf import GGUFReader

    reader = GGUFReader(path)
    fields = reader.fields
    gguf_arch = _field_value(fields["general.architecture"])

    # DiT (ComfyUI-GGUF convention, MIGRATION §2c): a single opaque diffusers config
    # blob, no structured `<arch>.*` KV. fni8-serve builds LLMs, not DiTs — point the
    # caller at the blob helper (ComfyUI-fni8's arch.py consumes it).
    if fields.get("general.config") is not None or fields.get("config") is not None:
        raise NotImplementedError(
            f"{path!r} is a DiT GGUF (arch={gguf_arch!r}): its diffusers config rides "
            "in an opaque `config` JSON blob, not <arch>.* KV. Use gguf_dit_config(path) "
            "(consumed by ComfyUI-fni8's arch registry), not the LLM gguf_config path."
        )

    kv = _KV(fields, gguf_arch)

    # ── dims: standard llama.cpp attention/block KV (gguf-py constants Keys) ──
    #   embedding_length / block_count / head_count[_kv] / feed_forward_length
    #   are the universal LLM keys convert_hf_to_gguf writes for every arch.
    hidden = kv.a("embedding_length")
    n_head = kv.a("attention.head_count")
    # head_dim: prefer the explicit key_length KV (present for GQA/hybrid models where
    # head_dim != hidden/head_count, e.g. qwen35 key_length=256); else hidden//heads.
    head_dim = kv.a("attention.key_length")

    # vocab: no `<arch>.vocab_size` KV exists — llama.cpp sizes the vocab from the
    # tokenizer token list (and the token_embd tensor). Read the token array length;
    # fall back to the embedding tensor's outer dim if the tokenizer is absent.
    vocab = None
    tok = fields.get("tokenizer.ggml.tokens")
    if tok is not None:
        vocab = len(tok.data)
    if not vocab:
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                vocab = int(max(t.shape))
                break

    # tie_word_embeddings: llama.cpp emits a separate `output.weight` only when the
    # head is UNTIED; a missing output.weight ⇒ tied (reuses token_embd). Detect from
    # the tensor list (header-only, no data read).
    tensor_names = {t.name for t in reader.tensors}
    tied = "output.weight" not in tensor_names

    # ── rope: freq_base → theta; dimension_count → partial-rotary factor ──
    #   qwen35.rope.freq_base=1e7, qwen35.rope.dimension_count=64 (< head_dim 256 ⇒
    #   partial rotary 0.25). A model with no rope.dimension_count rotates the full
    #   head_dim (partial factor 1.0).  (research/llamacpp-hybrid-linattn.md §1.5.)
    rope_theta = kv.a("rope.freq_base")
    rope_dim = kv.a("rope.dimension_count")
    partial_rotary = (rope_dim / head_dim) if (rope_dim and head_dim) else 1.0

    # ── hybrid gated-DeltaNet / SSM (qwen35/qwen3next): §2a, §5a ──
    #   full_attention_interval + the ssm.* param block mark the linear-attn layers.
    #   Map ssm.* back to the HF `linear_*` knobs the qwen3_next builder reads from
    #   config.extra (reverse of research/llamacpp-hybrid-linattn.md §1.5):
    #     linear_key_head_dim    ← ssm.state_size
    #     linear_num_key_heads   ← ssm.group_count
    #     linear_num_value_heads ← ssm.time_step_rank
    #     linear_conv_kernel_dim ← ssm.conv_kernel
    #     linear_value_head_dim  ← ssm.inner_size / ssm.time_step_rank
    full_interval = kv.a("full_attention_interval", 0) or 0
    ssm_state = kv.a("ssm.state_size")
    has_ssm = ssm_state is not None
    ssm_extra: dict = {}
    if has_ssm:
        n_v_heads = kv.a("ssm.time_step_rank")
        inner = kv.a("ssm.inner_size")
        ssm_extra = {
            "linear_key_head_dim": ssm_state,
            "linear_num_key_heads": kv.a("ssm.group_count"),
            "linear_num_value_heads": n_v_heads,
            "linear_conv_kernel_dim": kv.a("ssm.conv_kernel"),
            "linear_value_head_dim": (inner // n_v_heads) if (inner and n_v_heads) else None,
        }

    # ── MoE routing (research/llamacpp-moe.md §1.2, §4.3) ──
    #   expert_count/used_count/shared_count, expert_feed_forward_length,
    #   expert_weights_norm (norm_topk_prob), expert_gating_func (1=softmax,2=sigmoid),
    #   expert_group_count/used_count (DeepSeek-V3 group routing).
    n_experts = kv.a("expert_count", 0) or 0
    moe_extra: dict = {}
    gating_resolved = True
    if n_experts:
        gate_func = kv.a("expert_gating_func")  # 1=softmax 2=sigmoid 4=sqrt-softplus
        gating_resolved = gate_func is not None
        moe_extra = {
            "expert_gating_func": gate_func,
            "expert_group_count": kv.a("expert_group_count"),
            "expert_group_used_count": kv.a("expert_group_used_count"),
            "expert_weights_scale": kv.a("expert_weights_scale"),
        }

    # ── MTP / nextn (research/llamacpp-mtp-spec.md §1.2) ──
    #   `<arch>.nextn_predict_layers` is the ONLY MTP KV. Absent ⇒ the quantizer
    #   dropped the MTP head (the common case, §5c) ⇒ 0 (engine runs non-spec).
    n_mtp = kv.a("nextn_predict_layers", 0) or 0

    # ── synthesize an HF-shaped dict, then reuse ModelConfig.from_hf verbatim ──
    hf: dict = {
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": kv.a("block_count"),
        "num_attention_heads": n_head,
        "num_key_value_heads": kv.a("attention.head_count_kv", n_head),
        "intermediate_size": kv.a("feed_forward_length", 0) or 0,
        "head_dim": head_dim,
        "max_position_embeddings": kv.a("context_length", 32768),
        "rms_norm_eps": kv.a("attention.layer_norm_rms_epsilon", 1e-6),
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial_rotary,
        "tie_word_embeddings": tied,
        # hybrid: set the scalar `linear_attention` flag explicitly (from_hf only
        # derives it from HF `layer_types`/`attn_type_list`, which GGUF lacks — the
        # ssm.* block is our signal) plus the full-attn stride.
        "linear_attention": bool(has_ssm or full_interval),
        "full_attention_interval": full_interval,
        # MoE
        "num_experts": n_experts,
        "num_experts_per_tok": kv.a("expert_used_count", 0) or 0,
        "moe_intermediate_size": kv.a("expert_feed_forward_length", 0) or 0,
        "norm_topk_prob": bool(kv.a("expert_weights_norm", True)),
        "shared_expert_intermediate_size": kv.a("expert_shared_feed_forward_length", 0) or 0,
        # MTP
        "num_nextn_predict_layers": n_mtp,
        # divergent-family knobs → land in ModelConfig.extra for the builders
        **{k: v for k, v in ssm_extra.items() if v is not None},
        **{k: v for k, v in moe_extra.items() if v is not None},
    }

    cfg = ModelConfig.from_hf(hf, arch=_resolve_arch(gguf_arch))

    # HARD assert (MIGRATION §5b): a MoE that silently defaults to softmax when it
    # wants sigmoid group-routing selects the wrong experts → garbage. Never default.
    if n_experts and not gating_resolved:
        raise ValueError(
            f"{path!r}: MoE arch {gguf_arch!r} has {n_experts} experts but no "
            f"`{gguf_arch}.expert_gating_func` KV — refusing to default to softmax "
            "(sigmoid group-routing would be silently wrong). Needs a custom "
            "fni8.moe.* KV or an arch-hardcoded gate."
        )
    return cfg


def dequant_kquant(qt: QTensor) -> torch.Tensor:
    """Dequantize a `gguf_kquant` QTensor's raw bytes back to an fp16 `[out, in]`
    weight, via llama.cpp's own block dequant. Used by the LinearW8A8 fallback (for
    k-quant types with no fused dp4a kernel yet — Q5_K/Q6_K — and CPU tensors) and by
    the mixed-type merge path. Byte-faithful: no re-quant, just the native dequant."""
    from gguf import GGMLQuantizationType, dequantize

    tag = {"q4_k": "Q4_K", "q5_k": "Q5_K", "q6_k": "Q6_K"}[qt.codebook]
    arr = qt.data.detach().cpu().numpy()  # uint8 [out, n_superblocks*type_size]
    deq = dequantize(arr, GGMLQuantizationType[tag]).astype(np.float32)  # [out, in]
    return torch.from_numpy(np.ascontiguousarray(deq)).to(qt.data.device).half()


def gguf_state_dict(path: str, *, device: str = "cuda") -> dict:
    """Load a GGUF's tensors as a build-ready state dict, RESIDENT and native — no
    dequant→requant transcode of the k-quant weights (MIGRATION §3a):

      Q4_K/Q5_K/Q6_K  → `gguf_kquant` QTensor holding the RAW GGUF bytes (the fused
                        dp4a kernel unpacks in-kernel; Q4_K is live, Q5_K/Q6_K use
                        the LinearW8A8 dequant fallback until their kernels land).
      Q8_0            → `per_row_i8` (already int8 blocks; dequant→per-row-int8 is
                        near-lossless — the one benign requant, source is 8-bit).
      F32/F16/BF16    → raw fp16 Tensor (norms, router gate, embeddings).

    Names are translated GGUF→HF by `gguf_import.gguf_name_to_hf`. Non-linear tensors
    (norms/router/embeddings, per `convert.is_quantizable_linear`) are returned as
    plain fp16 Tensors — exactly what the builders expect (LinearW8A8 wants a QTensor,
    RMSNorm/VocabEmbedding want a Tensor), matching `loader.load_fni8_state_dict`."""
    from gguf import GGMLQuantizationType, GGUFReader, dequantize

    from .convert import is_quantizable_linear
    from .gguf_import import gguf_name_to_hf

    reader = GGUFReader(path)
    out: dict = {}
    for t in reader.tensors:
        hf = gguf_name_to_hf(t.name)
        if hf is None:
            continue  # unmapped (e.g. an SSM/MoE tensor pending the P3 remap)
        gtype = GGMLQuantizationType(t.tensor_type).name
        quantizable = is_quantizable_linear(hf)

        if quantizable and gtype in _KQUANT:
            code, tsz = _KQUANT[gtype]
            data = torch.from_numpy(np.ascontiguousarray(t.data).copy()).to(device)  # uint8 [out, n_sb*tsz]
            out[hf] = QTensor(data, None, scheme="gguf_kquant", codebook=code, group_size=256)
        elif quantizable and gtype == "Q8_0":
            deq = dequantize(t.data, GGMLQuantizationType.Q8_0).astype(np.float32)
            w = torch.from_numpy(np.ascontiguousarray(deq)).to(device)
            from fni8.quant.core import quantize_int8_rowwise

            q, s = quantize_int8_rowwise(w)
            out[hf] = QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")
        else:
            # float weights + all non-linear (norms/router/embeddings) → fp16 Tensor.
            if gtype in _FLOAT_TYPES:
                w = torch.from_numpy(np.ascontiguousarray(t.data).copy()).to(device).half()
            else:  # a quantized non-linear (rare) — dequant to fp16
                deq = dequantize(t.data, GGMLQuantizationType(t.tensor_type)).astype(np.float32)
                w = torch.from_numpy(np.ascontiguousarray(deq)).to(device).half()
            out[hf] = w
    return out


def gguf_dit_config(path: str) -> dict:
    """Parse a DiT GGUF's opaque diffusers `config` JSON blob (ComfyUI-GGUF
    convention, MIGRATION §2c). Returns the raw diffusers config dict; the DiT arch
    registry (ComfyUI-fni8 `arch.py`) keys off it. Header-only, no tensor load."""
    from gguf import GGUFReader

    fields = GGUFReader(path).fields
    blob = fields.get("general.config") or fields.get("config")
    if blob is None:
        raise ValueError(f"{path!r} carries no `config`/`general.config` DiT blob")
    return json.loads(_field_value(blob))
