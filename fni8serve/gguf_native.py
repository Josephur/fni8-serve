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
    "Q2_K": ("q2_k", 84),
    "Q3_K": ("q3_k", 110),
    "Q4_K": ("q4_k", 144),
    "Q5_K": ("q5_k", 176),
    "Q6_K": ("q6_k", 210),
}
_FLOAT_TYPES = {"F32", "F16", "BF16"}


def _native_kquant_types() -> tuple[str, ...]:
    """Return GGUF types backed by the installed fni8 fused kernels.

    Probe the public operations directly.  The previous source-text probe only
    recognized Q4_K, so Q5_K/Q6_K were needlessly expanded to float and requantized
    to W8 during every load even after their native kernels shipped.
    """
    import fni8

    ops = {
        "Q2_K": "linear_q2k",
        "Q3_K": "linear_q3k",
        "Q4_K": "linear_q4k",
        "Q5_K": "linear_q5k",
        "Q6_K": "linear_q6k",
    }
    return tuple(kind for kind, op in ops.items() if callable(getattr(fni8, op, None)))


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
    "qwen35": "qwen3_5",  # Qwen3.5 hybrid dense (9B) → our qwen3_5 builder
    "qwen35moe": "qwen3_next",  # MoE variant → qwen3_next backbone
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


def _open_gguf(path: str):
    """Open one GGUF reader for all header and tensor phases of a cold load."""
    from gguf import GGUFReader

    return GGUFReader(path)


def gguf_config(path: str, *, _reader=None) -> ModelConfig:
    """Read a GGUF's KV metadata and build the same `ModelConfig` that
    `ModelConfig.from_hf` produces for the model — sourced from GGUF-KV, no `.fni8`.

    LLMs: read the standard `<arch>.*` keys and synthesize an HF-shaped config dict,
    then hand it to `ModelConfig.from_hf` so ALL of from_hf's derivations (hybrid
    layer schedule, MoE routing, VLM detection) are reused verbatim (DRY, no
    second copy of the schema). DiT GGUFs carry no `<arch>.*` KV — only an opaque
    diffusers `config` JSON blob — so they route to `gguf_dit_config` (§2c)."""
    reader = _reader if _reader is not None else _open_gguf(path)
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
        "num_hidden_layers": kv.a("block_count") - n_mtp,
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

    tag = {
        "q2_k": "Q2_K",
        "q3_k": "Q3_K",
        "q4_k": "Q4_K",
        "q5_k": "Q5_K",
        "q6_k": "Q6_K",
    }[qt.codebook]
    arr = qt.data.detach().cpu().numpy()  # uint8 [out, n_superblocks*type_size]
    deq = dequantize(arr, GGMLQuantizationType[tag]).astype(np.float32)  # [out, in]
    return torch.from_numpy(np.ascontiguousarray(deq)).to(qt.data.device).half()


def gguf_state_dict(
    path: str,
    *,
    device: str = "cuda",
    native_types: tuple[str, ...] = ("Q4_K", "Q5_K", "Q6_K"),
    _reader=None,
) -> dict:
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
    RMSNorm/VocabEmbedding want a Tensor), matching `loader.load_fni8_state_dict`.

    `native_types` selects which k-quant types stay RESIDENT as raw `gguf_kquant`
    bytes (default: all three — the native no-transcode residency PR-2 delivers). A
    type NOT in the set is dequantized to fp16 and re-quantized to `per_row_i8` for
    the proven dp4a W8A8 path. `load_gguf_engine` narrows this to the k-quant types
    whose FUSED dp4a kernel is actually built (MIGRATION P0), so decode runs fast on
    real dp4a today and flips to native residency automatically as the kernels land —
    the LinearW8A8 dequant fallback is per-forward and far too slow for a full model."""
    from gguf import GGMLQuantizationType, dequantize

    from .convert import is_quantizable_linear
    from .gguf_import import gguf_name_to_hf

    def _to_per_row_i8(np_fp32) -> QTensor:
        from fni8.quant.core import quantize_int8_rowwise

        w = torch.from_numpy(np.ascontiguousarray(np_fp32)).to(device)
        q, s = quantize_int8_rowwise(w)
        return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")

    reader = _reader if _reader is not None else _open_gguf(path)
    out: dict = {}
    for t in reader.tensors:
        hf = gguf_name_to_hf(t.name)
        if hf is None:
            continue  # unmapped (e.g. an SSM/MoE tensor pending the P3 remap)
        gtype = GGMLQuantizationType(t.tensor_type).name
        quantizable = is_quantizable_linear(hf)

        if quantizable and gtype in _KQUANT and gtype in native_types:
            code, _tsz = _KQUANT[gtype]
            data = torch.from_numpy(np.ascontiguousarray(t.data).copy()).to(device)
            out[hf] = QTensor(data, None, scheme="gguf_kquant", codebook=code, group_size=256)
        elif quantizable and gtype in _KQUANT:  # k-quant with no fused kernel → int8 dp4a
            out[hf] = _to_per_row_i8(
                dequantize(t.data, GGMLQuantizationType[gtype]).astype(np.float32)
            )
        elif quantizable and gtype == "Q8_0":
            out[hf] = _to_per_row_i8(
                dequantize(t.data, GGMLQuantizationType.Q8_0).astype(np.float32)
            )
        else:
            # float weights + all non-linear (norms/router/embeddings) → fp16 Tensor.
            if gtype in _FLOAT_TYPES:
                w = torch.from_numpy(np.ascontiguousarray(t.data).copy()).to(device).half()
            else:  # a quantized non-linear (rare) — dequant to fp16
                deq = dequantize(t.data, GGMLQuantizationType(t.tensor_type)).astype(np.float32)
                w = torch.from_numpy(np.ascontiguousarray(deq)).to(device).half()
            out[hf] = w
    return out


def _remap_hybrid_qwen35(sd: dict, cfg) -> dict:
    """Post-process a GGUF state dict for Qwen3.5 hybrid models.

    The llama.cpp qwen35 GGUF layout uses fused `attn_qkv` and `attn_gate`
    tensors that need layer-type-dependent remapping:

    * **DeltaNet layers** (linear_attention): `attn_gate.weight` →
      `linear_attn.in_proj_a.weight` (the dt/decay projection).  The fused
      `attn_qkv.weight` is already mapped to `linear_attn.in_proj_qkv.weight`
      by `gguf_name_to_hf`.

    * **Full-attention layers**: `attn_q.weight` + `attn_gate.weight` must be
      FUSED into `self_attn.q_proj.weight` (the gated-QAttention convention:
      q_proj carries [query | gate] on the output axis).  K/V stay separate.

    This runs ONCE after `gguf_state_dict` and before the model builder.
    Only touches layers whose GGUF tensors are present (safe for non-hybrid
    models — returns `sd` unchanged if no `self_attn.attn_gate` keys exist).
    """
    # Quick bail: no attn_gate tensors → nothing to remap.
    gate_keys = [k for k in sd if k.endswith(".self_attn.attn_gate.weight")]
    if not gate_keys:
        return sd

    for gk in gate_keys:
        prefix = gk.rsplit(".self_attn.attn_gate.weight", 1)[0]
        layer_idx = int(prefix.split(".")[-1])
        kind = cfg.attention_kind(layer_idx)
        gate = sd.pop(gk)

        if kind == "linear":
            # DeltaNet: attn_gate → linear_attn.in_proj_z (z gate, output gate)
            sd[f"{prefix}.linear_attn.in_proj_z.weight"] = gate
        else:
            # Full attention: fuse gate into q_proj.
            # HF q_proj = [query_heads | gate_heads] on the output axis.
            qk = f"{prefix}.self_attn.q_proj.weight"
            q = sd.pop(qk)
            # gate is [H, H] (one scalar gate per head, broadcast to head_dim),
            # q is [nh*hd, H].  Concat on output axis → [(nh*hd + nh*hd), H].
            sd[qk] = torch.cat([q, gate], dim=0)
    return sd


def _restore_hf_qwen35_weights(sd: dict, cfg) -> tuple[dict, bool]:
    """Undo llama.cpp-only Qwen3.5 transforms before using the HF-layout builder.

    llama.cpp stores zero-centered RMSNorm gains with one added and stores the
    DeltaNet decay parameter as ``-exp(A_log)``. Its unequal K/V-head tensors also
    remain in tiled V-head order; preserving that order avoids expanding native
    k-quant weights merely to permute columns, and the mixer selects the matching
    tiled Q/K broadcast.
    """
    if not getattr(cfg, "linear_attention", False):
        return sd, False

    for name, weight in list(sd.items()):
        if name.endswith("norm.weight") and ".linear_attn.norm.weight" not in name:
            sd[name] = weight - 1
        elif name.endswith(".linear_attn.A_log"):
            sd[name] = (-weight.float()).clamp_min(1e-30).log().to(weight.dtype)

    x = getattr(cfg, "extra", {})
    tiled = x.get("linear_num_key_heads") != x.get("linear_num_value_heads")
    return sd, tiled


def _gguf_eos_id(path: str, *, _reader=None) -> int | None:
    """Read the GGUF tokenizer's EOS id (`tokenizer.ggml.eos_token_id`) so decode can
    stop naturally, mirroring the `.fni8` load path's `eos_id`."""
    reader = _reader if _reader is not None else _open_gguf(path)
    f = reader.fields.get("tokenizer.ggml.eos_token_id")
    return None if f is None else int(_field_value(f))


def load_gguf_engine(
    path: str,
    *,
    device: str = "cuda",
    max_num_seqs: int = 16,
    max_len: int = 2048,
    eos_id: int | None = None,
    spec_decode: bool | None = None,
):
    """Build an `LLMEngine` straight from a `.gguf` — the P1 milestone: end-to-end
    single-stream decode with ZERO `.fni8`. Mirrors `api.server.load_engine`'s `.fni8`
    seam: `gguf_config` + `gguf_state_dict` → `LLMEngine`, over the unchanged,
    format-agnostic builders + engine.

    **Speculative decode** is wired through the engine's EXISTING spec path (the same
    n-gram cascade + `GraphedVerify` the `.fni8` path uses), so a GGUF-loaded model
    gets it for free. Which drafter engages is chosen from the GGUF's OWN metadata:

      * ``num_mtp_layers > 0`` (the GGUF carries ``nextn.*`` MTP tensors) → the learned
        MTP head is built (``model.mtp``) and drives the draft; the n-gram lookup stays
        as the cascade's zero-cost first stage.
      * ``num_mtp_layers == 0`` (the common case — quantizers strip ``nextn.*``, e.g.
        Qwen3-8B-Q4_K_M) → no head is built (``model.mtp is None``); the drafter cleanly
        FALLS BACK to the model-agnostic n-gram lookup, which needs no extra weights.

    ``spec_decode`` (None → `FNI8SERVE_MTP_SPEC` env, default off) turns the path on;
    it stays OFF by default so a GGUF load never regresses vs plain greedy on the
    current M=1 decode kernel (spec's net win lands with the warp-per-column kernel).
    """
    import dataclasses

    from .engine import LLMEngine

    reader = _open_gguf(path)
    cfg = gguf_config(path, _reader=reader)
    # Keep k-quant weights RESIDENT (native gguf_kquant) only for types whose fused
    # dp4a kernel is built; otherwise dequant→per_row_i8 so decode runs on real dp4a
    # NOW (the per-forward dequant fallback is far too slow for a whole model).
    native_types = _native_kquant_types()
    weights = gguf_state_dict(path, device=device, native_types=native_types, _reader=reader)

    # ── Unsloth GGUF [in, out] → [out, in] transposition ──────────────────────
    # The Unsloth GGUF converter (used for the Qwen3.5-9B UD-Q4_K_XL checkpoint)
    # stores 2-D weight tensors as [in_features, out_features], transposed from
    # PyTorch's [out_features, in_features] convention.  Detect by checking if
    # token_embd has shape [hidden_size, vocab_size] (wrong) instead of
    # [vocab_size, hidden_size] (correct).  When detected, transpose every2-D
    # tensor so the builders and kernels see the expected layout.
    _embed = weights.get("model.embed_tokens.weight")
    if _embed is not None and _embed.dim() == 2 and _embed.shape[0] == cfg.hidden_size:
        import logging

        _log = logging.getLogger("fni8serve.gguf_native")
        _log.info("GGUF weights are [in, out] (Unsloth convention) — transposing all 2D tensors")
        _new = {}
        for k, v in weights.items():
            if isinstance(v, QTensor):
                if v.scheme == "gguf_kquant":
                    # K-quant raw bytes: the kernel unpacks from the block layout
                    # directly.  Transpose the logical shape metadata but keep the
                    # bytes intact — the kernel determines N/K from the blocks.
                    _new[k] = QTensor(
                        v.data,
                        v.scale,
                        scheme=v.scheme,
                        codebook=v.codebook,
                        group_size=v.group_size,
                    )
                    # Swap the logical [in, blocks] → [blocks, in] by reshaping.
                    # Actually, k-quant QTensor data is flat uint8 — no2-D shape
                    # to transpose.  The kernel ignores the tensor shape entirely.
                    pass  # leave as-is
                elif v.scheme == "per_row_i8":
                    # Dequantize → transpose → re-quantize.  Expensive but correct:
                    # the per-row scales must correspond to the new rows.
                    from .convert import quantize_weight_i8

                    fp = (v.data.float() * v.scale.unsqueeze(-1)).half()
                    _new[k] = quantize_weight_i8(fp.T.contiguous())
                else:
                    _new[k] = v  # raw — shouldn't happen for2-D but be safe
            elif isinstance(v, torch.Tensor) and v.dim() == 2:
                _new[k] = v.T.contiguous()
            else:
                _new[k] = v
        weights = _new

    # Hybrid Qwen3.5/3.6: remap the llama.cpp qwen35 GGUF layout (fused attn_qkv,
    # attn_gate) to the HF naming the qwen3_5 builder expects.  No-op for non-hybrid
    # models (no attn_gate keys → returns immediately).
    if cfg.linear_attention:
        weights = _remap_hybrid_qwen35(weights, cfg)

    # DeltaNet in_proj_* → qkv_proj / z_proj / beta_proj / dt_proj / conv_weight:
    # the same rename the .fni8 conversion path applies (convert._remap_qwen3_next).
    # Needed because the GGUF name mapping produces `in_proj_*` names, but the
    # qwen3_5 builder reads `qkv_proj` / `z_proj` / etc.
    from .convert import _remap_qwen3_next

    weights = _remap_qwen3_next(weights, cfg)

    # llama.cpp's qwen35 converter changes norm/decay semantics and, when K/V head
    # counts differ, stores DeltaNet V heads in tiled order. Restore scalar semantics
    # and tell the mixer which broadcast matches the still-native k-quant row layout.
    weights, tiled_delta_heads = _restore_hf_qwen35_weights(weights, cfg)
    if tiled_delta_heads:
        cfg = dataclasses.replace(cfg, extra={**cfg.extra, "gguf_tiled_linear_attention": True})

    # qk_norm is detect-from-tensors (no GGUF KV for it): Qwen3/Gemma3 carry per-head
    # q_norm/k_norm and skipping them feeds un-normalized Q/K into RoPE → garbage.
    # Same guard api.server.load_engine applies on the .fni8 path.
    if not getattr(cfg, "qk_norm", False) and any(".q_norm.weight" in n for n in weights):
        cfg = dataclasses.replace(cfg, qk_norm=True)

    if eos_id is None:
        eos_id = _gguf_eos_id(path, _reader=reader)

    # MTP-head detection (research/llamacpp-mtp-spec.md §1.2): the head is built by the
    # arch builder iff `cfg.num_mtp_layers > 0` (it reads the GGUF `nextn_predict_layers`
    # KV). Log which drafter the spec path will use so an absent MTP head (stripped by
    # the quantizer) is an explicit, greppable fallback to n-gram — never a silent one.
    import logging

    _log = logging.getLogger("fni8serve.gguf_native")
    if cfg.num_mtp_layers > 0:
        _log.info(
            "GGUF carries %d MTP/nextn layer(s) → MTP head wired as spec drafter "
            "(n-gram stays the cascade's free first stage).",
            cfg.num_mtp_layers,
        )
    else:
        _log.info(
            "GGUF has no nextn.* MTP tensors (num_mtp_layers=0) → spec-decode falls "
            "back to the model-agnostic n-gram cascade drafter (no MTP head)."
        )

    return LLMEngine(
        cfg,
        weights,
        device=device,
        max_num_seqs=max_num_seqs,
        max_len=max_len,
        eos_id=eos_id,
        spec_decode=spec_decode,
        consume_weights=True,
    )


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
