# SPDX-License-Identifier: MIT
"""HF checkpoint -> `.fni8` conversion.

Quantizes the linear weights (attention + MLP + experts + untied LM head) to int8
`per_row_i8` or 4-bit `per_group_i4`, and keeps the numerically load-bearing
tensors (all norms, embeddings, and the MoE router gate) in fp16 `raw`. Weights are
stored under their HF names; the model builders merge q/k/v and gate/up at load
(cheap concat, valid per-row). The on-disk bytes are the resident dp4a layout, so
serving loads with no dequant/repack (`fni8.format`).

`quantize_state_dict` is the file-free core (tested directly). `convert_hf_to_fni8`
wraps it with config.json + safetensors IO.
"""
from __future__ import annotations

import json
import os

import torch

from fni8 import QTensor, save_fni8
from fni8.quant.core import quantize_int8_rowwise
from fni8.quant.lowbit import quantize_lowbit

from .models.config import ModelConfig

# A weight is a quantizable linear iff its name ends with one of these. The MoE
# router (`mlp.gate.weight`) ends with `.gate.weight` (NOT `.gate_proj.weight`), so
# it correctly stays fp16.
# DENYLIST, not an allowlist: quantize EVERY 2-D matmul weight (the caller also
# checks `w.dim()==2 and w.shape[-1]%4==0`) EXCEPT the numerically-sensitive /
# non-matmul ones below. A name allowlist silently skipped any model with
# non-standard naming (LFM2 `feed_forward.w1/2/3`, other MoE/hybrid layouts),
# leaving the bulk of the model fp16 -> "4-bit" files nearly fp16-sized. For the
# standard Llama/Qwen names this denylist quantizes the identical set (q/k/v/o,
# gate/up/down, lm_head); it just no longer misses everything else.
#   - `embed`/`wte`/`wpe`: token embeddings are an index LOOKUP, not a matmul
#     (the serve embedding layer reads them as fp) -> keep raw. (Tied lm_head is
#     quantized separately on the serve side.)
#   - `norm`: layernorm/rmsnorm gains (tiny, load-bearing).
#   - MoE `router`/`.gate.weight`/`.wg.weight`: routing logits are sensitive and
#     tiny (NOT the MLP `gate_proj`, which stays quantizable).
_QUANT_DENY_SUBSTR = ("norm", "embed", "wte", "wpe", "rotary", "router", "relative_attention")
_QUANT_DENY_SUFFIX = (".gate.weight", ".router.weight", ".wg.weight")


def is_quantizable_linear(name: str) -> bool:
    n = name.lower()
    if any(s in n for s in _QUANT_DENY_SUBSTR):
        return False
    return not n.endswith(_QUANT_DENY_SUFFIX)


def quantize_weight_i8(w: torch.Tensor) -> QTensor:
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def quantize_weight_i4(w: torch.Tensor, group_size: int) -> QTensor:
    codes, scale = quantize_lowbit(w, 4, dim=-1, group_size=group_size)   # [O,I], [O,I//g]
    c = codes.to(torch.int64)
    packed = ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    return QTensor(packed.contiguous(), scale.float().contiguous(), scheme="per_group_i4",
                   group_size=group_size, codebook="int4")


_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_inv", ".input_scale")


def quantize_state_dict(sd: dict, *, weight_bits: int = 8, group_size: int = 128,
                        fp8_scales: dict | None = None,
                        fp8_block_size: list | None = None,
                        fp8_scale_fmt: str | None = None) -> dict:
    """HF state dict (fp16/bf16/**fp8**) -> dict[name, QTensor]. Linears quantized
    from full precision; everything else stored raw fp16 (fp32 only if fp16 would
    overflow — see _raw_dtype).

    FP8 checkpoints (DeepSeek/Hy3): each `float8` weight is first reconstructed to
    fp32 via its companion scale (`fp8_scales`, pre-collected so it works even when
    the scale lives in another shard), then quantized as usual. Scale tensors
    themselves are consumed, not emitted. `fp8_scale_fmt="ue8m0"` (microscaling /
    MXFP8 checkpoints, e.g. DeepSeek-V4-Flash, MiniMax-M3-MXFP8) selects the
    power-of-2 MX dequant (`dequantize_mxfp8`) instead of the fp32-multiply block
    dequant, and additionally drops the MX-specific scale-tensor names (`.scale`,
    `.hc_attn_scale`, `.hc_ffn_scale`, `.hc_head_scale`) — gated on `fp8_scale_fmt`
    being set (i.e. only for a checkpoint already known to be fp8-sourced) so a
    plain `.scale`-suffixed tensor in an unrelated fp16/bf16 model is never
    silently dropped."""
    from .fp8 import MX_SCALE_SUFFIXES, dequantize_fp8, dequantize_mxfp8, fp8_scale_name, is_fp8

    fp8_scales = fp8_scales or {}
    out: dict[str, QTensor] = {}
    for name, w in sd.items():
        if name.endswith(_SCALE_SUFFIXES):
            continue                                    # consumed by its weight / activation scale
        if fp8_scale_fmt is not None and name.endswith(MX_SCALE_SUFFIXES):
            continue                                    # consumed MX/UE8M0 companion scale
        w = w.detach().cpu()                            # keep NATIVE dtype (don't truncate bf16)
        if is_fp8(w):                                   # reconstruct fp32 from fp8 * scale
            sname = fp8_scale_name(name, set(fp8_scales) | set(sd))
            sc = fp8_scales.get(sname) if sname else (sd.get(sname) if sname else None)
            if sc is None:
                w = w.float()                           # no scale found -> best-effort raw upcast
            elif fp8_scale_fmt == "ue8m0":
                w = dequantize_mxfp8(w, sc.detach().cpu(), block_size=fp8_block_size or (128, 128))
            else:
                w = dequantize_fp8(w, sc.detach().cpu(), block_size=fp8_block_size)
        if is_quantizable_linear(name) and w.dim() == 2 and w.shape[-1] % 4 == 0:
            wf = w.float()                              # quantize from full precision
            if weight_bits == 4 and w.shape[-1] % group_size == 0:
                out[name] = quantize_weight_i4(wf, group_size)
            else:
                out[name] = quantize_weight_i8(wf)
        else:
            out[name] = QTensor(_raw_dtype(w), None, scheme="raw")
    return out


def _raw_dtype(w: torch.Tensor) -> torch.Tensor:
    """Store a raw passthrough tensor (norm/embedding/router) as fp16 — which has MORE
    mantissa than bf16, only less range — and upcast to fp32 ONLY if fp16 would
    overflow a finite value. bf16-native models (Gemma, most DiTs) can carry
    out-of-fp16-range tensors that otherwise become inf -> NaN / black output."""
    w16 = w.to(torch.float16)
    if torch.isinf(w16).any() and not torch.isinf(w.float()).any():
        return w.float()
    return w16


def convert_hf_to_fni8(hf_dir: str, out_path: str, *, weight_bits: int = 8,
                       group_size: int = 128, arch: str | None = None) -> ModelConfig:
    """Read an HF model dir (config.json + *.safetensors) and write a `.fni8`.
    Returns the ModelConfig (also embedded in the file meta).

    Shards are streamed one at a time — load, quantize, drop the raw shard, next —
    so peak RAM is the quantized-so-far dict plus a single raw shard, never the full
    fp16/bf16 model (the 220GB-model-in-RAM OOM that killed GLM-4.5-Air mid-batch).
    Each HF tensor lives wholly in one shard, and q/k/v & gate/up stay unmerged on
    disk (the model builders merge them at load time, see fni8serve/models/weights.py),
    so per-shard quantization needs no cross-shard state."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with open(os.path.join(hf_dir, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = ModelConfig.from_hf(hf_cfg, arch=arch)
    cfg.weight_bits = weight_bits

    shards = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors in {hf_dir}")

    # FP8 source? Pre-collect the (tiny) scale tensors across ALL shards first, so a
    # weight can be dequantized even when its scale lives in a different shard.
    qcfg = hf_cfg.get("quantization_config") or {}
    is_fp8_src = str(qcfg.get("quant_method", "")).lower() == "fp8"
    fp8_block_size = qcfg.get("weight_block_size") if is_fp8_src else None
    # "ue8m0" (UE8M0 block scales) marks a microscaling (MX) FP8 checkpoint, e.g.
    # DeepSeek-V4-Flash / MiniMax-M3-MXFP8 — see fp8.dequantize_mxfp8.
    fp8_scale_fmt = str(qcfg.get("scale_fmt", "")).lower() if is_fp8_src else None
    fp8_scales: dict = {}
    if is_fp8_src:
        from .fp8 import MX_SCALE_SUFFIXES
        scale_suffixes = _SCALE_SUFFIXES + MX_SCALE_SUFFIXES
        for shard in shards:
            with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
                for k in f.keys():
                    if k.endswith(scale_suffixes):
                        fp8_scales[k] = f.get_tensor(k)

    qsd: dict[str, QTensor] = {}
    for shard in shards:
        sd = load_file(os.path.join(hf_dir, shard))
        qsd.update(quantize_state_dict(sd, weight_bits=weight_bits, group_size=group_size,
                                       fp8_scales=fp8_scales, fp8_block_size=fp8_block_size,
                                       fp8_scale_fmt=fp8_scale_fmt))
        del sd

    meta = {"arch": cfg.arch, "weight_bits": weight_bits, "fp8_source": is_fp8_src,
            "config": {k: v for k, v in vars(cfg).items() if not isinstance(v, dict)}}
    save_fni8(out_path, qsd, meta=meta)
    return cfg


def main():
    """CLI: python -m fni8serve.convert <hf_dir> <out.fni8> [--bits 8|4] [--group 128]"""
    import argparse

    ap = argparse.ArgumentParser(description="Convert an HF checkpoint to .fni8")
    ap.add_argument("hf_dir")
    ap.add_argument("out_path")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--arch", default=None)
    a = ap.parse_args()
    cfg = convert_hf_to_fni8(a.hf_dir, a.out_path, weight_bits=a.bits,
                             group_size=a.group, arch=a.arch)
    print(f"wrote {a.out_path}  arch={cfg.arch}  bits={a.bits}  layers={cfg.num_hidden_layers}")


if __name__ == "__main__":
    main()
