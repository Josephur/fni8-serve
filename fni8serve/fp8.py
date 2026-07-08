# SPDX-License-Identifier: MIT
"""Dequantize FP8 checkpoints so they can be re-quantized to `.fni8` int8.

Big MoE releases (DeepSeek-V3/V4, Tencent Hy3) ship **FP8-only** — often the only
weights available, and always ~half the download of bf16. FP8 (`float8_e4m3fn`) is
lossy storage plus a companion scale; to reach our int8 dp4a format we first
reconstruct the real values (`fp8 * scale`), then feed fp32 to the int8 quantizer.

Two scale layouts are handled (detected from `config.json`'s `quantization_config`):

- **static / per-channel** (`weight_block_size` absent, e.g. Hy3-FP8): one
  `weight_scale` per tensor (scalar) or per output row (`[O]`/`[O,1]`).
- **block** (`weight_block_size=[bh,bw]`, e.g. DeepSeek-V3.1): a `weight_scale_inv`
  grid `[ceil(O/bh), ceil(I/bw)]`, each entry scaling a `bh×bw` tile. Despite the
  ``_inv`` name it is the dequant MULTIPLIER (`real = fp8 * scale_inv`).

`dequantize_fp8` is pure/tested; `is_fp8` gates the convert path.
"""
from __future__ import annotations

import torch

# torch exposes float8_e4m3fn (and e5m2); e4m3 is what these checkpoints use.
_FP8_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if d is not None
)


def is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in _FP8_DTYPES


def dequantize_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: list[int] | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Reconstruct fp32 values from an fp8 `weight` [O, I] and its `scale`.

    `block_size=[bh,bw]` -> block layout (`scale` is the `[O/bh, I/bw]` grid);
    otherwise static/per-channel (`scale` scalar, `[O]`, or `[O,1]`)."""
    wf = weight.to(torch.float32)
    sc = scale.to(torch.float32)

    if block_size is None:
        if sc.numel() == 1:
            return wf * sc.reshape(())
        # per-output-channel: one scale per row
        return wf * sc.reshape(-1, 1)

    if wf.dim() != 2:
        raise ValueError(f"block fp8 dequant expects a 2-D weight, got {tuple(wf.shape)}")
    bh, bw = int(block_size[0]), int(block_size[1])
    o, i = wf.shape
    # Expand the [ceil(O/bh), ceil(I/bw)] grid to [O, I], clipping the ragged tail.
    full = sc.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)[:o, :i]
    if full.shape != wf.shape:
        raise ValueError(f"fp8 block scale {tuple(sc.shape)} x{block_size} -> "
                         f"{tuple(full.shape)} != weight {tuple(wf.shape)}")
    return wf * full


def fp8_scale_name(weight_name: str, names: set[str]) -> str | None:
    """Find the scale tensor paired with an fp8 `weight_name` (`X.weight` ->
    `X.weight_scale_inv` for block, or `X.weight_scale` for static)."""
    for suf in (".weight_scale_inv", ".weight_scale"):
        cand = weight_name[: -len(".weight")] + suf if weight_name.endswith(".weight") else None
        if cand and cand in names:
            return cand
    return None
