# SPDX-License-Identifier: MIT
"""Linear seam — W8A8 / W4A8 dp4a matmul, fed from `.fni8` weights.

The weight is int8 (`per_row_i8`) or 4-bit (`per_group_i4`, int4/NF4) straight from
the `.fni8` container — no dequant at load. At runtime the activation is quantized to
int8 per-token and the matmul runs on dp4a.

STATUS: the fused int8/W4A8 dp4a GEMM op is the #1 kernel still to build in `fni8`.
Until it lands, `forward` uses a NUMERICALLY-CORRECT fp16 fallback (reconstruct the
weight, torch.matmul) so the whole server runs end-to-end today; flipping to dp4a is a
one-line swap to `fni8.int8_gemm(x_i8, w_i8, ...)`. The fallback is marked SLOW.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fni8 import QTensor

try:  # NF4 codebook for 4-bit weight reconstruction (present in fni8)
    from fni8.quant.lowbit import NF4_CODEBOOK
except Exception:  # pragma: no cover
    NF4_CODEBOOK = None


def _dequant_weight(qt: QTensor) -> torch.Tensor:
    """Reconstruct an fp16 weight [out, in] from a stored QTensor (fallback path)."""
    if qt.scheme == "per_row_i8":
        return (qt.data.float() * qt.scale.unsqueeze(-1)).to(torch.float16)
    if qt.scheme == "per_group_i4":
        packed = qt.data.to(torch.int32)                       # [out, in//2]
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        codes = torch.stack([lo, hi], -1).reshape(qt.data.shape[0], -1)  # [out, in], 0..15
        if qt.codebook == "nf4":
            cb = torch.tensor(NF4_CODEBOOK, device=codes.device, dtype=torch.float32)
            vals = cb[codes.long()]
        else:  # signed int4 two's-complement
            vals = codes.float() - 16.0 * (codes >= 8).float()
        g = qt.group_size
        out, in_ = vals.shape
        scale = qt.scale.reshape(out, in_ // g, 1)
        return (vals.reshape(out, in_ // g, g) * scale).reshape(out, in_).to(torch.float16)
    raise ValueError(f"LinearW8A8 cannot use scheme {qt.scheme!r}")


class LinearW8A8(nn.Module):
    def __init__(self, weight: QTensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.weight = weight                       # QTensor (int8 or i4), resident dp4a layout
        self.bias = bias
        self.out_features = weight.data.shape[0]
        # in_features: int8 -> data.shape[1]; i4 packs 2/byte
        self.in_features = weight.data.shape[1] * (2 if weight.scheme == "per_group_i4" else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(fni8 GEMM): x_i8, x_scale = quantize per-token int8;
        #                  y = fni8.int8_gemm(x_i8, self.weight.data, x_scale, self.weight.scale, ...)
        w = _dequant_weight(self.weight).to(x.dtype)   # SLOW fp16 fallback (no dp4a yet)
        y = torch.nn.functional.linear(x, w, self.bias)
        return y
