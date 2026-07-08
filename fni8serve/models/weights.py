# SPDX-License-Identifier: MIT
"""Weight helpers shared by the concrete models.

A model builder receives a dict of fp16 tensors (an HF-style state dict, or the
raw tensors from a `.fni8` file) and turns linear weights into int8 `per_row_i8`
QTensors for the dp4a GEMM. QKV and gate/up projections are MERGED here (concat on
the output axis) so the runtime issues fewer, wider matmuls — the same merge the
offline `.fni8` conversion performs. Norms/embeddings stay fp16.
"""
from __future__ import annotations

import torch

from fni8 import QTensor
from fni8.quant.core import quantize_int8_rowwise


def to_qtensor(w) -> QTensor:
    """fp16/fp32 weight [out, in] (in % 4 == 0) -> per_row_i8 QTensor. Idempotent:
    an already-quantized QTensor (from a `.fni8` load) passes through unchanged, so
    the same model builders serve both runtime-quant (fp16 in) and offline (.fni8)."""
    if isinstance(w, QTensor):
        return w
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def merge_qtensor(rows: list) -> QTensor:
    """Fuse q/k/v -> qkv (and gate/up -> gate_up) on the output (row) axis. For fp16
    rows: concat then quantize as one. For pre-quantized QTensors: concat the int8/
    int4 data AND the per-row scales along the row axis (each output row keeps its
    own scale, so the fused weight is exact)."""
    if isinstance(rows[0], QTensor):
        data = torch.cat([r.data for r in rows], dim=0)
        scale = torch.cat([r.scale for r in rows], dim=0)
        r0 = rows[0]
        return QTensor(data.contiguous(), scale.contiguous(), scheme=r0.scheme,
                       group_size=r0.group_size, codebook=r0.codebook)
    return to_qtensor(torch.cat(rows, dim=0))


def qkv_weight(sd: dict, prefix: str) -> QTensor:
    return merge_qtensor([sd[f"{prefix}.q_proj.weight"],
                          sd[f"{prefix}.k_proj.weight"],
                          sd[f"{prefix}.v_proj.weight"]])


def gate_up_weight(sd: dict, prefix: str) -> QTensor:
    return merge_qtensor([sd[f"{prefix}.gate_proj.weight"], sd[f"{prefix}.up_proj.weight"]])
