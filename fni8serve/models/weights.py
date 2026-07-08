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


def to_qtensor(w: torch.Tensor) -> QTensor:
    """fp16/fp32 weight [out, in] (in % 4 == 0) -> per_row_i8 QTensor."""
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def merge_qtensor(rows: list[torch.Tensor]) -> QTensor:
    """Concat weights on the output (row) axis, then quantize as one — used to fuse
    q/k/v -> qkv and gate/up -> gate_up."""
    return to_qtensor(torch.cat(rows, dim=0))


def qkv_weight(sd: dict, prefix: str) -> QTensor:
    return merge_qtensor([sd[f"{prefix}.q_proj.weight"],
                          sd[f"{prefix}.k_proj.weight"],
                          sd[f"{prefix}.v_proj.weight"]])


def gate_up_weight(sd: dict, prefix: str) -> QTensor:
    return merge_qtensor([sd[f"{prefix}.gate_proj.weight"], sd[f"{prefix}.up_proj.weight"]])
