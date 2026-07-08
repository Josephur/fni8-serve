# SPDX-License-Identifier: MIT
"""Token embedding + LM head.

Embeddings stay fp16 (a lookup, not a matmul — no dp4a benefit, and the table is
numerically sensitive). Gemma scales embeddings by sqrt(hidden_size) after lookup
(`embed_scale`); Qwen3 does not. The LM head is either tied to the embedding
(small models) or its own weight, and may be int8-quantized (a real GEMM) via a
`QTensor`, or kept fp16.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fni8 import QTensor

from .linear import LinearW8A8


class VocabEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, embed_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(weight)          # [vocab, hidden] fp16
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = F.embedding(input_ids, self.weight)
        if self.embed_scale != 1.0:
            h = h * self.embed_scale
        return h


class LMHead(nn.Module):
    """Projects hidden states to vocab logits. `weight` is a QTensor (int8 GEMM) or
    an fp16 tensor `[vocab, hidden]` (tied embedding / plain matmul)."""

    def __init__(self, weight: QTensor | torch.Tensor, logit_softcap: float | None = None):
        super().__init__()
        self.logit_softcap = logit_softcap
        if isinstance(weight, QTensor):
            self.proj = LinearW8A8(weight)
            self._fp16_w = None
        else:
            self.proj = None
            self.weight = nn.Parameter(weight)
            self._fp16_w = True

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden) if self.proj is not None else F.linear(hidden, self.weight)
        if self.logit_softcap:                       # Gemma-style final-logit soft cap
            logits = self.logit_softcap * torch.tanh(logits.float() / self.logit_softcap)
        return logits
