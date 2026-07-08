# SPDX-License-Identifier: MIT
"""Sparse MoE FFN — Qwen3-MoE routing, with an optional shared expert.

Routing (Qwen3-MoE, verified against HF): softmax over ALL experts (fp32) -> top-k
-> renormalize the k weights (`norm_topk_prob`). Each expert is a GatedMLP on the
fni8 dp4a GEMM. A shared expert (Qwen3-Next / Qwen2-MoE) runs on every token in
parallel and is added in; plain Qwen3-MoE has none.

v0 loops experts in Python (correct, simple). A batched grouped-GEMM that runs all
active experts in one launch is the eventual fni8 kernel (moe grouped-GEMM), noted
in the coverage matrix; the win is at high expert counts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fni8 import QTensor

from ..layers.mlp import GatedMLP


class SparseMoE(nn.Module):
    def __init__(
        self,
        *,
        gate: torch.Tensor,                 # router weight [num_experts, hidden] fp16
        experts: list[tuple[QTensor, QTensor]],   # per-expert (gate_up, down)
        top_k: int,
        norm_topk_prob: bool = True,
        act: str = "silu",
        shared_expert: tuple[QTensor, QTensor] | None = None,
        shared_expert_gate: torch.Tensor | None = None,   # [1, hidden] fp16 or None
    ):
        super().__init__()
        self.gate = nn.Parameter(gate)
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.experts = nn.ModuleList([GatedMLP(gu, dn, act=act) for gu, dn in experts])
        self.shared = GatedMLP(*shared_expert, act=act) if shared_expert else None
        self.shared_gate = nn.Parameter(shared_expert_gate) if shared_expert_gate is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        xf = x.reshape(-1, H)                                    # [T, H]
        router_logits = F.linear(xf.float(), self.gate.float())  # [T, E]
        weights = F.softmax(router_logits, dim=-1)               # softmax over ALL experts, fp32
        topw, topi = torch.topk(weights, self.top_k, dim=-1)     # [T, k]
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw.to(x.dtype)

        out = torch.zeros_like(xf)
        # gather tokens per expert (only run experts that got routed to)
        for e in range(len(self.experts)):
            mask = (topi == e)                                   # [T, k]
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)           # which tokens, which slot
            contrib = self.experts[e](xf[tok_idx].unsqueeze(1)).squeeze(1)  # [n, H]
            out.index_add_(0, tok_idx, contrib * topw[tok_idx, slot].unsqueeze(-1))

        if self.shared is not None:
            shared_out = self.shared(x).reshape(-1, H)
            if self.shared_gate is not None:
                g = torch.sigmoid(F.linear(xf.float(), self.shared_gate.float())).to(x.dtype)
                shared_out = shared_out * g
            out = out + shared_out
        return out.reshape(B, S, H)
