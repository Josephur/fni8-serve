# SPDX-License-Identifier: MIT
"""Token sampler: greedy (temperature 0) or temperature + top-p (nucleus).

Operates on the last-position logits of each sequence in a batch. Temperatures are
per-sequence so a batch can mix greedy and sampled sequences in one call.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Sampler(nn.Module):
    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,               # [num_seqs, vocab] fp16/fp32
        temperatures: torch.Tensor,         # [num_seqs] fp32 (0 -> greedy)
        top_p: torch.Tensor | None = None,  # [num_seqs] in (0,1], or None
    ) -> torch.Tensor:
        logits = logits.float()
        greedy = logits.argmax(dim=-1)
        temp = temperatures.clamp_min(1e-5)
        scaled = logits / temp.unsqueeze(-1)

        if top_p is not None:
            scaled = self._apply_top_p(scaled, top_p)

        probs = torch.softmax(scaled, dim=-1)
        # Gumbel-max trick: argmax(log p + gumbel noise) ~ categorical(p), vectorized.
        noise = torch.empty_like(probs).exponential_(1.0)
        sampled = (probs / noise).argmax(dim=-1)

        return torch.where(temperatures == 0, greedy, sampled)

    @staticmethod
    def _apply_top_p(scaled: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        # keep the smallest prefix whose cumulative prob >= top_p (always keep top-1)
        mask = cumsum - sorted_probs > top_p.unsqueeze(-1)
        sorted_logits = scaled.gather(-1, sorted_idx).masked_fill(mask, float("-inf"))
        return sorted_logits.scatter(-1, sorted_idx, sorted_logits)
