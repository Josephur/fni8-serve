# SPDX-License-Identifier: MIT
"""Token sampler: greedy (temperature 0) or temperature + top-p (nucleus).

Operates on the last-position logits of each sequence in a batch. Temperatures are
per-sequence so a batch can mix greedy and sampled sequences in one call. Also the
per-step logit-processor hook: a per-sequence list of (input_ids, logits) -> logits
callables applied before sampling -- what structured outputs / grammars plug into.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

LogitsProcessor = Callable[[list[int], torch.Tensor], torch.Tensor]


class Sampler(nn.Module):
    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,  # [num_seqs, vocab] fp16/fp32
        temperatures: torch.Tensor,  # [num_seqs] fp32 (0 -> greedy)
        top_p: torch.Tensor | None = None,  # [num_seqs] in (0,1], or None
        logit_processors: list[list[LogitsProcessor]] | None = None,  # per-seq processors
        input_ids: list[list[int]] | None = None,  # per-seq token history, for the hook
        *,
        all_greedy: bool | None = None,  # pure-python hint: every row is temp==0
        any_top_p: bool | None = None,  # pure-python hint: some row has top_p < 1.0
    ) -> torch.Tensor:
        logits = logits.float()
        if logit_processors is not None:
            logits = self._apply_logit_processors(logits, input_ids, logit_processors)
        greedy = logits.argmax(dim=-1)

        # Fast path (issue #183): an all-greedy batch (every temp==0) is just the
        # argmax -- skip the temp-scale + softmax + top-p sort + Gumbel-max noise,
        # which otherwise run over the full 151,936-token vocab EVERY decode step
        # (~0.9 ms). Bit-identical to the general path below: `torch.where` would
        # select `greedy` for every row anyway. The caller (EngineRunner._sample)
        # passes `all_greedy` computed from python params so this decision costs no
        # device->host sync; falling back to a tensor reduction only when unhinted.
        if all_greedy is None:
            all_greedy = bool((temperatures == 0).all())
        if all_greedy:
            return greedy

        temp = temperatures.clamp_min(1e-5)
        scaled = logits / temp.unsqueeze(-1)

        # top_p == 1.0 keeps the whole distribution, so the nucleus filter (and its
        # full-vocab torch.sort) is a no-op -- skip it unless some row is < 1.0.
        if any_top_p is None:
            any_top_p = top_p is not None and bool((top_p < 1.0).any())
        if top_p is not None and any_top_p:
            scaled = self._apply_top_p(scaled, top_p)

        probs = torch.softmax(scaled, dim=-1)
        # Gumbel-max trick: argmax(log p + gumbel noise) ~ categorical(p), vectorized.
        noise = torch.empty_like(probs).exponential_(1.0)
        sampled = (probs / noise).argmax(dim=-1)

        return torch.where(temperatures == 0, greedy, sampled)

    @staticmethod
    def _apply_logit_processors(
        logits: torch.Tensor,
        input_ids: list[list[int]],
        logit_processors: list[list[LogitsProcessor]],
    ) -> torch.Tensor:
        rows = list(logits.unbind(0))
        for i, procs in enumerate(logit_processors):
            for proc in procs or ():
                rows[i] = proc(input_ids[i], rows[i])
        return torch.stack(rows)

    @staticmethod
    def _apply_top_p(scaled: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        # keep the smallest prefix whose cumulative prob >= top_p (always keep top-1)
        mask = cumsum - sorted_probs > top_p.unsqueeze(-1)
        sorted_logits = scaled.gather(-1, sorted_idx).masked_fill(mask, float("-inf"))
        return sorted_logits.scatter(-1, sorted_idx, sorted_logits)
