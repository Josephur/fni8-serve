# SPDX-License-Identifier: MIT
"""ModelRunner — the minimal single-batch execution loop over a CausalLM.

Owns the KV cache, drives prefill then greedy/token decode. This is the v0 runner
the engine (PR3) wraps with a scheduler + paged block manager for continuous
batching; the model and cache contracts stay identical. Arch-agnostic: it only
calls `model.forward` / `model.compute_logits`, so every registered family runs
through it unchanged.
"""
from __future__ import annotations

import torch

from .base import ForwardContext
from .cache import KVCache
from .config import ModelConfig


class ModelRunner:
    def __init__(self, model, cfg: ModelConfig, *, max_batch: int, max_len: int, device="cuda"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.cache = KVCache(cfg.num_hidden_layers, max_batch, cfg.num_key_value_heads,
                             max_len, cfg.resolved_head_dim(), device=device)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, S] -> next-token logits [B, vocab]."""
        B, S = input_ids.shape
        pos = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
        self.cache.reset()
        ctx = ForwardContext(is_prefill=True, kv_cache=self.cache)
        hidden = self.model(input_ids, pos, ctx)
        self.cache.advance(S)
        return self.model.compute_logits(hidden[:, -1])

    @torch.inference_mode()
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, 1] -> next-token logits [B, vocab]."""
        B = token_ids.shape[0]
        pos = torch.full((B, 1), self.cache.length, device=self.device, dtype=torch.long)
        ctx = ForwardContext(is_prefill=False, kv_cache=self.cache)
        hidden = self.model(token_ids, pos, ctx)
        self.cache.advance(1)
        return self.model.compute_logits(hidden[:, -1])

    @torch.inference_mode()
    def generate_greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """prompt_ids: [B, S] -> generated token ids [B, max_new_tokens]."""
        logits = self.prefill(prompt_ids)
        tok = logits.argmax(-1, keepdim=True)
        out = [tok]
        for _ in range(max_new_tokens - 1):
            tok = self.decode(tok).argmax(-1, keepdim=True)
            out.append(tok)
        return torch.cat(out, dim=1)
