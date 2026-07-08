# SPDX-License-Identifier: MIT
"""Rotary position embedding (RoPE), Qwen3/Llama "rotate_half" convention.

Qwen3 applies RoPE to Q and K per head AFTER the per-head QK-norm. cos/sin are
precomputed for all positions up to `max_position` and gathered by the per-token
position ids (so prefill and paged decode share one table). Kept in fp32 for the
gather then applied in the tensor's dtype.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float = 1e6):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                      # [max_pos, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)               # [max_pos, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [...] int; q, k: [..., n_heads, head_dim]. Broadcasts cos/sin
        over the head axis."""
        cos = self.cos[positions].unsqueeze(-2).to(q.dtype)   # [..., 1, head_dim]
        sin = self.sin[positions].unsqueeze(-2).to(q.dtype)
        q_r = q * cos + _rotate_half(q) * sin
        k_r = k * cos + _rotate_half(k) * sin
        return q_r, k_r
