# SPDX-License-Identifier: MIT
"""Model interface + decode-strategy abstraction — the seam the engine talks to.

The engine is architecture-agnostic: it only sees `CausalLM` (forward hidden ->
logits) and a `DecodeStrategy` (how tokens are produced — autoregressive next-token
vs diffusion iterative-denoising). A new model family implements `CausalLM.build`
and registers itself; a new *generation paradigm* implements `DecodeStrategy`.
Weights arrive as a `dict[str, QTensor | Tensor]` from the `.fni8` loader.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .config import ModelConfig

Weights = dict[str, object]   # name -> QTensor (quantized) or torch.Tensor (raw)


@runtime_checkable
class CausalLM(Protocol):
    """What every model exposes to the runner. Implementations are nn.Modules."""

    config: ModelConfig

    def forward(
        self,
        input_ids: torch.Tensor,       # [num_tokens] (flattened batch, token-major)
        positions: torch.Tensor,       # [num_tokens]
        ctx: "ForwardContext",
    ) -> torch.Tensor:                 # hidden states [num_tokens, hidden]
        ...

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [num_seqs, hidden] (last positions) -> logits [num_seqs, vocab]."""
        ...


class ForwardContext:
    """Per-step data the attention layers need: KV cache handles, block tables /
    slot mapping (paged), sequence lengths, and the prefill/decode phase flag. The
    concrete cache object is owned by the runner; the model reads through it."""

    def __init__(
        self,
        *,
        is_prefill: bool,
        kv_cache=None,
        lin_cache=None,
        cu_seqlens: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        attn_mask=None,
        slots: list[int] | None = None,        # engine: which cache slot each batch row uses
        slot_lengths: list[int] | None = None,  # engine: per-row KV length (ragged decode)
    ):
        self.is_prefill = is_prefill
        self.kv_cache = kv_cache
        self.lin_cache = lin_cache     # RecurrentStateCache for linear-attn layers
        self.cu_seqlens = cu_seqlens
        self.seq_lens = seq_lens
        self.slot_mapping = slot_mapping
        self.block_tables = block_tables
        self.attn_mask = attn_mask     # e.g. bidirectional mask for diffusion
        self.slots = slots
        self.slot_lengths = slot_lengths


class DecodeStrategy(Protocol):
    """How the engine turns a prompt into tokens. Two paradigms:
      - AutoregressiveStrategy: prefill then one-token-at-a-time (optionally with an
        MTP/spec-decode draft+verify inner loop).
      - DiffusionStrategy: allocate a masked span, iteratively denoise with
        bidirectional attention until unmasked.
    The engine drives `step`; the strategy owns the paradigm-specific control flow.
    """

    def is_autoregressive(self) -> bool: ...
