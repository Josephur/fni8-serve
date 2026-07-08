# SPDX-License-Identifier: MIT
"""EngineRunner — executes a scheduled batch (prefill or ragged decode) and samples.

Prefill runs one sequence at a time into its slot (varlen batched prefill is an
optimization behind the fni8 varlen path). Decode batches every running sequence:
the QKV/O/MLP GEMMs run as one [num_seqs, 1, hidden] matmul set, and the
memory-bound attention call reads the whole ragged batch in ONE paged-decode launch
(`PagedKVCache.decode_attn`) instead of looping per slot. Sampling is batched via
the Sampler.
"""
from __future__ import annotations

import torch

from ..layers.sampler import Sampler
from ..models.base import ForwardContext
from .sequence import Sequence


class EngineRunner:
    def __init__(self, model, cache, *, device="cuda"):
        self.model = model
        self.cache = cache
        self.device = device
        self.sampler = Sampler()

    def _sample(self, logits: torch.Tensor, batch: list[Sequence]) -> list[int]:
        temps = torch.tensor([s.params.temperature for s in batch],
                             device=self.device, dtype=torch.float32)
        top_p = torch.tensor([s.params.top_p for s in batch],
                             device=self.device, dtype=torch.float32)
        toks = self.sampler(logits, temps, top_p=top_p)
        return toks.tolist()

    @torch.inference_mode()
    def prefill(self, batch: list[Sequence]) -> list[int]:
        out = []
        for seq in batch:
            ids = torch.tensor([seq.prompt_ids], device=self.device)     # [1, S]
            pos = torch.arange(seq.num_prompt, device=self.device).unsqueeze(0)
            self.cache.ensure_capacity([seq.slot], [seq.num_prompt])
            ctx = ForwardContext(is_prefill=True, kv_cache=self.cache, slots=[seq.slot])
            hidden = self.model(ids, pos, ctx)
            seq.length = seq.num_prompt
            logits = self.model.compute_logits(hidden[:, -1])
            out.append(self._sample(logits, [seq])[0])
        return out

    @torch.inference_mode()
    def decode(self, batch: list[Sequence]) -> list[int]:
        ids = torch.tensor([[s.last_token] for s in batch], device=self.device)   # [B,1]
        pos = torch.tensor([[s.length] for s in batch], device=self.device)       # [B,1]
        slots = [s.slot for s in batch]
        lengths = [s.length for s in batch]
        self.cache.ensure_capacity(slots, [n + 1 for n in lengths])
        ctx = ForwardContext(is_prefill=False, kv_cache=self.cache,
                             slots=slots, slot_lengths=lengths)
        hidden = self.model(ids, pos, ctx)
        for s in batch:
            s.length += 1
        return self._sample(self.model.compute_logits(hidden[:, -1]), batch)
