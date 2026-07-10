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
from ..models.cache import RecurrentStateCache
from .cuda_graph import GraphedDecode, cuda_graph_enabled_by_env
from .sequence import Sequence


class EngineRunner:
    def __init__(
        self, model, cache, *, device="cuda", enable_cuda_graph: bool | None = None, lin_cache=None
    ):
        self.model = model
        self.cache = cache
        self.device = device
        self.lin_cache = lin_cache or RecurrentStateCache()
        self.sampler = Sampler()
        # Decode is dispatch-bound (~3,200 cudaLaunchKernel/step on a 28-layer
        # 0.6B model for ~18ms of real GPU work -- issue #42): `GraphedDecode`
        # captures the whole decode step into a CUDA graph so replay re-issues
        # every one of those launches as ONE `cudaGraphLaunch`. Defaults on;
        # override with the `enable_cuda_graph` kwarg or `FNI8SERVE_CUDA_GRAPH=0`
        # (eager stays available for debugging either way -- `decode()` falls
        # back per-step whenever the graph can't serve a batch).
        if enable_cuda_graph is None:
            enable_cuda_graph = cuda_graph_enabled_by_env()
        self.graphed = GraphedDecode(model, cache, device=device) if enable_cuda_graph else None
        # Persistent host/device staging for the per-step sampling params (issue #183):
        # `temps`/`top_p` used to be rebuilt every step with `torch.tensor(list,
        # device=cuda)` -- a blocking pageable host->device copy per step. We keep a
        # pinned host buffer filled in place + a non_blocking copy into a persistent
        # device tensor instead. Grown lazily to the batch size actually seen.
        self._pin = device != "cpu" and torch.cuda.is_available()
        self._temps_host: torch.Tensor | None = None
        self._top_p_host: torch.Tensor | None = None
        self._temps_dev: torch.Tensor | None = None
        self._top_p_dev: torch.Tensor | None = None

    def _ensure_sample_buffers(self, n: int):
        if self._temps_host is not None and self._temps_host.numel() >= n:
            return
        self._temps_host = torch.empty(n, dtype=torch.float32, pin_memory=self._pin)
        self._top_p_host = torch.empty(n, dtype=torch.float32, pin_memory=self._pin)
        self._temps_dev = torch.empty(n, dtype=torch.float32, device=self.device)
        self._top_p_dev = torch.empty(n, dtype=torch.float32, device=self.device)

    def _sample(self, logits: torch.Tensor, batch: list[Sequence]) -> list[int]:
        n = len(batch)
        self._ensure_sample_buffers(n)
        temp_vals = [s.params.temperature for s in batch]
        top_p_vals = [s.params.top_p for s in batch]
        # Fill the pinned host slice in place, then async-copy the used slice to the
        # persistent device tensor -- no per-step device allocation, no blocking H2D.
        self._temps_host[:n].copy_(torch.tensor(temp_vals, dtype=torch.float32))
        self._top_p_host[:n].copy_(torch.tensor(top_p_vals, dtype=torch.float32))
        temps = self._temps_dev[:n]
        top_p = self._top_p_dev[:n]
        temps.copy_(self._temps_host[:n], non_blocking=self._pin)
        top_p.copy_(self._top_p_host[:n], non_blocking=self._pin)
        # Decide greedy / top-p short-circuits from the python params (no device sync)
        # and hand the sampler the answer so its fast path stays sync-free.
        all_greedy = all(t == 0.0 for t in temp_vals)
        any_top_p = any(p < 1.0 for p in top_p_vals)
        procs = [s.params.logit_processors for s in batch]
        has_procs = any(procs)
        toks = self.sampler(
            logits,
            temps,
            top_p=top_p,
            logit_processors=procs if has_procs else None,
            input_ids=[s.all_token_ids for s in batch] if has_procs else None,
            all_greedy=all_greedy,
            any_top_p=any_top_p,
        )
        # The one necessary device->host readback: the caller (llm_engine) appends
        # these as python ints. Sampling itself stays fully on-device above.
        return toks.tolist()

    @torch.inference_mode()
    def encode(self, batch: list[Sequence]) -> torch.Tensor:
        hiddens = []
        for seq in batch:
            self.lin_cache.reset()
            ids = torch.tensor([seq.prompt_ids], device=self.device)
            pos = torch.arange(seq.num_prompt, device=self.device).unsqueeze(0)
            self.cache.ensure_capacity([seq.slot], [seq.num_prompt])
            ctx = ForwardContext(
                is_prefill=True,
                kv_cache=self.cache,
                lin_cache=self.lin_cache,
                slots=[seq.slot],
                prefill_start=seq.prefix_matched_len,
            )
            hidden = self.model(ids, pos, ctx)
            pooled = hidden.mean(dim=1)
            hiddens.append(pooled)
        return torch.cat(hiddens, dim=0)

    @torch.inference_mode()
    def prefill(self, batch: list[Sequence]) -> list[int]:
        from .kv_cache import PagedKVCache

        if len(batch) > 1 and isinstance(self.cache, PagedKVCache):
            return self._prefill_varlen(batch)
        out = []
        for seq in batch:
            self.lin_cache.reset()
            ids = torch.tensor([seq.prompt_ids], device=self.device)  # [1, S]
            pos = torch.arange(seq.num_prompt, device=self.device).unsqueeze(0)
            self.cache.ensure_capacity([seq.slot], [seq.num_prompt])
            ctx = ForwardContext(
                is_prefill=True,
                kv_cache=self.cache,
                lin_cache=self.lin_cache,
                slots=[seq.slot],
                prefill_start=seq.prefix_matched_len,
                pixel_values=seq.pixel_values,
            )
            hidden = self.model(ids, pos, ctx)
            seq.length = seq.num_prompt
            logits = self.model.compute_logits(hidden[:, -1])
            out.append(self._sample(logits, [seq])[0])
        return out

    @torch.inference_mode()
    def _prefill_varlen(self, batch: list[Sequence]) -> list[int]:
        """Pack multiple sequences into one varlen forward pass with cumulative
        sequence lengths. Attention cost scales with total tokens, not
        max_len × batch (fni8.attn_int8_varlen kernel)."""
        self.lin_cache.reset()

        self.cache.ensure_capacity([s.slot for s in batch], [s.num_prompt for s in batch])

        all_ids: list[int] = []
        all_positions: list[int] = []
        cu_seqlens: list[int] = [0]
        slot_mapping_flat: list[int] = []

        for seq in batch:
            n = seq.num_prompt
            all_ids.extend(seq.prompt_ids)
            all_positions.extend(range(n))
            cu_seqlens.append(cu_seqlens[-1] + n)
            for t in range(n):
                slot_mapping_flat.append(self.cache._slot_mapping([seq.slot], [t]).item())

        total_tokens = cu_seqlens[-1]
        ids = torch.tensor([all_ids], device=self.device)  # [1, total_tok]
        pos = torch.tensor([all_positions], device=self.device)
        cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device)
        sm = torch.tensor(slot_mapping_flat, dtype=torch.int32, device=self.device)

        ctx = ForwardContext(
            is_prefill=True,
            kv_cache=self.cache,
            lin_cache=self.lin_cache,
            slots=[s.slot for s in batch],
            cu_seqlens=cu,
            slot_mapping=sm,
        )

        hidden = self.model(ids, pos, ctx)  # [1, total_tok, hidden]

        for seq in batch:
            seq.length = seq.num_prompt

        last_indices = torch.tensor(
            [cu_seqlens[i] - 1 for i in range(1, len(cu_seqlens))],
            device=self.device,
            dtype=torch.long,
        )
        logits = self.model.compute_logits(hidden[:, last_indices]).squeeze(0)  # [B, vocab]

        return self._sample(logits, batch)

    @torch.inference_mode()
    def decode(self, batch: list[Sequence]) -> list[int]:
        logits = self.graphed.try_decode(batch) if self.graphed is not None else None
        if logits is None:
            return self._decode_eager(batch)
        for s in batch:
            s.length += 1
        return self._sample(logits, batch)

    @torch.inference_mode()
    def _decode_eager(self, batch: list[Sequence]) -> list[int]:
        ids = torch.tensor([[s.last_token] for s in batch], device=self.device)  # [B,1]
        pos = torch.tensor([[s.length] for s in batch], device=self.device)  # [B,1]
        slots = [s.slot for s in batch]
        lengths = [s.length for s in batch]
        self.cache.ensure_capacity(slots, [n + 1 for n in lengths])
        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=self.cache,
            lin_cache=self.lin_cache,
            slots=slots,
            slot_lengths=lengths,
        )
        hidden = self.model(ids, pos, ctx)
        for s in batch:
            s.length += 1
        return self._sample(self.model.compute_logits(hidden[:, -1]), batch)
