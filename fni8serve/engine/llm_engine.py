# SPDX-License-Identifier: MIT
"""LLMEngine — request queue + scheduler + runner, with an offline generate() API.

Arch-agnostic: it drives any registered CausalLM. Build it from a ModelConfig + a
weights dict (the `.fni8` loader output, or an HF state dict). Multi-GPU (PP +
MoE-EP, never TP) is a later layer; KV storage is `PagedKVCache` -- int8
quantize-on-write, block-table addressed, one batched decode launch per step.
"""

from __future__ import annotations

import itertools
import time

from ..models.base import ForwardContext
from ..models.cache import MLALatentCache, RecurrentStateCache
from ..models.config import ModelConfig
from ..models.registry import build_model
from .cuda_graph import cuda_graph_enabled_by_env
from .decode_strategy import DiffusionDecodeStrategy
from .kv_cache import PagedKVCache
from .model_runner import EngineRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, Status


class LLMEngine:
    def __init__(
        self,
        cfg: ModelConfig,
        weights: dict,
        *,
        device="cuda",
        max_num_seqs: int = 16,
        max_len: int = 2048,
        max_batch_tokens: int = 8192,
        eos_id: int | None = None,
        enable_cuda_graph: bool | None = None,
        num_diffusion_steps: int = 8,
    ):
        self.cfg = cfg
        self.device = device
        self.eos_id = eos_id
        self.model = build_model(cfg, weights).to(device).eval()
        self._diffusion_strategy = (
            DiffusionDecodeStrategy(num_steps=num_diffusion_steps)
            if cfg.decode_strategy == "diffusion"
            else None
        )
        graph_wanted = (
            cuda_graph_enabled_by_env() if enable_cuda_graph is None else enable_cuda_graph
        )
        # `GraphedDecode` pins one extra, never-freed cache slot for its padding
        # rows (see kv_cache.py `_scratch`). Give it a dedicated slot beyond
        # `max_num_seqs` so real scheduling capacity -- what `max_num_seqs`
        # promises the caller -- isn't silently reduced by one; the scheduler
        # itself still never admits more than `max_num_seqs` running sequences.
        num_slots = max_num_seqs + 1 if graph_wanted else max_num_seqs
        if cfg.latent_attention:
            self.cache = MLALatentCache(
                cfg.num_hidden_layers,
                num_slots,
                cfg.mla_cache_dim(),
                max_len,
                device=device,
            )
        else:
            self.cache = PagedKVCache(
                cfg.num_hidden_layers,
                num_slots,
                cfg.num_key_value_heads,
                max_len,
                cfg.resolved_head_dim(),
                device=device,
            )
        self.scheduler = Scheduler(
            self.cache, max_num_seqs=max_num_seqs, max_batch_tokens=max_batch_tokens, eos_id=eos_id
        )
        self.lin_cache = RecurrentStateCache()
        self.runner = EngineRunner(
            self.model,
            self.cache,
            device=device,
            enable_cuda_graph=enable_cuda_graph,
            lin_cache=self.lin_cache,
        )
        self._ids = itertools.count()
        self._out: dict[int, Sequence] = {}
        # Optional telemetry sink (fni8serve.metrics.StatsCollector). The API layer
        # sets this; offline `generate()` leaves it None. `step()` records only
        # host-side ints + a perf_counter span into it -- never a GPU sync.
        self.stats = None

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> int:
        seq = Sequence(next(self._ids), list(prompt_ids), params or SamplingParams())
        self.scheduler.add(seq)
        self._out[seq.seq_id] = seq
        return seq.seq_id

    def sequence(self, seq_id: int) -> Sequence:
        return self._out[seq_id]

    def forget(self, seq_id: int) -> None:
        """Drop bookkeeping for a finished request. Needed by long-lived callers (the
        API server) that generate() never returns to -- without this, `_out` would
        retain every request's Sequence for the life of the process."""
        self._out.pop(seq_id, None)

    def step(self):
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return
        # Token count for this step from host-side ints we already hold: prefill
        # processes every prompt token, decode emits exactly one token per row.
        # (No `.item()`/`.cpu()` -- the existing `int(tok)` below already
        # materialises decode tokens on host, so the perf_counter span honestly
        # reflects real GPU time without any *added* sync.)
        step_t0 = time.perf_counter() if self.stats is not None else 0.0
        num_tokens = sum(seq.num_prompt for seq in batch) if is_prefill else len(batch)
        if self._diffusion_strategy is not None and is_prefill:
            for seq in batch:
                total_len = seq.num_prompt + seq.params.max_tokens
                self.cache.ensure_capacity([seq.slot], [total_len])
                seq.length = total_len
                ctx = ForwardContext(
                    is_prefill=True, kv_cache=self.cache, lin_cache=self.lin_cache, slots=[seq.slot]
                )
                out_tokens = self._diffusion_strategy.generate(
                    self.model, self.cache, self.device, seq, ctx
                )
                seq.output_ids.extend(out_tokens)
                self.cache.store_prefix(seq.prompt_ids, seq.slot)
        else:
            toks = self.runner.prefill(batch) if is_prefill else self.runner.decode(batch)
            for seq, tok in zip(batch, toks):
                seq.output_ids.append(int(tok))
            if is_prefill:
                for seq in batch:
                    self.cache.store_prefix(seq.prompt_ids, seq.slot)
        self.scheduler.postprocess(batch, is_prefill)
        if self.stats is not None:
            self.stats.record_step(
                is_prefill=is_prefill,
                num_tokens=num_tokens,
                running=len(self.scheduler.running),
                waiting=len(self.scheduler.waiting),
                dt=time.perf_counter() - step_t0,
            )

    def encode(self, prompt_ids: list[int]) -> list[float]:
        seq_id = self.add_request(prompt_ids)
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            self.forget(seq_id)
            return []
        pooled = self.runner.encode(batch)
        for seq in batch:
            self.cache.store_prefix(seq.prompt_ids, seq.slot)
            seq.status = Status.FINISHED
        self.scheduler.postprocess(batch, is_prefill)
        self.forget(seq_id)
        return pooled[0].cpu().tolist()

    def generate(
        self, prompts: list[list[int]], params: SamplingParams | None = None
    ) -> list[list[int]]:
        """Offline batched generation: returns the output token ids per prompt."""
        ids = [self.add_request(p, params) for p in prompts]
        while self.scheduler.has_work():
            self.step()
        return [self._out[i].output_ids for i in ids]
