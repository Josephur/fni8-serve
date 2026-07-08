# SPDX-License-Identifier: MIT
"""LLMEngine — request queue + scheduler + runner, with an offline generate() API.

Arch-agnostic: it drives any registered CausalLM. Build it from a ModelConfig + a
weights dict (the `.fni8` loader output, or an HF state dict). Multi-GPU (PP +
MoE-EP, never TP) is a later layer; KV storage is `PagedKVCache` -- int8
quantize-on-write, block-table addressed, one batched decode launch per step.
"""
from __future__ import annotations

import itertools

from ..models.config import ModelConfig
from ..models.registry import build_model
from .cuda_graph import cuda_graph_enabled_by_env
from .kv_cache import PagedKVCache
from .model_runner import EngineRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence


class LLMEngine:
    def __init__(self, cfg: ModelConfig, weights: dict, *, device="cuda",
                 max_num_seqs: int = 16, max_len: int = 2048, max_batch_tokens: int = 8192,
                 eos_id: int | None = None, enable_cuda_graph: bool | None = None):
        self.cfg = cfg
        self.device = device
        self.eos_id = eos_id
        self.model = build_model(cfg, weights).to(device).eval()
        graph_wanted = (cuda_graph_enabled_by_env() if enable_cuda_graph is None
                       else enable_cuda_graph)
        # `GraphedDecode` pins one extra, never-freed cache slot for its padding
        # rows (see kv_cache.py `_scratch`). Give it a dedicated slot beyond
        # `max_num_seqs` so real scheduling capacity -- what `max_num_seqs`
        # promises the caller -- isn't silently reduced by one; the scheduler
        # itself still never admits more than `max_num_seqs` running sequences.
        num_slots = max_num_seqs + 1 if graph_wanted else max_num_seqs
        self.cache = PagedKVCache(cfg.num_hidden_layers, num_slots,
                                  cfg.num_key_value_heads, max_len, cfg.resolved_head_dim(),
                                  device=device)
        self.scheduler = Scheduler(self.cache, max_num_seqs=max_num_seqs,
                                   max_batch_tokens=max_batch_tokens, eos_id=eos_id)
        self.runner = EngineRunner(self.model, self.cache, device=device,
                                   enable_cuda_graph=enable_cuda_graph)
        self._ids = itertools.count()
        self._out: dict[int, Sequence] = {}

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
        toks = self.runner.prefill(batch) if is_prefill else self.runner.decode(batch)
        for seq, tok in zip(batch, toks):
            seq.output_ids.append(int(tok))
        self.scheduler.postprocess(batch, is_prefill)

    def generate(self, prompts: list[list[int]],
                 params: SamplingParams | None = None) -> list[list[int]]:
        """Offline batched generation: returns the output token ids per prompt."""
        ids = [self.add_request(p, params) for p in prompts]
        while self.scheduler.has_work():
            self.step()
        return [self._out[i].output_ids for i in ids]
