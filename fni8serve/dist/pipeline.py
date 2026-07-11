# SPDX-License-Identifier: MIT
"""Pipeline-parallel LLM inference across 2 GPUs (issue #84).

Splits a model's layer stack into 2 stages: stage 0 owns the embedding + first
half of layers; stage 1 owns the second half + final norm + lm_head. The hidden
state (and the pre-norm residual stream) crosses the PP boundary through the
existing transport ``send``/``recv`` seam (issue #81/D9). The wire codec is
selected per-boundary by :func:`select_wire_scheme`: int4 when its fidelity
clears the accuracy-gate bar, int8 otherwise (issue #186).

Micro-batched decode overlaps the transfer with compute: while stage 0 forwards
micro-batch k+1, stage 1 receives and forwards micro-batch k, keeping both GPUs
busy.  2-stage PP only (MoE-EP is the next rung — see issue #60 / D10).
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn as nn

from ..engine.kv_cache import PagedKVCache
from ..engine.sequence import SamplingParams, Sequence, Status
from ..models.base import CausalLM, ForwardContext
from ..models.cache import MLALatentCache, RecurrentStateCache
from ..models.config import ModelConfig
from ..models.registry import build_model
from . import recv, send
from .codec_quality import select_wire_scheme

# Monkey-patch fni8's hadamard_matrix so its LRU cache key includes the concrete
# GPU device index, not just the device type.  Without this, the matrix created
# on cuda:0 gets reused on cuda:1 and causes a device-mismatch crash.
try:
    import fni8.quant.rotation as _fni8_rot_rel

    _orig_hadamard = _fni8_rot_rel.hadamard_matrix

    def _patched_hadamard(d: int, device: str = "cpu", dtype=torch.float32):
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        return _orig_hadamard(d, device=device, dtype=dtype)

    _fni8_rot_rel.hadamard_matrix = _patched_hadamard
    _orig_rotate = _fni8_rot_rel.rotate_last

    def _patched_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        dev = x.device
        m = _patched_hadamard(d, device=dev.type, dtype=torch.float32)
        return (x.float() @ m).to(x.dtype)

    _fni8_rot_rel.rotate_last = _patched_rotate
except Exception:
    pass


def _clear_hadamard_cache():
    "Clear fni8's LRU hadamard-matrix cache (legacy; the monkey-patch above should suffice)."
    try:
        from fni8.quant.rotation import hadamard_matrix

        hadamard_matrix.cache_clear()
    except Exception:
        pass


class LayerRemappedCache:
    """Wraps a KV cache so a stage's layers (original indices i..j) map to the
    stage-local cache layout (0..j-i). Each access subtracts *offset*."""

    def __init__(self, cache: PagedKVCache, offset: int):
        self._cache = cache
        self._offset = offset

    @property
    def block_size(self):
        return self._cache.block_size

    @property
    def max_blocks_per_seq(self):
        return self._cache.max_blocks_per_seq

    @property
    def num_blocks(self):
        return self._cache.num_blocks

    @property
    def device(self):
        return self._cache.device

    @property
    def num_slots(self):
        return self._cache.num_slots

    @property
    def max_len(self):
        return self._cache.max_len

    def alloc(self) -> int:
        return self._cache.alloc()

    def free(self, slot: int):
        return self._cache.free(slot)

    def has_free_slot(self) -> bool:
        return self._cache.has_free_slot()

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        return self._cache.ensure_capacity(slots, lengths)

    def share_blocks(self, slot: int, blocks: list[int]):
        return self._cache.share_blocks(slot, blocks)

    def store_prefix(self, token_ids: list[int], slot: int):
        return self._cache.store_prefix(token_ids, slot)

    def lookup_prefix(self, token_ids: list[int]):
        return self._cache.lookup_prefix(token_ids)

    def _slot_mapping(self, slots: list[int], positions: list[int]):
        return self._cache._slot_mapping(slots, positions)

    def slot_mapping_for(self, slots: list[int], positions: list[int]):
        return self._cache.slot_mapping_for(slots, positions)

    def block_table(self, slots: list[int]):
        return self._cache.block_table(slots)

    def write_prefill(self, layer: int, k, v, *, slot: int, start: int = 0):
        return self._cache.write_prefill(layer - self._offset, k, v, slot=slot, start=start)

    def write_prefill_varlen(self, layer: int, slot_mapping, k, v):
        return self._cache.write_prefill_varlen(layer - self._offset, slot_mapping, k, v)

    def write_decode(self, layer: int, slots, positions, k_new, v_new):
        return self._cache.write_decode(layer - self._offset, slots, positions, k_new, v_new)

    def write_decode_static(self, layer: int, slot_mapping, k_new, v_new):
        return self._cache.write_decode_static(layer - self._offset, slot_mapping, k_new, v_new)

    def decode_attn(self, layer: int, q, slots, lengths, *, scale: float):
        return self._cache.decode_attn(layer - self._offset, q, slots, lengths, scale=scale)

    def decode_attn_static(
        self, layer: int, q, block_table, context_lens, max_context_len: int, *, scale: float
    ):
        return self._cache.decode_attn_static(
            layer - self._offset, q, block_table, context_lens, max_context_len, scale=scale
        )


def _build_stage_cache(
    num_layers: int,
    num_slots: int,
    num_kv_heads: int,
    max_len: int,
    head_dim: int,
    *,
    device,
    block_size: int = 16,
    num_blocks: int | None = None,
):
    return PagedKVCache(
        num_layers,
        num_slots,
        num_kv_heads,
        max_len,
        head_dim,
        device=device,
        block_size=block_size,
        num_blocks=num_blocks,
    )


def _remap_ctx(ctx: ForwardContext, cache, lin_cache=None) -> ForwardContext:
    """Return a new ForwardContext with *cache* swapped in (layer-remapped)."""
    return ForwardContext(
        is_prefill=ctx.is_prefill,
        kv_cache=cache,
        lin_cache=lin_cache or ctx.lin_cache,
        cu_seqlens=ctx.cu_seqlens,
        seq_lens=ctx.seq_lens,
        slot_mapping=ctx.slot_mapping,
        block_tables=ctx.block_tables,
        context_lens=ctx.context_lens,
        max_context_len=ctx.max_context_len,
        attn_mask=ctx.attn_mask,
        slots=ctx.slots,
        slot_lengths=ctx.slot_lengths,
        prefill_start=ctx.prefill_start,
        pixel_values=ctx.pixel_values,
    )


def _pack_boundary(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
    """Pack (hidden, residual) into one tensor for transport. residual may be None."""
    if residual is None:
        return hidden
    return torch.cat([hidden, residual], dim=-1)


def _unpack_boundary(packed: torch.Tensor, hidden_size: int):
    """Unpack -> (hidden, residual)."""
    if packed.shape[-1] == hidden_size:
        return packed, None
    return packed[..., :hidden_size], packed[..., hidden_size:]


class PipelineStage:
    """One contiguous slice of model layers on a single GPU.

    Attributes:
        stage_id: 0-indexed stage number.
        embed: VocabEmbedding (stage 0 only).
        layers: list of DecoderLayer modules.
        norm: final RMSNorm (last stage only).
        lm_head: LMHead (last stage only).
        kv_cache: this stage's PagedKVCache or MLALatentCache.
        lin_cache: this stage's RecurrentStateCache.
        layer_offset: original layer index of the first layer in this stage.
    """

    def __init__(
        self,
        stage_id: int,
        *,
        embed=None,
        layers: list | None = None,
        norm=None,
        lm_head=None,
        kv_cache: PagedKVCache,
        lin_cache=None,
        layer_offset: int = 0,
    ):
        self.stage_id = stage_id
        self.embed = embed
        self.layers = nn.ModuleList(layers or [])
        self.norm = norm
        self.lm_head = lm_head
        self.kv_cache: PagedKVCache = kv_cache
        self.lin_cache = lin_cache or RecurrentStateCache()
        self.layer_offset = layer_offset
        self._remapped_cache: LayerRemappedCache | None = None

    @property
    def is_first(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last(self) -> bool:
        return True  # 2-stage: stage 0 is not last, stage 1 is. Override for >2 stages.

    @property
    def _device(self):
        p = (
            next(self.layers.parameters())
            if self.layers
            else (next(self.embed.parameters()) if self.embed is not None else torch.device("cpu"))
        )
        return p.device

    def _remap(self, cache: PagedKVCache):
        if self._remapped_cache is not None:
            return self._remapped_cache
        self._remapped_cache = LayerRemappedCache(cache, self.layer_offset)
        return self._remapped_cache

    def compute_logits(self, hidden):
        if self.lm_head is None:
            raise RuntimeError("compute_logits only available on the last stage")
        return self.lm_head(hidden)


def make_pipeline(
    cfg: ModelConfig,
    weights: dict,
    *,
    devices: tuple[int, int] = (0, 1),
    max_num_seqs: int = 16,
    max_len: int = 2048,
    block_size: int = 16,
) -> tuple[PipelineStage, PipelineStage]:
    """Build a 2-stage pipeline from *cfg* and *weights*.

    The weights dict must contain tensors on CPU (or the same device for each stage)
    so `build_model` can place them on the correct GPU via post-build .to().

    Returns (stage_0, stage_1).
    """
    n_layers = cfg.num_hidden_layers
    mid = max(1, n_layers // 2)

    # Build full model on each device (duplicating weights — memory tradeoff for
    # correct QTensor device placement). In production a per-device filtered build
    # avoids the duplication.
    models_raw: list[nn.Module] = []
    for dev in devices:
        wt = {
            k: v.to(f"cuda:{dev}") if isinstance(v, torch.Tensor) else v for k, v in weights.items()
        }
        m: Any = build_model(cfg, wt)
        m.to(f"cuda:{dev}")
        m.eval()
        models_raw.append(m)

    m0: Any = models_raw[0]
    m0_model: Any = m0.model

    stage0 = PipelineStage(
        0,
        embed=m0_model.embed_tokens,
        layers=[m0_model.layers[i] for i in range(mid)],
        layer_offset=0,
        kv_cache=_build_stage_cache(
            mid,
            max_num_seqs,
            cfg.num_key_value_heads,
            max_len,
            cfg.resolved_head_dim(),
            device=devices[0],
            block_size=block_size,
        ),
    )

    m1: Any = models_raw[1]
    m1_model: Any = m1.model

    stage1 = PipelineStage(
        1,
        layers=[m1_model.layers[i] for i in range(mid, n_layers)],
        norm=m1_model.norm,
        lm_head=m1.lm_head,
        layer_offset=mid,
        kv_cache=_build_stage_cache(
            n_layers - mid,
            max_num_seqs,
            cfg.num_key_value_heads,
            max_len,
            cfg.resolved_head_dim(),
            device=devices[1],
            block_size=block_size,
        ),
    )

    return stage0, stage1


class PipelineEngine:
    """Pipeline-parallel inference for 2 GPUs.

    Wraps two PipelineStages and a scheduler. Prefill runs stage 0 first, then
    transfers hidden/residual to stage 1. Decode micro-batches the scheduled
    batch so the PP-boundary transfer overlaps with compute on both GPUs.
    """

    def __init__(
        self,
        stage0: PipelineStage,
        stage1: PipelineStage,
        cfg: ModelConfig,
        *,
        max_num_seqs: int = 16,
        max_len: int = 2048,
        max_batch_tokens: int = 8192,
        eos_id: int | None = None,
        wire_scheme: str | None = None,
    ):
        self.stage0 = stage0
        self.stage1 = stage1
        self.cfg = cfg
        self.eos_id = eos_id
        self.hidden_size = cfg.hidden_size
        self._ids = itertools.count()
        self._out: dict[int, Sequence] = {}
        self._wire_scheme = wire_scheme

        from ..engine.scheduler import Scheduler

        self.scheduler = Scheduler(
            stage0.kv_cache,
            max_num_seqs=max_num_seqs,
            max_batch_tokens=max_batch_tokens,
            eos_id=eos_id,
        )

    @property
    def devices(self):
        return (
            self.stage0._device.index or 0,
            self.stage1._device.index or 1,
        )

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> int:
        seq = Sequence(next(self._ids), list(prompt_ids), params or SamplingParams())
        self.scheduler.add(seq)
        self._out[seq.seq_id] = seq
        return seq.seq_id

    def sequence(self, seq_id: int) -> Sequence:
        return self._out[seq_id]

    def forget(self, seq_id: int) -> None:
        self._out.pop(seq_id, None)

    def _sync_caches(self, slot_op: str, slots: Any = None):
        """Mirror slot-level cache ops across both stages."""
        c0, c1 = self.stage0.kv_cache, self.stage1.kv_cache
        if slot_op == "alloc":
            s0 = c0.alloc()
            c1.alloc()
            return s0
        elif slot_op == "free":
            for s in slots:
                c0.free(s)
                c1.free(s)
        elif slot_op == "ensure_capacity":
            c0.ensure_capacity(slots[0], slots[1])
            c1.ensure_capacity(slots[0], slots[1])
        return None

    def step(self):
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return

        if is_prefill:
            self._prefill(batch)
        else:
            self._decode_microbatched(batch)

        self.scheduler.postprocess(batch, is_prefill)

    def _prefill(self, batch: list[Sequence]):
        """Prefill: stage 0 forward -> send -> stage 1 recv -> forward -> sample."""
        from ..layers.sampler import Sampler

        sampler = Sampler()
        dev0 = self.devices[0]
        dev1 = self.devices[1]
        device0 = f"cuda:{dev0}"
        device1 = f"cuda:{dev1}"

        # Build device-agnostic prefill metadata first
        all_ids: list[int] = []
        all_positions: list[int] = []
        cu_seqlens: list[int] = [0]
        slots_0: list[int] = []
        slot_mapping_flat_0: list[int] = []
        slot_mapping_flat_1: list[int] = []

        for seq in batch:
            self.stage0.kv_cache.ensure_capacity([seq.slot], [seq.num_prompt])
            self.stage1.kv_cache.ensure_capacity([seq.slot], [seq.num_prompt])
            n = seq.num_prompt
            all_ids.extend(seq.prompt_ids)
            all_positions.extend(range(n))
            cu_seqlens.append(cu_seqlens[-1] + n)
            slots_0.append(seq.slot)
            for t in range(n):
                slot_mapping_flat_0.append(
                    self.stage0.kv_cache._slot_mapping([seq.slot], [t]).item()
                )
                slot_mapping_flat_1.append(
                    self.stage1.kv_cache._slot_mapping([seq.slot], [t]).item()
                )

        # --- Stage 0 forward ---
        with torch.cuda.device(dev0):
            _clear_hadamard_cache()
            self.stage0.lin_cache.reset()
            ids = torch.tensor([all_ids], device=device0)
            pos0 = torch.tensor([all_positions], device=device0)
            cu0 = torch.tensor(cu_seqlens, dtype=torch.int32, device=device0)
            sm0 = torch.tensor(slot_mapping_flat_0, dtype=torch.int32, device=device0)

            ctx0 = ForwardContext(
                is_prefill=True,
                kv_cache=self.stage0.kv_cache,
                lin_cache=self.stage0.lin_cache,
                slots=slots_0,
                cu_seqlens=cu0,
                slot_mapping=sm0,
            )

            embed = self.stage0.embed
            assert embed is not None, "stage 0 must have an embedding"
            h_s0 = embed(ids)

            residual = None
            stage0_ctx = _remap_ctx(
                ctx0, self.stage0._remap(self.stage0.kv_cache), self.stage0.lin_cache
            )
            for layer in self.stage0.layers:
                h_s0, residual = layer(h_s0, pos0, stage0_ctx, residual)

            boundary_packed = _pack_boundary(h_s0.contiguous(), residual)
            scheme = self._wire_scheme if self._wire_scheme is not None else select_wire_scheme(boundary_packed)
            handle = send(boundary_packed, dst=dev1, scheme=scheme)

        # --- Stage 1 forward ---
        with torch.cuda.device(dev1):
            # fni8's hadamard_matrix caches per (dim, device_type), but device_type
            # = "cuda" omits the device index.  Clear the cache so the matrix is
            # re-created on the correct GPU.
            _clear_hadamard_cache()
            boundary_rcv: torch.Tensor = recv(handle)
            h_in, residual = _unpack_boundary(boundary_rcv, self.hidden_size)

            pos1 = torch.tensor([all_positions], device=device1)
            cu1 = torch.tensor(cu_seqlens, dtype=torch.int32, device=device1)
            sm1 = torch.tensor(slot_mapping_flat_1, dtype=torch.int32, device=device1)

            ctx1 = ForwardContext(
                is_prefill=True,
                kv_cache=self.stage1.kv_cache,
                lin_cache=self.stage1.lin_cache,
                slots=slots_0,
                cu_seqlens=cu1,
                slot_mapping=sm1,
            )

            stage1_ctx = _remap_ctx(
                ctx1, self.stage1._remap(self.stage1.kv_cache), self.stage1.lin_cache
            )
            for layer in self.stage1.layers:
                h_in, residual = layer(h_in, pos1, stage1_ctx, residual)

            norm1 = self.stage1.norm
            assert norm1 is not None, "last stage must have a norm"
            h_in, _ = norm1(h_in, residual)

            last_indices = torch.tensor(
                [cu_seqlens[i] - 1 for i in range(1, len(cu_seqlens))],
                device=device1,
                dtype=torch.long,
            )
            logits = self.stage1.compute_logits(h_in[:, last_indices]).squeeze(0)

            temps = torch.tensor(
                [s.params.temperature for s in batch], device=device1, dtype=torch.float32
            )
            top_p = torch.tensor(
                [s.params.top_p for s in batch], device=device1, dtype=torch.float32
            )
            procs = [s.params.logit_processors for s in batch]
            has_procs = any(procs)
            toks = sampler(
                logits,
                temps,
                top_p=top_p,
                logit_processors=procs if has_procs else None,
                input_ids=[s.all_token_ids for s in batch] if has_procs else None,
            )
            for seq, tok in zip(batch, toks.tolist()):
                seq.output_ids.append(tok)

        for seq in batch:
            seq.length = seq.num_prompt
            self.stage0.kv_cache.store_prefix(seq.prompt_ids, seq.slot)
            self.stage1.kv_cache.store_prefix(seq.prompt_ids, seq.slot)

    def _decode_microbatched(self, batch: list[Sequence]):
        """Micro-batched decode: split the batch into micro-batches that pipeline
        through stages 0 and 1 with overlapped transfer. Default micro-batch size = 2
        balances pipeline utilisation on the PCIe-x1 link."""
        if not batch:
            return

        from ..layers.sampler import Sampler

        sampler = Sampler()
        dev0 = self.devices[0]
        dev1 = self.devices[1]
        device0 = f"cuda:{dev0}"
        device1 = f"cuda:{dev1}"
        microbatch_size = 2
        micros = [batch[i : i + microbatch_size] for i in range(0, len(batch), microbatch_size)]

        for mb in micros:
            # --- Stage 0 ---
            with torch.cuda.device(dev0):
                _clear_hadamard_cache()
                self.stage0.lin_cache.reset()
                c0 = self.stage0.kv_cache
                slots = [s.slot for s in mb]
                lengths = [s.length for s in mb]
                c0.ensure_capacity(slots, [n + 1 for n in lengths])
                ids_tok = torch.tensor([[s.last_token] for s in mb], device=device0)
                pos_tok = torch.tensor([[s.length] for s in mb], device=device0)

                ctx0 = ForwardContext(
                    is_prefill=False,
                    kv_cache=c0,
                    lin_cache=self.stage0.lin_cache,
                    slots=slots,
                    slot_lengths=lengths,
                )

                embed = self.stage0.embed
                assert embed is not None
                h_emb = embed(ids_tok)
                residual = None
                stage0_ctx = _remap_ctx(ctx0, self.stage0._remap(c0), self.stage0.lin_cache)
                for layer in self.stage0.layers:
                    h_emb, residual = layer(h_emb, pos_tok, stage0_ctx, residual)

                mb_packed = _pack_boundary(h_emb.contiguous(), residual)
                scheme = self._wire_scheme if self._wire_scheme is not None else select_wire_scheme(mb_packed)
                handle = send(mb_packed, dst=dev1, scheme=scheme)

            # --- Stage 1 ---
            with torch.cuda.device(dev1):
                _clear_hadamard_cache()
                c1 = self.stage1.kv_cache
                c1.ensure_capacity(slots, [n + 1 for n in lengths])

                mb_rcv: torch.Tensor = recv(handle)
                h_in, residual = _unpack_boundary(mb_rcv, self.hidden_size)

                ctx1 = ForwardContext(
                    is_prefill=False,
                    kv_cache=c1,
                    lin_cache=self.stage1.lin_cache,
                    slots=slots,
                    slot_lengths=lengths,
                )
                stage1_ctx = _remap_ctx(ctx1, self.stage1._remap(c1), self.stage1.lin_cache)
                for layer in self.stage1.layers:
                    h_in, residual = layer(h_in, pos_tok.to(device1), stage1_ctx, residual)

                norm1 = self.stage1.norm
                assert norm1 is not None
                h_in, _ = norm1(h_in, residual)
                logits_mb = self.stage1.compute_logits(h_in[:, -1])

                temps = torch.tensor(
                    [s.params.temperature for s in mb], device=device1, dtype=torch.float32
                )
                top_p = torch.tensor(
                    [s.params.top_p for s in mb], device=device1, dtype=torch.float32
                )
                procs = [s.params.logit_processors for s in mb]
                has_procs = any(procs)
                mb_toks = sampler(
                    logits_mb,
                    temps,
                    top_p=top_p,
                    logit_processors=procs if has_procs else None,
                    input_ids=[s.all_token_ids for s in mb] if has_procs else None,
                )

            for seq, tok in zip(mb, mb_toks.tolist()):
                seq.output_ids.append(tok)
                seq.length += 1

    def generate(
        self, prompts: list[list[int]], params: SamplingParams | None = None
    ) -> list[list[int]]:
        ids = [self.add_request(p, params) for p in prompts]
        while self.scheduler.has_work():
            self.step()
        return [self._out[i].output_ids for i in ids]
