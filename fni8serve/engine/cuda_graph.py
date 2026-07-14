# SPDX-License-Identifier: MIT
"""GraphedDecode -- CUDA-graph capture/replay for `EngineRunner.decode`.

Decode is dispatch-bound, not compute-bound: a 28-layer 0.6B model launches
~3,200 CUDA kernels/step (~114/layer) for ~18ms of real GPU work inside a
131ms step (measured on Qwen3-0.6B int8, batch=8, V100 idx-4 -- see issue #42).
Capturing the whole decode step (model forward + logits) into a CUDA graph and
replaying it re-issues every one of those launches as ONE `cudaGraphLaunch`,
collapsing dispatch overhead toward the real kernel time.

Requirements for capture, and how each is met here:
  * **Static shapes.** Decode batches are padded up to a batch-size bucket
    (1, 2, 4, 8, 16, ...) and a max-context-length bucket; pad rows point at a
    dedicated scratch cache slot (allocated once, never freed) and are simply
    sliced off the output before sampling.
  * **Persistent input buffers.** `ids`/`pos`/`slot_mapping`/`block_table`/
    `context_lens` are allocated once per (batch_bucket, context_bucket) and
    refreshed with `copy_` before every `replay()` -- captured kernels always
    read/write the SAME memory, so replay only picks up new values if we
    mutate that memory in place; a captured graph has no way to consume a
    freshly-built python-list-derived tensor from a later call.
  * **No mid-step sync.** `PagedKVCache.decode_attn_static` takes
    `max_context_len` as a plain python int (the bucket value) instead of
    `context_lens.max().item()` -- see kv_cache.py.
  * **Sampling stays outside the graph.** The graph captures model forward +
    `compute_logits` only; `EngineRunner._sample` (argmax/top-p + the final
    `.tolist()` host copy) runs eagerly on the sliced, unpadded logits after
    `replay()`.
  * **Paged-decode attention consumes tensors, not python lists** -- see the
    `ctx.slot_mapping is not None` branch in `GQAAttention._decode_batched`.

Only plain full-attention (dense, non-MoE, no sliding-window) decode is
capturable: MoE's per-expert token routing (`mask.nonzero()`, a data-dependent
op) and the sliding-window fallback (`PagedKVCache.read_dense` + a python loop)
cannot be represented in a CUDA graph. `_check_supported` detects this up
front from the model config; unsupported models fall back to eager decode
for every step (logged once).
"""

from __future__ import annotations

import logging
import os

import torch

from ..models.base import ForwardContext
from .sequence import Sequence

log = logging.getLogger(__name__)

DEFAULT_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_CONTEXT_BUCKET_SIZE = 128  # context-length bucket granularity (rounds up)
DEFAULT_MAX_GRAPHS = 32  # cap the captured-graph set (each pins its own workspace)
_WARMUP_ITERS = 3


def cuda_graph_enabled_by_env(default: bool = True) -> bool:
    """`FNI8SERVE_CUDA_GRAPH=0` disables graphed decode (eager stays available for
    debugging); unset or any other value keeps the default."""
    val = os.environ.get("FNI8SERVE_CUDA_GRAPH")
    if val is None:
        return default
    return val not in ("0", "false", "False")


def _next_bucket(buckets: tuple[int, ...], n: int) -> int | None:
    for b in buckets:
        if n <= b:
            return b
    return None


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


class _CapturedGraph:
    __slots__ = (
        "graph",
        "ids",
        "pos",
        "slot_mapping",
        "block_table",
        "context_lens",
        "logits",
        "batch_bucket",
        "context_bucket",
        "ids_host",
        "pos_host",
        "context_lens_host",
        "slot_idx",
        "slot_idx_host",
    )

    def __init__(
        self,
        *,
        graph,
        ids,
        pos,
        slot_mapping,
        block_table,
        context_lens,
        logits,
        batch_bucket,
        context_bucket,
        ids_host,
        pos_host,
        context_lens_host,
        slot_idx=None,
        slot_idx_host=None,
    ):
        self.graph = graph
        self.ids = ids
        self.pos = pos
        self.slot_mapping = slot_mapping
        self.block_table = block_table
        self.context_lens = context_lens
        self.logits = logits  # static output buffer -- read it before the next replay()
        self.batch_bucket = batch_bucket
        self.context_bucket = context_bucket
        # Pinned host staging (issue #183): the per-step ids/pos/context_lens refresh
        # fills these in place then non_blocking-copies into the captured device
        # buffers above, instead of rebuilding `torch.tensor(list, device=cuda)` (a
        # blocking pageable H2D) every step. The device buffers keep the SAME pointer
        # the graph captured -- we only ever mutate their contents in place.
        self.ids_host = ids_host
        self.pos_host = pos_host
        self.context_lens_host = context_lens_host
        # Recurrent decode only (None otherwise): persistent device row->slot index
        # the captured recurrent-state gather/scatter reads, plus its pinned host
        # staging. Refreshed in place each step before replay -- same pointer the
        # graph captured (models/cache.py `bind_graph`).
        self.slot_idx = slot_idx
        self.slot_idx_host = slot_idx_host


class _CapturedVerify:
    __slots__ = (
        "graph", "ids", "pos", "verify_slot_mapping", "block_table", "context_lens",
        "hidden", "true_tokens", "batch_bucket", "context_bucket",
        "ids_host", "pos_host", "context_lens_host", "vsm_host",
        "slot_idx", "slot_idx_host", "vtraj",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class GraphedVerify:
    """CUDA-graph capture/replay for the spec-decode VERIFY forward — the sibling of
    :class:`GraphedDecode` that makes MTP + fused-DeltaNet spec-decode compose with
    graphs (issue #259). Base decode is graphed (~1 launch/step) but the spec verify
    ran 100% eager, so on this dispatch-bound fleet every spec mode was a NET SLOWDOWN
    graphs-on (#266: cascade 0.84×, mtp 0.72×) despite high acceptance + bit-identical
    output. This captures the multi-token verify forward at a FIXED draft length so the
    per-step launch overhead collapses to one ``cudaGraphLaunch``.

    What is captured (the dominant, shape-static GPU work): the verify forward over
    ``S = spec_k + 1`` tokens (every row padded to the full ``spec_k`` drafts, so S is a
    compile-time constant), including all qkv/o/mlp GEMMs, the paged verify attention
    (``GQAAttention._verify_batched``'s static branch), the per-token DeltaNet /
    short-conv recurrence + its per-token state trajectory, ``compute_logits`` over all
    S positions, and the greedy ``argmax`` per position. Reuses the decode graph's
    contract exactly: persistent device input buffers refreshed via ``copy_`` before
    replay, a fixed ``max_context_len`` bucket int (no ``.item()`` sync), and the
    fixed-address per-slot recurrent-state buffers (``bind_graph``).

    What stays eager (truly variable / host-side, off the captured critical path):
    drafting (n-gram / MTP), the accept-longest-greedy-prefix loop (host argmax
    compare), EOS/budget truncation, the MTP prefix-KV extension, and the final
    recurrent-state COMMIT by accept length — the last reads the captured trajectory
    (fixed-address graph-pool tensors) and scatters, per row, the state after that
    row's last accepted token, exactly as the eager path does.

    Bit-identity: the captured verify uses the SAME kernels and the SAME per-position
    context lengths as the eager verify, so accepted tokens + committed K/V + committed
    recurrent state are byte-identical (guarded by tests/test_graph_verify.py)."""

    def __init__(self, graphed_decode: "GraphedDecode"):
        self.gd = graphed_decode
        self.model = graphed_decode.model
        self.cache = graphed_decode.cache
        self.lin_cache = graphed_decode.lin_cache
        self.device = graphed_decode.device
        self.has_recurrent = graphed_decode.has_recurrent
        self.batch_buckets = graphed_decode.batch_buckets
        self.context_bucket_size = graphed_decode.context_bucket_size
        self.max_graphs = graphed_decode.max_graphs
        self.supported = graphed_decode.supported
        self._graphs: dict[tuple[int, int, int], _CapturedVerify] = {}

    # -- public entry point ------------------------------------------------
    def try_run(self, batch, verify_ids, verify_pos, lengths):
        """Replay (capturing on first hit) the verify forward for ``batch``.

        ``verify_ids``/``verify_pos``: python ``[B][S]`` lists (base token + spec_k
        drafts, padded to a fixed S) and their absolute positions. ``lengths``:
        committed length per row (``seq.length``). Returns ``(hidden_v[:B] [B,S,H],
        true_tokens[:B] [B,S])`` with the recurrent verify trajectory bound onto
        ``lin_cache`` for the runner's subsequent ``commit_verify`` — or ``None`` if this
        step can't be served from a graph (unsupported model, batch over the widest
        bucket, or the graph cap hit), so the caller falls back to the eager verify."""
        if not self.supported or not batch:
            return None
        B = len(batch)
        batch_bucket = _next_bucket(self.batch_buckets, B)
        if batch_bucket is None:
            return None
        S = len(verify_ids[0])  # base token + fixed spec_k drafts (compile-time per key)
        max_real_ctx = max(lengths) + 1 + S
        cap = self.cache.max_blocks_per_seq * self.cache.block_size
        context_bucket = min(_round_up(max_real_ctx, self.context_bucket_size), cap)
        if max_real_ctx > cap:
            return None  # verify would overrun the paged capacity — let eager handle it
        key = (batch_bucket, context_bucket, S)
        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) >= self.max_graphs:
                return None
            g = self._capture(batch_bucket, context_bucket, S)
            self._graphs[key] = g
            log.info(
                "cuda-graph verify: captured bucket batch=%d max_context=%d S=%d (%d graphs)",
                batch_bucket, context_bucket, S, len(self._graphs),
            )
        self._fill_inputs(g, batch, verify_ids, verify_pos, lengths)
        g.graph.replay()
        # Bind the captured recurrent trajectory + rebind eager (python-slot) so the
        # runner's commit_verify(row_last) scatters the accepted-length state onto the
        # real slots. bind() clears the graph row-index the replay used.
        if self.has_recurrent and g.vtraj is not None:
            self.lin_cache._vtraj = g.vtraj
            self.lin_cache._vcap = True
            self.lin_cache.bind([s.slot for s in batch])
        return g.hidden[:B], g.true_tokens[:B]

    # -- capture ------------------------------------------------------------
    def _capture(self, batch_bucket: int, context_bucket: int, S: int) -> _CapturedVerify:
        cache = self.cache
        scratch = self.gd._scratch()
        cache.ensure_capacity([scratch], [S + 1])
        dev = self.device

        ids = torch.zeros(batch_bucket, S, dtype=torch.long, device=dev)
        pos = torch.zeros(batch_bucket, S, dtype=torch.long, device=dev)
        flat_slots = [scratch] * (batch_bucket * S)
        flat_pos = [t for _ in range(batch_bucket) for t in range(S)]
        verify_slot_mapping = cache.slot_mapping_for(flat_slots, flat_pos)  # [Bmax*S]
        block_table = cache.block_table([scratch] * batch_bucket)
        context_lens = torch.ones(batch_bucket, dtype=torch.int32, device=dev)

        slot_idx = slot_idx_host = None
        if self.has_recurrent and self.lin_cache is not None:
            slot_idx = torch.full((batch_bucket,), scratch, dtype=torch.long, device=dev)
            slot_idx_host = torch.empty(batch_bucket, dtype=torch.long,
                                        pin_memory=(dev != "cpu"))
            self.lin_cache.bind_graph(slot_idx)

        def _ctx():
            return ForwardContext(
                is_prefill=False, is_verify=True, kv_cache=cache,
                lin_cache=self.lin_cache if self.has_recurrent else None,
                slots=[scratch] * batch_bucket,
                slot_lengths=[1] * batch_bucket,
                verify_slot_mapping=verify_slot_mapping,
                block_tables=block_table, context_lens=context_lens,
                max_context_len=context_bucket,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_ITERS):
                with torch.inference_mode():
                    if self.has_recurrent:
                        self.lin_cache.begin_verify_capture()
                    hidden = self.model(ids, pos, _ctx())
                    self.model.compute_logits(hidden).argmax(-1)
                    if self.has_recurrent:
                        self.lin_cache.end_verify_capture()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        if self.has_recurrent:
            self.lin_cache.begin_verify_capture()
        with torch.inference_mode(), torch.cuda.graph(graph):
            hidden = self.model(ids, pos, _ctx())
            logits = self.model.compute_logits(hidden)  # [Bmax, S, vocab]
            true_tokens = logits.argmax(-1)  # [Bmax, S]
        vtraj = None
        if self.has_recurrent:
            vtraj = dict(self.lin_cache._vtraj)  # fixed-address graph-pool trajectory
            self.lin_cache._vcap = False

        pin = dev != "cpu"
        return _CapturedVerify(
            graph=graph, ids=ids, pos=pos, verify_slot_mapping=verify_slot_mapping,
            block_table=block_table, context_lens=context_lens,
            hidden=hidden, true_tokens=true_tokens,
            batch_bucket=batch_bucket, context_bucket=context_bucket,
            ids_host=torch.empty(batch_bucket, S, dtype=torch.long, pin_memory=pin),
            pos_host=torch.empty(batch_bucket, S, dtype=torch.long, pin_memory=pin),
            context_lens_host=torch.empty(batch_bucket, dtype=torch.int32, pin_memory=pin),
            vsm_host=torch.empty(batch_bucket * S, dtype=torch.int32, pin_memory=pin),
            slot_idx=slot_idx, slot_idx_host=slot_idx_host, vtraj=vtraj,
        )

    # -- per-step input refresh (eager, before replay) ---------------------
    def _fill_inputs(self, g: _CapturedVerify, batch, verify_ids, verify_pos, lengths):
        cache, S = self.cache, g.ids.shape[1]
        B, Bmax = len(batch), g.batch_bucket
        scratch = self.gd._scratch()
        pad = Bmax - B
        real_slots = [s.slot for s in batch]
        cache.ensure_capacity(real_slots, [n + 1 + S for n in lengths])

        # ids / pos: real rows carry the verify tokens/positions; pad rows are inert.
        g.ids_host[:B].copy_(torch.tensor(verify_ids, dtype=torch.long))
        g.pos_host[:B].copy_(torch.tensor(verify_pos, dtype=torch.long))
        g.context_lens_host[:B].copy_(torch.tensor([n + 1 for n in lengths], dtype=torch.int32))
        if pad:
            g.ids_host[B:].zero_()
            g.pos_host[B:].zero_()
            g.context_lens_host[B:].fill_(1)
        g.ids.copy_(g.ids_host, non_blocking=True)
        g.pos.copy_(g.pos_host, non_blocking=True)
        g.context_lens.copy_(g.context_lens_host, non_blocking=True)

        # verify_slot_mapping [Bmax*S]: (row b, verify pos t) -> paged slot for position
        # length[b]+1+t; pad rows map every position onto the scratch slot.
        flat_slots, flat_pos = [], []
        for b in range(B):
            base = lengths[b] + 1
            flat_slots.extend([real_slots[b]] * S)
            flat_pos.extend(base + t for t in range(S))
        for _ in range(pad):
            flat_slots.extend([scratch] * S)
            flat_pos.extend(range(S))
        cache.fill_slot_mapping(g.verify_slot_mapping, flat_slots, flat_pos)
        cache.fill_block_table(g.block_table, real_slots + [scratch] * pad)

        if g.slot_idx is not None:
            g.slot_idx_host.copy_(torch.tensor(real_slots + [scratch] * pad, dtype=torch.long))
            g.slot_idx.copy_(g.slot_idx_host, non_blocking=True)
            self.lin_cache.bind_graph(g.slot_idx)


class GraphedDecode:
    """Lazily captures one CUDA graph per (batch_bucket, context_bucket) bucket
    pair, replays on an exact match, and returns None (caller falls back to
    eager `EngineRunner.decode`) on a miss: unsupported model, batch larger
    than the widest bucket, or the captured-graph cap already reached."""

    def __init__(
        self,
        model,
        cache,
        *,
        device: str = "cuda",
        lin_cache=None,
        batch_buckets: tuple[int, ...] = DEFAULT_BATCH_BUCKETS,
        context_bucket_size: int = DEFAULT_CONTEXT_BUCKET_SIZE,
        max_graphs: int = DEFAULT_MAX_GRAPHS,
    ):
        self.model = model
        self.cache = cache
        self.lin_cache = lin_cache
        self.device = device
        # Never bucket past what the engine could ever schedule (`cache.num_slots`
        # == `max_num_seqs`) -- a bigger bucket would just never be hit.
        self.batch_buckets = tuple(sorted(b for b in set(batch_buckets) if b <= cache.num_slots))
        self.context_bucket_size = context_bucket_size
        self.max_graphs = max_graphs
        self._graphs: dict[tuple[int, int], _CapturedGraph] = {}
        self._scratch_slot: int | None = None
        # Recurrent (short-conv / DeltaNet / lightning) mixers carry per-slot decode
        # state through `lin_cache`. When present AND capturable, we switch that cache
        # to fixed-address per-slot buffers so the recurrent update is static and can
        # be captured with the rest of the step (see models/cache.py).
        _modules = getattr(self.model, "modules", None)
        self.has_recurrent = callable(_modules) and any(
            getattr(m, "is_recurrent", False) for m in _modules()
        )
        self.supported, self._unsupported_reason = self._check_supported()
        if not self.supported:
            log.warning("cuda-graph decode disabled: %s", self._unsupported_reason)
        elif self.has_recurrent and self.lin_cache is not None:
            # Enable BEFORE the first prefill so prefill state lands in the same
            # fixed buffers the graphed decode replays against.
            self.lin_cache.enable_static_buffers(cache.num_slots)

    # -- capability check ------------------------------------------------
    def _check_supported(self) -> tuple[bool, str]:
        if self.device == "cpu" or not torch.cuda.is_available():
            return False, "no CUDA device"
        if not self.batch_buckets:
            return False, "no batch bucket <= max_num_seqs"
        cfg = self.model.config
        if cfg.is_moe():
            return False, "MoE routing is data-dependent (mask.nonzero()), not graph-capturable"
        # Latent attention (MLA / DeepSeek) decode reads a per-step-growing latent
        # slice out of MLALatentCache -- not yet expressible as a fixed-address
        # static buffer, so keep declining it (a separate optimization).
        if cfg.latent_attention:
            return False, "latent (MLA) decode reads a growing latent slice, not yet static"
        # Recurrent mixers (LFM2 short-conv, Qwen3-Next DeltaNet, MiniMax lightning)
        # ARE capturable now: their per-slot decode state lives in fixed-address
        # buffers (models/cache.py `enable_static_buffers`), gathered/scattered in
        # place through a persistent device row->slot index. So 'linear' layers are
        # accepted alongside 'full'; only 'sliding' (a data-dependent python-loop
        # fallback) and 'latent' remain uncapturable.
        for i in range(cfg.num_hidden_layers):
            kind = cfg.attention_kind(i)
            if kind not in ("full", "linear"):
                return (
                    False,
                    f"layer {i} uses the '{kind}' attention backend "
                    "(sliding-window decode is a data-dependent python-loop fallback)",
                )
        return True, ""

    def _scratch(self) -> int:
        """A dedicated cache slot every pad row's slot/block-table entries point
        at -- allocated once and never freed, so pad rows always read/write valid
        (if inert) memory regardless of which real sequences are live."""
        if self._scratch_slot is None:
            self._scratch_slot = self.cache.alloc()
            self.cache.ensure_capacity([self._scratch_slot], [1])
        return self._scratch_slot

    # -- public entry point -----------------------------------------------
    def try_decode(self, batch: list[Sequence]) -> torch.Tensor | None:
        """Returns next-token logits `[len(batch), vocab]` for `batch` via a
        captured graph, or None if this step can't be served from one (the
        caller should fall back to eager decode). Never mutates `Sequence`
        state -- the caller owns `seq.length` bookkeeping either way."""
        if not self.supported or not batch:
            return None
        B = len(batch)
        batch_bucket = _next_bucket(self.batch_buckets, B)
        if batch_bucket is None:
            log.info(
                "cuda-graph decode miss: batch size %d exceeds largest bucket %d",
                B,
                self.batch_buckets[-1],
            )
            return None
        max_real_ctx = max(s.length for s in batch) + 1
        cap = self.cache.max_blocks_per_seq * self.cache.block_size
        context_bucket = min(_round_up(max_real_ctx, self.context_bucket_size), cap)
        key = (batch_bucket, context_bucket)

        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) >= self.max_graphs:
                log.warning(
                    "cuda-graph decode miss: bucket cap (%d graphs) reached, "
                    "not capturing batch=%d max_context=%d",
                    self.max_graphs,
                    batch_bucket,
                    context_bucket,
                )
                return None
            g = self._capture(batch_bucket, context_bucket)
            self._graphs[key] = g
            log.info(
                "cuda-graph decode: captured bucket batch=%d max_context=%d (%d graphs total)",
                batch_bucket,
                context_bucket,
                len(self._graphs),
            )

        self._fill_inputs(g, batch)
        g.graph.replay()
        return g.logits[:B]

    # -- capture ------------------------------------------------------------
    def _capture(self, batch_bucket: int, context_bucket: int) -> _CapturedGraph:
        cache = self.cache
        scratch = self._scratch()

        ids = torch.zeros(batch_bucket, 1, dtype=torch.long, device=self.device)
        pos = torch.zeros(batch_bucket, 1, dtype=torch.long, device=self.device)
        slot_mapping = cache.slot_mapping_for([scratch] * batch_bucket, [0] * batch_bucket)
        block_table = cache.block_table([scratch] * batch_bucket)
        context_lens = torch.ones(batch_bucket, dtype=torch.int32, device=self.device)

        # Recurrent models: a persistent device row->slot index the captured
        # recurrent-state gather/scatter reads. Seed every row at the scratch slot
        # (warmup/capture must never touch a real sequence's state); _fill_inputs
        # refreshes it to the batch's real slots before each replay.
        slot_idx = slot_idx_host = None
        if self.has_recurrent and self.lin_cache is not None:
            slot_idx = torch.full((batch_bucket,), scratch, dtype=torch.long, device=self.device)
            pin = self.device != "cpu" and torch.cuda.is_available()
            slot_idx_host = torch.empty(batch_bucket, dtype=torch.long, pin_memory=pin)
            self.lin_cache.bind_graph(slot_idx)

        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=cache,
            lin_cache=self.lin_cache if self.has_recurrent else None,
            slot_mapping=slot_mapping,
            block_tables=block_table,
            context_lens=context_lens,
            max_context_len=context_bucket,
        )

        # Standard two-phase capture (PyTorch CUDA-graph guidance): warm up a few
        # iterations on a side stream first so the caching allocator reaches a
        # steady state (capture fails if it has to grow the pool mid-capture),
        # THEN capture. Warmup runs against the same scratch-only inputs the
        # graph is seeded with above -- it never touches a real sequence's data.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_ITERS):
                with torch.inference_mode():
                    hidden = self.model(ids, pos, ctx)
                    self.model.compute_logits(hidden[:, -1])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            hidden = self.model(ids, pos, ctx)
            logits = self.model.compute_logits(hidden[:, -1])

        pin = self.device != "cpu" and torch.cuda.is_available()
        ids_host = torch.empty(batch_bucket, 1, dtype=torch.long, pin_memory=pin)
        pos_host = torch.empty(batch_bucket, 1, dtype=torch.long, pin_memory=pin)
        context_lens_host = torch.empty(batch_bucket, dtype=torch.int32, pin_memory=pin)

        return _CapturedGraph(
            graph=graph,
            ids=ids,
            pos=pos,
            slot_mapping=slot_mapping,
            block_table=block_table,
            context_lens=context_lens,
            logits=logits,
            batch_bucket=batch_bucket,
            context_bucket=context_bucket,
            ids_host=ids_host,
            pos_host=pos_host,
            context_lens_host=context_lens_host,
            slot_idx=slot_idx,
            slot_idx_host=slot_idx_host,
        )

    # -- per-step input refresh (eager -- runs before replay(), not captured) ----
    def _fill_inputs(self, g: _CapturedGraph, batch: list[Sequence]):
        cache = self.cache
        B, Bmax = len(batch), g.batch_bucket
        scratch = self._scratch()
        pad = Bmax - B

        real_slots = [s.slot for s in batch]
        real_lengths = [s.length for s in batch]
        cache.ensure_capacity(real_slots, [n + 1 for n in real_lengths])

        slots = real_slots + [scratch] * pad
        lengths = real_lengths + [0] * pad

        # Fill the pinned host staging in place (CPU-only work, no GPU sync): real
        # rows carry the sequence's next token / position / context length, pad rows
        # are inert (id 0, pos 0, context_len 1 -- always point at the scratch slot).
        pin = g.ids_host.is_pinned() if hasattr(g.ids_host, "is_pinned") else False
        g.ids_host[:B, 0].copy_(torch.tensor([s.last_token for s in batch], dtype=torch.long))
        g.pos_host[:B, 0].copy_(torch.tensor([s.length for s in batch], dtype=torch.long))
        g.context_lens_host[:B].copy_(
            torch.tensor([s.length + 1 for s in batch], dtype=torch.int32)
        )
        if pad:
            g.ids_host[B:, 0].zero_()
            g.pos_host[B:, 0].zero_()
            g.context_lens_host[B:].fill_(1)

        # One non_blocking H2D per buffer, into the SAME device tensor the graph
        # captured (contents mutated in place -- pointer preserved for replay).
        g.ids.copy_(g.ids_host, non_blocking=pin)
        g.pos.copy_(g.pos_host, non_blocking=pin)
        g.context_lens.copy_(g.context_lens_host, non_blocking=pin)
        cache.fill_slot_mapping(g.slot_mapping, slots, lengths)
        cache.fill_block_table(g.block_table, slots)

        # Recurrent models: refresh the persistent row->slot index in place (real
        # rows -> their slot, pad rows -> scratch) so the captured recurrent-state
        # gather/scatter reads/writes each live sequence's own fixed buffer row.
        # Re-bind every step: an intervening eager step (prefill / fallback) clears
        # the graph index, so the graphed path must reinstate its own before replay.
        if g.slot_idx is not None:
            g.slot_idx_host.copy_(torch.tensor(slots, dtype=torch.long))
            g.slot_idx.copy_(g.slot_idx_host, non_blocking=pin)
            self.lin_cache.bind_graph(g.slot_idx)
