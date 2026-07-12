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
        self, model, cache, *, device="cuda", enable_cuda_graph: bool | None = None,
        lin_cache=None, chunked_prefill_size: int = 0,
    ):
        self.model = model
        self.cache = cache
        self.device = device
        self.lin_cache = lin_cache or RecurrentStateCache()
        self._chunk_size = chunked_prefill_size
        # Families with recurrent (DeltaNet / lightning / short-conv) layers carry
        # per-slot decode state through `lin_cache`. Two consequences for the engine:
        # (1) each such layer's state is keyed per slot, so prefill must clear+bind
        # its slot and decode must bind the whole batch's slots; (2) the recurrence
        # is order-dependent, so multiple sequences can't be packed into one varlen
        # forward (that would run the scan across sequence boundaries) — they prefill
        # one at a time instead. Detected by a marker on the mixer modules.
        self.has_recurrent = any(getattr(m, "is_recurrent", False) for m in model.modules())
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
        self.graphed = (
            GraphedDecode(model, cache, device=device, lin_cache=self.lin_cache)
            if enable_cuda_graph
            else None
        )
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
            self.lin_cache.clear_slot(seq.slot)
            self.lin_cache.bind([seq.slot])
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

        if len(batch) > 1 and isinstance(self.cache, PagedKVCache) and not self.has_recurrent:
            return self._prefill_varlen(batch)
        out = []
        for seq in batch:
            self.lin_cache.clear_slot(seq.slot)
            self.lin_cache.bind([seq.slot])
            n = seq.num_prompt
            self.cache.ensure_capacity([seq.slot], [n])
            chunk_size = self._chunk_size
            if chunk_size > 0 and n > chunk_size:
                out.append(self._prefill_chunked(seq))
            else:
                ids = torch.tensor([seq.prompt_ids], device=self.device)
                pos = torch.arange(n, device=self.device).unsqueeze(0)
                ctx = ForwardContext(
                    is_prefill=True,
                    kv_cache=self.cache,
                    lin_cache=self.lin_cache,
                    slots=[seq.slot],
                    prefill_start=seq.prefix_matched_len,
                    pixel_values=seq.pixel_values,
                    image_grid_thw=seq.image_grid_thw,
                )
                hidden = self.model(ids, pos, ctx)
                seq.length = n
                logits = self.model.compute_logits(hidden[:, -1])
                out.append(self._sample(logits, [seq])[0])
        return out

    def _prefill_chunked(self, seq: Sequence) -> int:
        """Chunked prefill: split the prompt into chunks, accumulate fp16 K/V
        across chunks for bit-identical attention, write each chunk to the paged
        cache, and sample from the last token.

        When *prefix_matched_len > 0*, the shared prefix tokens are still
        processed through the model (accumulating fp16 K/V for attention) but
        their K/V is NOT written to the paged cache (it is already there from
        the original request that filled the prefix)."""
        n = seq.num_prompt
        chunk_size = self._chunk_size
        prompt = seq.prompt_ids
        prefix_len = seq.prefix_matched_len
        acc_buf: list = []
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            chunk_ids = prompt[chunk_start:chunk_end]
            ids = torch.tensor([chunk_ids], device=self.device)
            pos = torch.arange(chunk_start, chunk_end, device=self.device).unsqueeze(0)
            ctx = ForwardContext(
                is_prefill=True,
                kv_cache=self.cache,
                lin_cache=self.lin_cache,
                slots=[seq.slot],
                prefill_start=prefix_len,
                pixel_values=seq.pixel_values,
                image_grid_thw=seq.image_grid_thw,
                acc_kv_buffer=acc_buf,
            )
            hidden = self.model(ids, pos, ctx)
        seq.length = n
        logits = self.model.compute_logits(hidden[:, -1])
        return self._sample(logits, [seq])[0]

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

    def _spec_decode_allowed(self, batch: list[Sequence], mtp) -> bool:
        """Gate MTP speculative decode. It engages only when ALL hold:

        * an MTP head exists and every sequence is greedy (temp 0) — the
          accept-longest-greedy-prefix rule is only bit-identical to plain greedy;
        * the model has NO recurrent layers. The multi-token verify forward advances
          DeltaNet / lightning / short-conv recurrent state by EVERY drafted token,
          but only the accepted prefix should count, and the paged-KV canonicalizer
          can't roll back a running scan (it's position-addressed, the recurrence is
          not). Verify also isn't wired through the gated / linear mixers at all
          (``GatedGQAAttention``/``GatedDeltaNetAttention`` ignore ``is_verify`` and
          the M=1 paged-decode kernel asserts on the S=k+1 verify tensor). So on a
          hybrid family (Qwen3.5, Qwen3-Next, MiniMax) spec-decode is UNSAFE — fall
          back to plain decode until a recurrent-aware verify path exists;
        * NO sequence carries an image (``pixel_values``). MTP×vision is the exact
          interaction that broke in llama.cpp; the vision families here are also the
          recurrent ones, so this is belt-and-suspenders — a text-only spec path must
          never run the verify forward over spliced image embeds / image positions.
        """
        if mtp is None:
            return False
        if not all(s.params.temperature == 0.0 for s in batch):
            return False
        if self.has_recurrent:
            return False
        if any(s.pixel_values is not None for s in batch):
            return False
        return True

    @torch.inference_mode()
    def decode(self, batch: list[Sequence]) -> list[int] | None:
        mtp = getattr(self.model, "mtp", None)
        if self._spec_decode_allowed(batch, mtp):
            return self._spec_decode_eager(batch, mtp)
        logits = self.graphed.try_decode(batch) if self.graphed is not None else None
        if logits is None:
            return self._decode_eager(batch)
        for s in batch:
            s.length += 1
        return self._sample(logits, batch)

    @torch.inference_mode()
    def _spec_decode_eager(self, batch: list[Sequence], mtp) -> None:
        """Speculative-decode step: draft *k* tokens via MTP heads, verify in
        ONE target forward with :func:`fni8.attn_int8_verify`, accept the longest
        greedy-matching prefix, and commit all accepted tokens to each sequence's
        ``output_ids`` and ``length``.

        Returns ``None`` — the caller (``llm_engine.step``) must skip its usual
        per-sequence token append when it sees a ``None`` return."""
        B = len(batch)
        device = self.device
        k = mtp.num_depths()
        slots = [s.slot for s in batch]
        lengths = [s.length for s in batch]

        # -- 1. Base forward: run the model on the last token -------------------
        ids = torch.tensor([[s.last_token] for s in batch], device=device)
        pos = torch.tensor([[s.length] for s in batch], device=device)
        self.cache.ensure_capacity(slots, [n + 1 for n in lengths])
        self.lin_cache.bind(slots)
        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=self.cache,
            lin_cache=self.lin_cache,
            slots=slots,
            slot_lengths=lengths,
        )
        hidden = self.model(ids, pos, ctx)
        base_logits = self.model.compute_logits(hidden[:, -1])  # [B, vocab]
        base_tok = base_logits.argmax(-1)  # [B]

        # -- 2. MTP draft: propose k candidate tokens (greedy) ------------------
        drafts = mtp.draft_greedy(hidden[:, -1:], base_tok.unsqueeze(-1), pos, ctx)
        # drafts: list of k tensors, each [B, 1]

        # -- 3. Verify forward: run the main model on [base, draft_1..k] --------
        S = k + 1  # total verify tokens
        verify_ids = torch.cat([base_tok.unsqueeze(-1)] + drafts, dim=-1)  # [B, S]
        verify_pos = torch.tensor(
            [[s.length + 1 + t for t in range(S)] for s in batch], device=device
        )

        # Ensure enough KV blocks for all verify positions. The highest verify
        # position is `n + 1 + k` (base at n+1, then k drafts), so the cache must
        # hold `n + 2 + k == n + 1 + S` tokens — the earlier `n + 1 + k` was one
        # short and could IndexError when that top position crossed a block
        # boundary (e.g. n+1+k a multiple of block_size).
        self.cache.ensure_capacity(slots, [n + 1 + S for n in lengths])

        # Precompute flat slot mapping for the verify token positions
        flat_slots, flat_positions = [], []
        for s in batch:
            base = s.length + 1
            for t in range(S):
                flat_slots.append(s.slot)
                flat_positions.append(base + t)
        verify_slot_mapping = self.cache.slot_mapping_for(flat_slots, flat_positions)

        # The prefix now has `length + 1` committed tokens (step 1 wrote at pos `length`)
        v_ctx = ForwardContext(
            is_prefill=False,
            is_verify=True,
            kv_cache=self.cache,
            lin_cache=self.lin_cache,
            slots=slots,
            slot_lengths=[n + 1 for n in lengths],
            verify_slot_mapping=verify_slot_mapping,
        )
        hidden_v = self.model(verify_ids, verify_pos, v_ctx)
        logits_v = self.model.compute_logits(hidden_v)  # [B, S, vocab]
        true_tokens = logits_v.argmax(-1)  # [B, S] — true greedy at each verify slot

        # -- 4. Accept the longest greedy-matching prefix -----------------------
        # Intermediate accepted tokens (all but the last, when n_acc>=2) need their
        # K/V re-committed canonically -- see _canonicalize_accepted_kv below.
        canon_rows: list[list[tuple[int, int]]] = []  # per-seq [(token, position)]
        for b in range(B):
            seq = batch[b]
            orig_len = seq.length
            # base token (index 0 in verify input) is always accepted
            n_acc = 1
            for t in range(k):
                draft_tok = drafts[t][b, 0].item()
                true_tok = true_tokens[b, t].item()  # true prediction for position N+2+t
                if draft_tok == true_tok:
                    n_acc += 1
                else:
                    break
            # Never emit past the request's token budget: a spec step can accept
            # several tokens at once, so cap n_acc so the sequence stops at exactly
            # max_tokens (greedy spec-decode must return the SAME token stream as
            # non-spec greedy, not one that overshoots). Capping n_acc only shortens
            # the committed prefix; the surplus verify positions are simply never
            # read (reads are gated by the sequence's committed length).
            remaining = seq.params.max_tokens - len(seq.output_ids)
            n_acc = max(1, min(n_acc, remaining))
            # Commit accepted tokens
            accepted = [base_tok[b].item()] + [drafts[t][b, 0].item() for t in range(n_acc - 1)]
            for tok in accepted:
                seq.output_ids.append(tok)
            seq.length += n_acc
            # The verify forward committed every accepted token's K/V using the
            # verify-attention path, whose prefix is dequantized then REQUANTIZED
            # (build_verify_cache) -- numerically slightly different from the paged
            # decode path (attn_paged_decode_cached reads the paged int8 directly).
            # The LAST accepted token is rewritten canonically by the next step's
            # base forward, but INTERMEDIATE accepted tokens (n_acc>=2) are never
            # rewritten, so their cache entry keeps the verify path's off-by-a-quant
            # K/V. That corrupts later reads and breaks greedy bit-identity on
            # tight-margin prompts (verified: layer-1 pos-of-the-intermediate-token
            # K/V differed). Re-decode them so the cache matches the decode path.
            canon_rows.append(
                [(accepted[i], orig_len + 1 + i) for i in range(n_acc - 1)]
            )
        self._canonicalize_accepted_kv(batch, canon_rows)

    @torch.inference_mode()
    def _canonicalize_accepted_kv(
        self, batch: list["Sequence"], canon_rows: list[list[tuple[int, int]]]
    ) -> None:
        """Re-commit intermediate accepted tokens' K/V through the canonical paged
        decode path so the cache is byte-identical to non-spec decode.

        ``canon_rows[b]`` lists ``(token, position)`` for sequence ``batch[b]``'s
        intermediate accepted tokens (in ascending position). Tokens are processed
        one index-round at a time across the batch so each attends an already
        canonicalized prefix; each round is ONE batched decode forward whose logits
        are discarded (we already know the tokens) -- only the K/V write matters."""
        max_rounds = max((len(r) for r in canon_rows), default=0)
        for i in range(max_rounds):
            rows = [(batch[b], canon_rows[b][i]) for b in range(len(batch)) if i < len(canon_rows[b])]
            if not rows:
                continue
            ids = torch.tensor([[tp[0]] for _, tp in rows], device=self.device)
            pos = torch.tensor([[tp[1]] for _, tp in rows], device=self.device)
            slots = [s.slot for s, _ in rows]
            # slot_lengths == the write position: _decode_batched writes each token's
            # K/V at slot_lengths[j] and attends context 0..slot_lengths[j] (itself
            # included), the exact canonical causal write for that position.
            slot_lengths = [tp[1] for _, tp in rows]
            self.cache.ensure_capacity(slots, [p + 1 for p in slot_lengths])
            self.lin_cache.bind(slots)
            ctx = ForwardContext(
                is_prefill=False,
                kv_cache=self.cache,
                lin_cache=self.lin_cache,
                slots=slots,
                slot_lengths=slot_lengths,
            )
            self.model(ids, pos, ctx)  # canonical K/V write; logits unused

    @torch.inference_mode()
    def _decode_eager(self, batch: list[Sequence]) -> list[int]:
        ids = torch.tensor([[s.last_token] for s in batch], device=self.device)  # [B,1]
        pos = torch.tensor([[s.length] for s in batch], device=self.device)  # [B,1]
        slots = [s.slot for s in batch]
        lengths = [s.length for s in batch]
        self.cache.ensure_capacity(slots, [n + 1 for n in lengths])
        # Bind this batch's slots so each recurrent layer gathers/scatters its
        # per-slot state aligned to the batch rows (no-op for non-recurrent models).
        self.lin_cache.bind(slots)
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
