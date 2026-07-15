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

import os

import torch

from ..layers.sampler import Sampler
from ..models.base import ForwardContext
from ..models.cache import MTPKVCache, RecurrentStateCache
from .cuda_graph import GraphedDecode, GraphedVerify, cuda_graph_enabled_by_env
from .drafters import NgramDrafter, cascade_draft, drafter_config
from .sequence import Sequence


class EngineRunner:
    def __init__(
        self,
        model,
        cache,
        *,
        device="cuda",
        enable_cuda_graph: bool | None = None,
        lin_cache=None,
        chunked_prefill_size: int = 0,
        spec_decode: bool | None = None,
        eos_id: int | None = None,
    ):
        self.model = model
        self.cache = cache
        self.device = device
        self.eos_id = eos_id
        self.lin_cache = lin_cache or RecurrentStateCache()
        self._chunk_size = chunked_prefill_size
        # MTP speculative decode is OPT-IN / OFF by default: the draft head is correct
        # (~92% accept on Qwen3.5-9B) but the verify path is a net decode SLOWDOWN on
        # this fleet (no int8 multi-query verify kernel for head_dim 256 -> a per-row
        # fp16 fallback that also breaks greedy bit-identity by the odd tie-break). Off
        # by default = zero regression risk; enable with FNI8SERVE_MTP_SPEC=1 or the
        # `spec_decode=True` kwarg once the verify kernel lands. Guarded further by
        # `_spec_decode_allowed` (greedy, text-only) whenever it IS enabled.
        self._spec_enabled = (
            (os.environ.get("FNI8SERVE_MTP_SPEC", "0") == "1")
            if spec_decode is None
            else spec_decode
        )
        # Families with recurrent (DeltaNet / lightning / short-conv) layers carry
        # per-slot decode state through `lin_cache`. Two consequences for the engine:
        # (1) each such layer's state is keyed per slot, so prefill must clear+bind
        # its slot and decode must bind the whole batch's slots; (2) the recurrence
        # is order-dependent, so multiple sequences can't be packed into one varlen
        # forward (that would run the scan across sequence boundaries) — they prefill
        # one at a time instead. Detected by a marker on the mixer modules.
        self.has_recurrent = any(getattr(m, "is_recurrent", False) for m in model.modules())
        # Per-slot fp16 KV store for the MTP draft head's own attention (prefix-KV
        # speculative decode). The head is a separate transformer layer from the
        # backbone, so it keeps its own K/V over the committed context — primed at
        # prefill and each accepted step. Only used when the head exposes the split
        # project/attend interface (Qwen3.5 gated block); otherwise the draft stays
        # cache-free. Mirrors qengine's inline nextn head (Haru-neo/qengine, Apache-2.0).
        self.mtp_cache = MTPKVCache()
        _mtp = getattr(model, "mtp", None)
        self._mtp_prefix_kv = _mtp is not None and _mtp.supports_prefix_kv()
        # -- speculative-decode drafters (cascade: n-gram then MTP head) --------
        # `_spec_k` is the max draft length per step (the verify tensor is S=k+1).
        # n-gram (prompt-lookup) proposes up to k tokens for free on repetitive
        # spans; the MTP head is the depth-1 fallback on a miss. Both feed the same
        # verify path. Configurable via FNI8SERVE_SPEC_{DRAFTER,K,NGRAM}.
        mode, self._spec_k, _min_n, _max_n = drafter_config()
        self._drafter_mode = mode
        self._ngram = (
            NgramDrafter(min_n=_min_n, max_n=_max_n, max_k=self._spec_k)
            if mode in ("cascade", "ngram")
            else None
        )
        # Recurrent (DeltaNet / short-conv) hybrids commit their per-slot state after
        # a spec step via the captured verify trajectory (no re-decode) — but only if
        # EVERY recurrent layer records one. Verify this once so a family with a
        # non-capturing recurrent layer safely falls back to plain decode.
        rec = [m for m in model.modules() if getattr(m, "is_recurrent", False)]
        self._spec_recurrent_ok = all(getattr(m, "spec_capture", False) for m in rec)
        # Speculative-decode acceptance telemetry (host-side ints; no device sync).
        # `steps` = spec steps, `drafts` = draft tokens proposed, `accepts` = drafts
        # that matched the target's greedy argmax. accept_rate = accepts / drafts.
        self.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
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
        # Spec-decode verify runs 100% eager by default (#259) — the per-step launch
        # overhead makes every spec mode a NET SLOWDOWN graphs-on despite high accept
        # (#266). `GraphedVerify` captures the fixed-draft-length verify forward the
        # same way `GraphedDecode` captures base decode, so the acceptance-length win
        # stacks on top of graphs instead of being eaten by dispatch. Created whenever
        # graphs are on + the model is capturable (it stays inert unless spec-decode
        # actually runs, so it is decoupled from `_spec_enabled`, which the bench /
        # server can flip AFTER construction); `FNI8SERVE_VERIFY_GRAPH=0` forces the
        # eager verify (the two are byte-identical — tests/test_graph_verify.py). It
        # captures one graph per (batch, context, draft-length) bucket, so a changed
        # `_spec_k` just triggers a fresh capture.
        self._verify_graph_enabled = (
            self.graphed is not None
            and self.graphed.supported
            and os.environ.get("FNI8SERVE_VERIFY_GRAPH", "1") not in ("0", "false", "False")
        )
        self.graphed_verify = (
            GraphedVerify(self.graphed) if self._verify_graph_enabled else None
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
                tok = self._sample(logits, [seq])[0]
                out.append(tok)
                # Prime the MTP draft head's prefix-KV cache over the whole prompt
                # (hidden is the full [1,S,H] post-final-norm sequence here).
                self._prime_mtp_prefix(seq, hidden[0], tok)
        return out

    @torch.inference_mode()
    def _prime_mtp_prefix(self, seq: Sequence, hidden_seq: torch.Tensor, sampled_tok: int) -> None:
        """Fill ``seq``'s MTP prefix-KV cache from the prompt. ``hidden_seq`` is the
        main model's post-final-norm hidden [S, H] over the prompt positions; the
        first generated token ``sampled_tok`` is the token at position S.

        MTP-position ``i`` (0..S-1) stores K/V projected from ``(h_main[i],
        emb(token[i+1]))`` at RoPE position ``i`` — so the first draft (at length S,
        RoPE position S) attends the full committed prefix 0..S-1. Skipped unless the
        head supports the prefix-KV interface; also skipped for image prompts (the
        vision guard already disables spec-decode there)."""
        if not self._mtp_prefix_kv or getattr(self.model, "mtp", None) is None:
            return
        if seq.pixel_values is not None:
            return
        self.mtp_cache.clear_slot(seq.slot)
        S = hidden_seq.shape[0]
        if S < 1:
            return
        prev = hidden_seq[0:S].unsqueeze(0)  # [1,S,H] = h_main[0..S-1]
        toks = torch.tensor(
            [seq.prompt_ids[1:S] + [int(sampled_tok)]], device=self.device
        )  # [1,S] = token[1..S] (the embedding each MTP position consumes)
        positions = torch.arange(0, S, device=self.device).unsqueeze(0)  # RoPE i=0..S-1
        k, v = self.model.mtp.compute_prefix_kv(prev, toks, positions)  # [1,S,nkv,hd]
        self.mtp_cache.reset_slot(seq.slot, k[0], v[0])

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

        toks = self._sample(logits, batch)
        for b, seq in enumerate(batch):
            seq_hidden = hidden[0, cu_seqlens[b] : cu_seqlens[b + 1]]  # [S_b, H]
            self._prime_mtp_prefix(seq, seq_hidden, toks[b])
        return toks

    def _spec_decode_allowed(self, batch: list[Sequence], mtp) -> bool:
        """Gate speculative decode. It engages only when ALL hold:

        * at least one DRAFTER exists — either the learned MTP head (``mtp``) OR the
          model-agnostic n-gram lookup (``self._ngram``). The n-gram drafter carries
          NO head weights, so spec-decode runs on any model, including a quantized
          GGUF whose converter stripped the ``nextn.*`` MTP tensors
          (``num_mtp_layers==0`` → ``model.mtp is None``): the free prompt-lookup
          still proposes drafts and the shared verify path commits them. With neither
          drafter (``FNI8SERVE_SPEC_DRAFTER=mtp`` on a head-less model) there is
          nothing to propose, so spec-decode stays off (it would only verify base_tok
          each step — a pointless net slowdown).
        * every sequence is greedy (temp 0) — the accept-longest-greedy-prefix rule
          is only bit-identical to plain greedy;
        * NO sequence carries an image (``pixel_values``). MTP×vision is the exact
          interaction that broke in llama.cpp — a text-only spec path must never run
          the verify forward over spliced image embeds / image positions.

        Recurrent (DeltaNet / lightning / short-conv) hybrids are NOW SUPPORTED. The
        multi-token verify forward still advances the recurrent state by every drafted
        token, but ``_spec_decode_eager`` snapshots the post-base per-slot state, runs
        verify from it, restores the snapshot, and replays only the accepted tokens
        through the normal decode path (``_canonicalize_accepted_kv``) to reach the
        correct committed state — the recurrent analogue of the paged-KV accepted-token
        canonicalizer. The gated/linear mixers are wired through the verify path too
        (``GatedGQAAttention``/``GQAAttention._verify_batched``), so the M=1 paged
        assert no longer fires on the S=k+1 verify tensor.
        """
        if mtp is None and self._ngram is None:
            return False
        if not all(s.params.temperature == 0.0 for s in batch):
            return False
        if any(s.pixel_values is not None for s in batch):
            return False
        # Recurrent hybrid whose state can't be committed from the verify trajectory
        # (a non-capturing recurrent layer) — plain decode instead of a wrong commit.
        if self.has_recurrent and not self._spec_recurrent_ok:
            return False
        return True

    @torch.inference_mode()
    def decode(self, batch: list[Sequence]) -> list[int] | None:
        mtp = getattr(self.model, "mtp", None)
        if self._spec_enabled and self._spec_decode_allowed(batch, mtp):
            return self._spec_decode_eager(batch, mtp)
        logits = self.graphed.try_decode(batch) if self.graphed is not None else None
        if logits is None:
            return self._decode_eager(batch)
        for s in batch:
            s.length += 1
        return self._sample(logits, batch)

    @torch.inference_mode()
    def _spec_decode_eager(self, batch: list[Sequence], mtp) -> None:
        """Restructured speculative-decode step (spec loop, task 1). Emits, per
        sequence, ``1 + (#accepted drafts)`` tokens from ONE weight-stream (the verify
        forward). The redundant per-step base forward and the accepted-token re-decode
        (``_canonicalize_accepted_kv``) are both gone:

          * **base pipelined (c):** ``base_tok``/``base_hidden`` for a step come from
            the PRIOR step's verify (the true token + hidden at that step's last
            accepted slot), carried on the sequence. Only a sequence's FIRST spec step
            runs a base forward (to bootstrap the carry + write its last token's K/V);
            every step after reads weights exactly once, in verify.
          * **no re-decode (b):** the verify forward computes attention through the
            SAME paged-decode kernel plain decode uses (``GQAAttention._verify_batched``
            walks the verify tokens as consecutive paged-decode steps), so every
            committed token's logits and K/V are byte-identical to non-spec greedy — no
            canonicalizing replay, no dequant→requant, no dedicated verify kernel.
          * **recurrent state captured in verify (d):** DeltaNet / short-conv layers
            record their per-token state trajectory during verify; after acceptance the
            state after each row's last accepted token is committed directly — no
            snapshot/restore/replay.

        Returns ``None`` — ``llm_engine.step`` skips its per-sequence append on None.
        """
        B = len(batch)
        device = self.device
        slots = [s.slot for s in batch]

        # -- 1. Bootstrap base forward for sequences on their FIRST spec step ----
        # A newly-admitted (just-prefilled) sequence has no pipelining carry yet: run
        # ONE base forward over exactly those rows to write their last token's K/V,
        # advance their recurrent state by one, and produce (base_tok, base_hidden).
        # In steady state every row already carries this from the prior verify, so no
        # base forward runs at all.
        need = [b for b in range(B) if batch[b].spec_base_tok is None]
        if need:
            nb = [batch[b] for b in need]
            nslots = [s.slot for s in nb]
            nlens = [s.length for s in nb]
            ids = torch.tensor([[s.last_token] for s in nb], device=device)
            pos = torch.tensor([[s.length] for s in nb], device=device)
            self.cache.ensure_capacity(nslots, [n + 1 for n in nlens])
            self.lin_cache.bind(nslots)
            bctx = ForwardContext(
                is_prefill=False, kv_cache=self.cache, lin_cache=self.lin_cache,
                slots=nslots, slot_lengths=nlens,
            )
            bh = self.model(ids, pos, bctx)  # [nb, 1, H]
            btok = self.model.compute_logits(bh[:, -1]).argmax(-1)  # [nb]
            for j, b in enumerate(need):
                batch[b].spec_base_tok = int(btok[j])
                batch[b].spec_base_hidden = bh[j, -1].clone()

        base_tok = torch.tensor([s.spec_base_tok for s in batch], device=device)  # [B]
        base_hidden = torch.stack([s.spec_base_hidden for s in batch], dim=0)  # [B, H]
        lengths = [s.length for s in batch]

        # -- 2. Draft: cascade (n-gram first, MTP head fallback) ----------------
        # Per-row variable-length drafts; the verify tensor is sized to the longest.
        draft_lists = self._compute_drafts(mtp, base_hidden, base_tok, lengths, slots, batch)
        # Graphed verify captures ONE fixed shape, so pad every row to the full spec_k
        # (S = spec_k+1 constant) when it's active; else size to the longest real draft
        # (rejected pads are harmless either way — the accept loop only walks each row's
        # REAL drafts). Result-identical: the extra pad positions never get accepted.
        actual_len = [len(d) for d in draft_lists]
        # Graphed verify captures ONE fixed S per (batch, context, S) bucket, so the
        # verify width must be shape-static — but forcing the full spec_k every step
        # wastes compute the graph can't hide when acceptance is low: a spec_k+1-wide
        # verify to emit ~1 token REGRESSES prose (measured ngram 0.41×→0.32× at fixed
        # K=6) even though it wins high-acceptance structured. So bucket the width UP to
        # the actual longest draft THIS step (a handful of reusable S-graphs {2,3,5,7}),
        # matching eager's variable width while staying capturable: MTP (draft 1) and
        # low-hit prose n-gram collapse to S=2, structured cascade keeps S=spec_k+1.
        k_step = (
            self._draft_bucket(max(1, max(actual_len)))
            if self.graphed_verify is not None
            else max(1, max(actual_len))
        )
        S = k_step + 1  # verify tokens: base_tok + k_step drafts
        drafts_mat = [(d + [0] * (k_step - len(d))) for d in draft_lists]  # [B][k_step]

        # -- 3. Verify forward: [base_tok, draft_1..k] in ONE causal pass -------
        verify_ids_l = [[int(base_tok[b])] + drafts_mat[b] for b in range(B)]  # [B][S]
        verify_pos_l = [[lengths[b] + 1 + t for t in range(S)] for b in range(B)]
        hidden_v = true_tokens = None
        if self.graphed_verify is not None:
            res = self.graphed_verify.try_run(batch, verify_ids_l, verify_pos_l, lengths)
            if res is not None:
                hidden_v, true_tokens = res  # KV writes + recurrent traj done in-graph
        if hidden_v is None:  # eager verify (graph miss / disabled)
            verify_ids = torch.tensor(verify_ids_l, device=device)  # [B, S]
            verify_pos = torch.tensor(verify_pos_l, device=device)
            self.cache.ensure_capacity(slots, [n + 1 + S for n in lengths])
            flat_slots, flat_positions = [], []
            for b in range(B):
                base = lengths[b] + 1
                for t in range(S):
                    flat_slots.append(slots[b])
                    flat_positions.append(base + t)
            verify_slot_mapping = self.cache.slot_mapping_for(flat_slots, flat_positions)
            v_ctx = ForwardContext(
                is_prefill=False, is_verify=True, kv_cache=self.cache, lin_cache=self.lin_cache,
                slots=slots, slot_lengths=[n + 1 for n in lengths],
                verify_slot_mapping=verify_slot_mapping,
            )
            if self.has_recurrent:
                self.lin_cache.bind(slots)
                self.lin_cache.begin_verify_capture()
            hidden_v = self.model(verify_ids, verify_pos, v_ctx)  # [B, S, H]
            logits_v = self.model.compute_logits(hidden_v)  # [B, S, vocab]
            true_tokens = logits_v.argmax(-1)  # [B, S] — true greedy at each verify slot

        # -- 4. Accept longest greedy prefix + commit (no re-decode) ------------
        row_last = []  # per-row 0-based index of the last accepted verify slot
        for b in range(B):
            seq = batch[b]
            orig_len = seq.length
            # base_tok (slot 0) is always accepted; then match drafts against truth,
            # only over the row's REAL (non-pad) drafts.
            n_draft_acc = 0
            for t in range(actual_len[b]):
                if drafts_mat[b][t] == int(true_tokens[b, t]):
                    n_draft_acc += 1
                else:
                    break
            # Tokens EMITTED this step = base_tok + accepted drafts (positions L+1..L+
            # n_acc). base_tok is a fresh token here (it was NOT emitted before: it is
            # either this step's bootstrap-base output, or the prior step's carried
            # next-base which is emitted only now — exactly the old loop's `accepted`).
            accepted = [int(base_tok[b])] + drafts_mat[b][:n_draft_acc]
            # The NEXT step's base = the true token after the last accepted slot; carried
            # (NOT emitted this step — it is emitted next step as that step's base).
            next_base_tok = int(true_tokens[b, n_draft_acc])
            next_base_hidden = hidden_v[b, n_draft_acc]
            # Truncate at EOS / budget so greedy spec-decode stops exactly where plain
            # greedy would (emit only the surviving prefix of `accepted`).
            emit = self._truncate_emit(seq, accepted)
            n_acc = len(emit)  # committed tokens = base + accepted drafts (post-truncate)
            for tok in emit:
                seq.output_ids.append(tok)
            # KV committed this step: verify wrote slots 0..k_step (positions L+1..L+S);
            # accepted committed positions are L+1..L+n_acc, so length advances by n_acc.
            seq.length = orig_len + n_acc
            row_last.append(n_acc - 1)  # recurrent state after slot n_acc-1
            # Carry the next base for the next step, UNLESS this step truncated (EOS /
            # budget) — then the sequence finishes and the carry is invalid.
            truncated = n_acc < len(accepted) or seq.is_finished(self.eos_id)
            if not truncated:
                seq.spec_base_tok = next_base_tok
                seq.spec_base_hidden = next_base_hidden.clone()
            else:
                seq.spec_base_tok = None
                seq.spec_base_hidden = None
            # Telemetry: real drafts proposed vs accepted (host-side ints).
            self.spec_stats["drafts"] += actual_len[b]
            self.spec_stats["accepts"] += min(n_draft_acc, n_acc - 1)
            # Extend the MTP draft head's prefix-KV cache with the committed tokens so
            # the NEXT step's draft attends them. MTP-position i (orig_len..orig_len+
            # n_acc-1) stores K/V from (h_main[i], emb(token[i+1])) at RoPE i, where
            # token[i+1] = accepted[i-orig_len] (base_tok is token@orig_len+1),
            # h_main[orig_len] = base_hidden and h_main[orig_len+j] = hidden_v[j-1].
            if self._mtp_prefix_kv and self.mtp_cache.length(seq.slot) == orig_len and n_acc >= 1:
                mtp_toks = accepted[:n_acc]  # token[i+1] for i=orig_len..orig_len+n_acc-1
                prev_rows = [base_hidden[b]] + [hidden_v[b, j - 1] for j in range(1, n_acc)]
                prev = torch.stack(prev_rows, dim=0).unsqueeze(0)  # [1,n_acc,H]
                toks = torch.tensor([mtp_toks], device=device)  # [1,n_acc]
                positions = torch.arange(orig_len, orig_len + n_acc, device=device).unsqueeze(0)
                k_new, v_new = self.model.mtp.compute_prefix_kv(prev, toks, positions)
                self.mtp_cache.append(seq.slot, k_new[0], v_new[0])
            elif self._mtp_prefix_kv:
                self.mtp_cache.clear_slot(seq.slot)

        # -- 5. Commit recurrent state from the captured verify trajectory ------
        if self.has_recurrent:
            self.lin_cache.commit_verify(row_last)
        self.spec_stats["steps"] += 1

    def _draft_bucket(self, n: int) -> int:
        """Round a step's longest real draft ``n`` UP to a small fixed set of verify
        widths so the graphed verify reuses a handful of captured S-graphs instead of
        one-per-length. The top bucket is always ``self._spec_k`` (drafts are capped
        there), so ``n`` is never truncated; ``_spec_k`` is read live because the
        server/bench can retune it after construction."""
        for b in (1, 2, 4, 8, 16, 32):
            if b >= self._spec_k:
                break
            if n <= b:
                return b
        return self._spec_k

    def _truncate_emit(self, seq: Sequence, emit: list[int]) -> list[int]:
        """Trim a step's emitted tokens so greedy spec-decode matches plain greedy:
        stop at the request's ``max_tokens`` budget and (unless ``ignore_eos``) at the
        first EOS (inclusive). Returns the kept prefix (always >= 1 token)."""
        remaining = seq.params.max_tokens - len(seq.output_ids)
        if remaining < len(emit):
            emit = emit[: max(1, remaining)]
        eos = self.eos_id
        if eos is not None and not seq.params.ignore_eos:
            for i, tok in enumerate(emit):
                if tok == eos:
                    return emit[: i + 1]
        return emit

    @torch.inference_mode()
    def _compute_drafts(self, mtp, base_hidden, base_tok, lengths, slots, batch) -> list:
        """Propose a per-row draft token LIST (variable length) via the cascade:
        n-gram (prompt-lookup) first, MTP head fallback. The n-gram context is the
        sequence's tokens so far PLUS ``base_tok`` (the first verify token), so it
        proposes the continuation that follows ``base_tok``. The MTP head is depth-1
        (one token) and attends its own prefix-KV over the committed context.

        Single draft-dispatch seam (tests may patch this)."""
        B = base_tok.shape[0]
        out = []
        for b in range(B):
            seq = batch[b]

            def mtp_fallback(b=b):
                if self._mtp_prefix_kv and mtp is not None and mtp.num_depths() == 1:
                    pk, pv = self.mtp_cache.read(slots[b])
                    rope = torch.tensor([[lengths[b]]], device=self.device)
                    tok, _kc, _vc = mtp.draft_greedy_cached(
                        base_hidden[b : b + 1].unsqueeze(1),
                        base_tok[b : b + 1].unsqueeze(-1), rope, pk, pv,
                    )
                    return [int(tok[0, 0])]
                return []

            ctx_tokens = seq.all_token_ids + [int(base_tok[b])]
            # "ngram" mode = n-gram only (no MTP fallback on a miss); "mtp" mode has
            # self._ngram is None so the cascade uses only the fallback; "cascade" uses
            # both. A step where the drafter proposes nothing still verifies base_tok
            # (k_step floored at 1) — it just commits 1 token like plain decode.
            fb = None if self._drafter_mode == "ngram" else mtp_fallback
            dl = cascade_draft(self._ngram, fb, ctx_tokens, self._spec_k)
            out.append(dl)
        return out

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
