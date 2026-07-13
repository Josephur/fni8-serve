# SPDX-License-Identifier: MIT
"""GQAAttention — the shared full/sliding softmax-attention block on fni8 dp4a.

Config-driven so Qwen3, Qwen3-MoE, Gemma3, GLM, Hunyuan all reuse it unchanged:
  * merged QKV projection (int8 dp4a), split to GQA heads;
  * optional per-head RMSNorm on Q and K over head_dim, applied BEFORE RoPE
    (Qwen3 / Gemma3 QK-norm);
  * RoPE with per-layer theta (Gemma3 local vs global) and partial-rotary (GLM);
  * softmax scale = query_pre_attn_scalar**-0.5 (Gemma) or head_dim**-0.5;
  * sliding-window local layers via fni8's window_left (prefill) / cache slice (decode).

This is the `full`/`sliding` AttentionBackend. `linear` (DeltaNet) and `latent`
(MLA) backends are separate — see layers/linear_attn.py and layers/mla_attn.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import fni8
from fni8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm
from .rotary import RotaryEmbedding


class GQAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_proj: QTensor,
        o_proj: QTensor,
        scale: float,
        rope: RotaryEmbedding,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
        rms_norm_eps: float = 1e-6,
        window_left: int = -1,
        qkv_bias=None,
        o_bias=None,
        causal: bool = True,
    ):
        super().__init__()
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, head_dim
        self.scale = scale
        self.window_left = window_left
        self.causal = causal
        self.qkv_proj = LinearW8A8(qkv_proj, qkv_bias)
        self.o_proj = LinearW8A8(o_proj, o_bias)
        self.rope = rope
        self.q_norm = RMSNorm(head_dim, rms_norm_eps, q_norm) if q_norm is not None else None
        self.k_norm = RMSNorm(head_dim, rms_norm_eps, k_norm) if k_norm is not None else None

    def forward(self, x, positions, ctx, layer_idx: int) -> torch.Tensor:
        """x: [B, S, hidden]; positions: [B, S] or [S]. Returns [B, S, hidden]."""
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(B, S, self.nh, self.hd)
        k = k.view(B, S, self.nkv, self.hd)
        v = v.view(B, S, self.nkv, self.hd)
        if self.q_norm is not None:  # per-head RMSNorm, pre-RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(positions, q, k)
        # q, k, v are [B, S, H, D] here. The transpose to the kernels' [B, H, S, D]
        # layout is now done ONLY in the two branches that genuinely need it
        # (non-varlen prefill, simple decode). The varlen-prefill and decode-batched
        # hot paths take token-major [total_tok, H, D], which is a zero-copy reshape
        # of [B, S, H, D] (Decode Lever 2: no more transpose().contiguous() churn).

        if ctx.is_prefill:
            if ctx.cu_seqlens is not None:
                # Varlen batched prefill: [1, S, H, D] -> [total_tok, H, D] via reshape
                # (B == 1 for the packed varlen layout; token-major, heads interleaved,
                # the flash_attn_varlen convention). No transpose, no contiguous copy.
                q_v = q.reshape(B * S, self.nh, self.hd)  # [total_tok, H, D]
                k_v = k.reshape(B * S, self.nkv, self.hd)  # [total_tok, Hkv, D]
                v_v = v.reshape(B * S, self.nkv, self.hd)  # [total_tok, Hkv, D]
                ctx.kv_cache.write_prefill_varlen(layer_idx, ctx.slot_mapping, k_v, v_v)
                max_seqlen = int((ctx.cu_seqlens[1:] - ctx.cu_seqlens[:-1]).max().item())
                out_v = fni8.attn_int8_varlen(
                    q_v,
                    k_v,
                    v_v,
                    ctx.cu_seqlens,
                    ctx.cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=self.causal,
                    scale=self.scale,
                )
                # out_v is [total_tok, H, D] == [B, S, H, D] (B==1) -> merge heads by reshape.
                out = out_v.reshape(B, S, self.nh * self.hd)
                return self.o_proj(out)
            else:
                # Non-varlen prefill genuinely needs [B, H, S, D].
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                slot = ctx.slots[0] if ctx.slots is not None else None
                ctx.kv_cache.write_prefill(
                    layer_idx, k, v, slot=slot, start=ctx.prefill_start, positions=positions,
                )
                # Chunked prefill: use accumulated fp16 K/V + current K/V with
                # attn_int8_fwd (same kernel as full prefill) by zero-padding Q
                # to match the accumulated K/V length.
                acc_buf = getattr(ctx, "acc_kv_buffer", None)
                if acc_buf is not None and layer_idx < len(acc_buf) and acc_buf[layer_idx] is not None:
                    k_prev, v_prev = acc_buf[layer_idx]
                    prev_len = k_prev.shape[2]
                    cur_len = k.shape[2]
                    k_all = torch.cat([k_prev, k], dim=2)
                    v_all = torch.cat([v_prev, v], dim=2)
                    total_len = k_all.shape[2]
                    q_pad = k_all.new_zeros(1, self.nh, total_len, self.hd)  # [1, H, total, D]
                    q_pad[:, :, -cur_len:, :] = q  # place actual Q at the end
                    out = fni8.attn_int8_fwd(
                        q_pad, k_all, v_all,
                        causal=self.causal, scale=self.scale, window_left=self.window_left,
                    )
                    out = out[:, :, -cur_len:, :]  # take only the actual Q positions
                    # Update buffer: store the concatenated K/V for next chunk
                    acc_buf[layer_idx] = (k_all.contiguous(), v_all.contiguous())
                else:
                    out = fni8.attn_int8_fwd(
                        q, k, v, causal=self.causal, scale=self.scale, window_left=self.window_left
                    )
                # Initialize or update accumulated buffer
                if acc_buf is not None and not (layer_idx < len(acc_buf) and acc_buf[layer_idx] is not None):
                    while len(acc_buf) <= layer_idx:
                        acc_buf.append(None)
                    acc_buf[layer_idx] = (k.contiguous(), v.contiguous())
        elif getattr(ctx, "is_verify", False):
            # Spec-decode verify: returns RAW [B, H, S, D]; the shared tail below
            # merges heads and applies o_proj exactly once (do NOT o_proj here).
            out = self._verify_batched(q, k, v, ctx, layer_idx)
        elif ctx.slot_lengths is not None or ctx.slot_mapping is not None:
            # Decode-batched hot path: takes [B, S=1, H, D] directly (no transpose here).
            out = self._decode_batched(q, k, v, ctx, layer_idx)  # engine continuous batch
            # (or CUDA-graph static path)
        else:
            # Simple decode genuinely needs [B, H, S, D].
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            k_all, v_all = ctx.kv_cache.append_decode(layer_idx, k, v)  # [B,Hkv,N,D]
            k_all, v_all = self._window(k_all, v_all)
            out = fni8.attn_int8_decode(q, k_all, v_all, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, S, self.nh * self.hd)
        return self.o_proj(out)

    def _window(self, k_all, v_all):
        if self.window_left >= 0 and k_all.shape[2] > self.window_left:
            k_all = k_all[:, :, -self.window_left :].contiguous()
            v_all = v_all[:, :, -self.window_left :].contiguous()
        return k_all, v_all

    def _decode_batched(self, q, k, v, ctx, layer_idx):
        """Continuous-batch decode: rows have different KV lengths (already batched
        GEMMs upstream). The paged int8 cache commits every row's new token with ONE
        `quantize_kv_write_paged` call and reads the whole ragged batch back with ONE
        `attn_paged_decode_cached` launch -- no more per-slot Python loop.

        Sliding-window layers are the one gap the paged-decode kernel doesn't cover
        (no window parameter yet), so they fall back to a per-slot dequantized read
        + `attn_int8_decode`, same as before this PR.

        Inputs q, k, v are [B, S=1, H, D] (token-major, as they leave RoPE). The one
        new token's K/V is sliced with a zero-copy index; q is transposed to the
        kernels' [B, H_q, 1, D] here (a size-1 permute — the write/attn kernels make
        their inputs contiguous internally)."""
        cache = ctx.kv_cache
        k_new, v_new = k[:, 0, :, :], v[:, 0, :, :]  # [B,Hkv,D]: the one new token
        q = q.transpose(1, 2)  # [B,S=1,H,D] -> [B,H,1,D] for the decode kernels
        if self.window_left < 0:
            if ctx.slot_mapping is not None:
                # CUDA-graph decode (engine/cuda_graph.py): slot_mapping/block_tables/
                # context_lens are persistent device buffers refreshed via `copy_`
                # before replay, and max_context_len is a compile-time bucket int --
                # no fresh per-layer tensor allocation and no `.item()` sync, so this
                # whole call is capturable.
                cache.write_decode_static(layer_idx, ctx.slot_mapping, k_new, v_new)
                return cache.decode_attn_static(
                    layer_idx,
                    q,
                    ctx.block_tables,
                    ctx.context_lens,
                    ctx.max_context_len,
                    scale=self.scale,
                )
            cache.write_decode(layer_idx, ctx.slots, ctx.slot_lengths, k_new, v_new)
            return cache.decode_attn(layer_idx, q, ctx.slots, ctx.slot_lengths, scale=self.scale)
        outs = []
        for b, (slot, n) in enumerate(zip(ctx.slots, ctx.slot_lengths)):
            cache.write_decode(layer_idx, [slot], [n], k_new[b : b + 1], v_new[b : b + 1])
            kb, vb = cache.read_dense(layer_idx, slot, n + 1, window=self.window_left)
            outs.append(fni8.attn_int8_decode(q[b : b + 1], kb, vb, scale=self.scale))
        return torch.cat(outs, dim=0)

    @staticmethod
    def _draft_attn(q, k, v, *, scale):
        """Cache-free single-step attention for MTP drafting.

        Computes attention without reading/writing any external KV cache.
        q, k, v are fp16 in [B, H, S, D] / [B, Hkv, S, D] layout (after
        transpose). Used by MTP heads during the draft phase so they never
        corrupt the main model's paged cache.
        """
        bs, H, S, D = q.shape
        Hkv = k.shape[1]
        if H != Hkv:
            rep = H // Hkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, S, S]
        attn = torch.softmax(scores, dim=-1)
        return (attn @ v).transpose(1, 2).reshape(bs, S, H * D)

    def _verify_batched(self, q, k, v, ctx, layer_idx):
        """Spec-decode verify: S = 1 + num_drafts tokens per batch row. Reads
        prefix K/V from paged cache, prepends prefix to the current forward's
        K/V (base + drafts), builds a contiguous int8 cache, and calls
        ``fni8.attn_int8_verify`` — once for the base token (prefix + itself)
        and once for the k drafts (prefix + base + preceding drafts).

        Also COMMITS every verify token's K/V to the paged store (at the
        positions in ``ctx.verify_slot_mapping``). This is load-bearing: the
        engine accepts the longest greedy-matching prefix and advances each
        sequence's length by ``n_acc``, but only the base token gets its K/V
        written by the preceding base decode forward. Without persisting the
        verify tokens here, every INTERMEDIATE accepted token (the first
        ``n_acc-1``) would leave a permanent hole in the paged cache and the
        next step would read stale/uninitialised K/V. We write all S positions
        unconditionally; positions past ``n_acc`` are never read (reads are
        gated by each row's committed length / ``context_lens``) and are
        overwritten by the next step, so writing the rejected tail is harmless.

        Returns the RAW attention output ``[B, H, S, D]`` (heads not merged, no
        ``o_proj``). The shared forward tail merges heads and applies ``o_proj``
        exactly once — returning an already-projected tensor here would apply
        ``o_proj`` twice and scramble the verify logits.
        """
        n_drafts = q.shape[1] - 1
        cache = ctx.kv_cache
        k_t = k.transpose(1, 2).contiguous()  # [B, Hkv, S, D]
        v_t = v.transpose(1, 2).contiguous()  # [B, Hkv, S, D]
        q_t = q.transpose(1, 2).contiguous()  # [B, H, S, D]

        B = k_t.shape[0]

        # Commit every verify token's K/V to the paged store (accepted-token KV).
        if ctx.verify_slot_mapping is not None:
            Hkv, S, D = k_t.shape[1:]
            k_flat = k_t.permute(0, 2, 1, 3).reshape(B * S, Hkv, D)  # token-major
            v_flat = v_t.permute(0, 2, 1, 3).reshape(B * S, Hkv, D)
            cache.write_decode_static(layer_idx, ctx.verify_slot_mapping, k_flat, v_flat)

        # The dedicated int8 verify kernel (`attn_int8_verify`, int8 dp4a PV) only
        # supports head dim in {32, 64, 128}. Qwen3.5's gated attention runs head
        # dim 256, so route those through the fp16-PV `attn_int8_fwd` path instead
        # (D in {32,64,72,80,128,256}) via the same causal zero-pad-Q trick the
        # chunked-prefill path uses: read the dequantized prefix, append this
        # forward's K/V, place the S verify queries at the sequence END, and take
        # the last S outputs. Correctness (not the exact bytes) is what verify
        # needs — the accepted tokens' K/V is re-committed canonically afterward.
        D = q_t.shape[-1]
        S = q_t.shape[2]
        if D not in (32, 64, 128):
            outs = []
            for b in range(B):
                prefix_len = ctx.slot_lengths[b]
                kb, vb = cache.read_dense(layer_idx, ctx.slots[b], prefix_len)  # [1,Hkv,P,D]
                k_all = torch.cat([kb, k_t[b:b + 1]], dim=2)  # [1,Hkv,P+S,D]
                v_all = torch.cat([vb, v_t[b:b + 1]], dim=2)
                total = prefix_len + S
                q_pad = k_all.new_zeros(1, q_t.shape[1], total, D)
                q_pad[:, :, -S:, :] = q_t[b:b + 1]  # actual queries at the end
                out_b = fni8.attn_int8_fwd(
                    q_pad, k_all, v_all, causal=True, scale=self.scale,
                )
                outs.append(out_b[:, :, -S:, :])  # [1,Hq,S,D]
            return torch.cat(outs, dim=0)  # [B, H, S, D]

        # Per-sequence verify attention to avoid zero-padding corruption in
        # ragged batches.  build_verify_cache pads every row to max_prefix with
        # zeros, but attn_int8_verify receives no per-row prefix-length mask, so
        # the zero entries leak into the softmax and corrupt attention for every
        # row whose prefix is shorter than the maximum.  Processing rows one at a
        # time guarantees pad_sz == 0 for every row (max_prefix == prefix_len).
        out_base_list, out_drafts_list = [], []
        for b in range(B):
            k_b, ks_b, v_b, vs_b = cache.build_verify_cache(
                layer_idx, [ctx.slots[b]], [ctx.slot_lengths[b]],
                k_t[b:b + 1, :, :1], v_t[b:b + 1, :, :1],
            )
            out_base_list.append(
                fni8.attn_int8_verify(
                    q_t[b:b + 1, :, :1], k_b, ks_b, v_b, vs_b, scale=self.scale,
                )
            )
            if n_drafts > 0:
                k_d, ks_d, v_d, vs_d = cache.build_verify_cache(
                    layer_idx, [ctx.slots[b]], [ctx.slot_lengths[b]],
                    k_t[b:b + 1], v_t[b:b + 1],
                )
                out_drafts_list.append(
                    fni8.attn_int8_verify(
                        q_t[b:b + 1, :, 1:], k_d, ks_d, v_d, vs_d, scale=self.scale,
                    )
                )
        out_base = torch.cat(out_base_list, dim=0)
        if n_drafts > 0:
            out_drafts = torch.cat(out_drafts_list, dim=0)
            out_t = torch.cat([out_base, out_drafts], dim=2)
        else:
            out_t = out_base

        return out_t  # [B, H, S, D]; heads merged + o_proj'd once by the forward tail
