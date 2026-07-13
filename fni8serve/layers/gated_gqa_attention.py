# SPDX-License-Identifier: MIT
"""GatedGQAAttention — full softmax attention with an output gate (Qwen3-Next /
Qwen3.5 `attn_output_gate`).

Identical to the shared `GQAAttention` full-attention block, with ONE addition: the
query projection also emits a per-(head, head_dim) gate. Following the HF
`Qwen3NextAttention` reference, `q_proj` outputs `num_heads * head_dim * 2`; the
last dim is reshaped to `[.., num_heads, 2*head_dim]` and `chunk(2, -1)`-split into
the query and the gate. QK-norm (pre-RoPE) applies to the QUERY only, RoPE to the
query; the gate is untouched until after attention, where the output is elementwise
multiplied by `sigmoid(gate)` BEFORE `o_proj`.

This lives in a separate module (not folded into `GQAAttention`) so the gate feature
is additive and does not perturb the shared block that every non-gated family uses.
It implements the two paths the standalone `ModelRunner` drives — non-varlen prefill
(`attn_int8_fwd`) and simple single-sequence decode (`attn_int8_decode`) — plus
varlen prefill for the engine's packed prefill. The engine's paged/continuous-batch
*decode* path (slot_mapping / slot_lengths) reuses `GQAAttention._decode_batched`
(the shared paged-decode kernels) on the query half and applies the sigmoid output
gate via the same `_gate_and_project` the standalone paths use — so gated Qwen3.5
decodes under the continuous-batching engine, not just the standalone ModelRunner.

Softmax/LSE stay fp32 inside the fni8 kernels; only the gate multiply is fp16, on
the healthy half2 CUDA-core pipe (never the dead Volta tensor cores).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import fni8
from fni8 import QTensor

from .gqa_attention import GQAAttention
from .linear import LinearW8A8
from .norm import RMSNorm
from .rotary import RotaryEmbedding


class GatedGQAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_gate_proj: QTensor,  # merged [2*nh*hd | nkv*hd | nkv*hd]
        o_proj: QTensor,
        scale: float,
        rope: RotaryEmbedding,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
        rms_norm_eps: float = 1e-6,
        qk_unit_offset: bool = False,
        window_left: int = -1,
        causal: bool = True,
    ):
        super().__init__()
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, head_dim
        self.scale = scale
        self.window_left = window_left
        self.causal = causal
        self.qkv_gate_proj = LinearW8A8(qkv_gate_proj)
        self.o_proj = LinearW8A8(o_proj)
        self.rope = rope
        self.q_norm = (
            RMSNorm(head_dim, rms_norm_eps, q_norm, add_unit_offset=qk_unit_offset)
            if q_norm is not None else None
        )
        self.k_norm = (
            RMSNorm(head_dim, rms_norm_eps, k_norm, add_unit_offset=qk_unit_offset)
            if k_norm is not None else None
        )

    def _project(self, x):
        """x: [B, S, hidden] -> (q, gate, k, v) each head-shaped, plus applies QK-norm
        (pre-RoPE) to q/k. q/gate: [B,S,nh,hd]; k/v: [B,S,nkv,hd]. gate is RAW (no
        norm, no RoPE)."""
        B, S, _ = x.shape
        qkvg = self.qkv_gate_proj(x)
        qg, k, v = qkvg.split(
            [2 * self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1
        )
        qg = qg.view(B, S, self.nh, 2 * self.hd)
        q, gate = qg[..., : self.hd], qg[..., self.hd :]  # chunk(2, -1): query | gate
        q = q.contiguous()
        k = k.view(B, S, self.nkv, self.hd)
        v = v.view(B, S, self.nkv, self.hd)
        if self.q_norm is not None:  # per-head RMSNorm, pre-RoPE, QUERY only (+ K)
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, gate, k, v

    def forward(self, x, positions, ctx, layer_idx: int) -> torch.Tensor:
        B, S, _ = x.shape
        q, gate, k, v = self._project(x)
        q, k = self.rope(positions, q, k)

        if ctx.is_prefill:
            if ctx.cu_seqlens is not None:
                q_v = q.reshape(B * S, self.nh, self.hd)
                k_v = k.reshape(B * S, self.nkv, self.hd)
                v_v = v.reshape(B * S, self.nkv, self.hd)
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
                out = out_v.reshape(B, S, self.nh, self.hd)  # token-major [B,S,H,D]
                return self._gate_and_project(out, gate, B, S)
            # Non-varlen prefill: kernels want [B, H, S, D].
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            vt = v.transpose(1, 2).contiguous()
            slot = ctx.slots[0] if ctx.slots is not None else None
            ctx.kv_cache.write_prefill(
                layer_idx, kt, vt, slot=slot, start=ctx.prefill_start, positions=positions,
            )
            # Chunked prefill: use accumulated fp16 K/V + current K/V with
            # attn_int8_fwd (same kernel as full prefill) by zero-padding Q.
            acc_buf = getattr(ctx, "acc_kv_buffer", None)
            if acc_buf is not None and layer_idx < len(acc_buf) and acc_buf[layer_idx] is not None:
                k_prev, v_prev = acc_buf[layer_idx]
                prev_len = k_prev.shape[2]
                cur_len = kt.shape[2]
                kt_all = torch.cat([k_prev, kt], dim=2)
                vt_all = torch.cat([v_prev, vt], dim=2)
                total_len = kt_all.shape[2]
                q_pad = kt_all.new_zeros(1, self.nh, total_len, self.hd)
                q_pad[:, :, -cur_len:, :] = qt  # actual Q at the end
                out = fni8.attn_int8_fwd(
                    q_pad, kt_all, vt_all,
                    causal=self.causal, scale=self.scale, window_left=self.window_left,
                )
                out = out[:, :, -cur_len:, :]
                acc_buf[layer_idx] = (kt_all.contiguous(), vt_all.contiguous())
            else:
                out = fni8.attn_int8_fwd(
                    qt, kt, vt, causal=self.causal, scale=self.scale, window_left=self.window_left
                )
            # Initialize or update accumulated buffer
            if acc_buf is not None and not (layer_idx < len(acc_buf) and acc_buf[layer_idx] is not None):
                while len(acc_buf) <= layer_idx:
                    acc_buf.append(None)
                acc_buf[layer_idx] = (kt.contiguous(), vt.contiguous())
            out = out.transpose(1, 2)  # [B,H,S,D] -> [B,S,H,D]
            return self._gate_and_project(out, gate, B, S)

        if getattr(ctx, "is_verify", False):
            # Spec-decode verify (S = 1 + num_drafts tokens per row). Reuse the shared
            # ``GQAAttention._verify_batched`` as an unbound method exactly like the
            # decode path below borrows ``_decode_batched`` — it only touches
            # attributes both blocks share (``scale``) and ``ctx``/``cache``, and runs
            # on the QUERY half (``q``); the GATE is held aside and applied after. It
            # returns RAW ``[B, H, S, D]`` (heads not merged, no o_proj); transpose to
            # token-major ``[B, S, nh, hd]`` so the SAME ``_gate_and_project`` the
            # standalone/decode paths use applies the sigmoid output gate PER verify
            # token before the single o_proj. Committing every verify token's K/V
            # (accepted-token KV, #235) is done inside ``_verify_batched``.
            out = GQAAttention._verify_batched(self, q, k, v, ctx, layer_idx)
            out = out.transpose(1, 2)  # [B,H,S,D] -> [B,S,nh,hd]
            return self._gate_and_project(out, gate, B, S)

        if ctx.slot_lengths is not None or ctx.slot_mapping is not None:
            # Engine paged/continuous-batch decode. Reuse the shared batched-decode
            # kernels verbatim (paged int8 write + one `attn_paged_decode_cached`
            # launch, or the CUDA-graph static path) by borrowing GQAAttention's
            # `_decode_batched` as an unbound method — it only touches attributes
            # GatedGQAAttention shares (`scale`, `window_left`) and `ctx.kv_cache`,
            # and runs on the QUERY half (q here is the query, gate is held aside).
            # It returns [B, H, 1, D]; transpose to token-major [B, 1, H, D] so the
            # SAME `_gate_and_project` the standalone paths use applies the sigmoid
            # output gate before o_proj — identical gate math to the ModelRunner path.
            out = GQAAttention._decode_batched(self, q, k, v, ctx, layer_idx)
            out = out.transpose(1, 2)  # [B,H,1,D] -> [B,1,H,D] = [B,S,nh,hd]
            return self._gate_and_project(out, gate, B, S)

        # Simple single-sequence decode: kernels want [B, H, 1, D].
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
        k_all, v_all = ctx.kv_cache.append_decode(layer_idx, kt, vt)
        k_all, v_all = self._window(k_all, v_all)
        out = fni8.attn_int8_decode(qt, k_all, v_all, scale=self.scale)
        out = out.transpose(1, 2)  # [B,H,1,D] -> [B,1,H,D]
        return self._gate_and_project(out, gate, B, S)

    def _gate_and_project(self, out, gate, B, S):
        """out, gate: [B, S, nh, hd]. Apply sigmoid gate, merge heads, o_proj."""
        out = out * torch.sigmoid(gate.to(out.dtype))
        out = out.reshape(B, S, self.nh * self.hd)
        return self.o_proj(out)

    def _window(self, k_all, v_all):
        if self.window_left >= 0 and k_all.shape[2] > self.window_left:
            k_all = k_all[:, :, -self.window_left :].contiguous()
            v_all = v_all[:, :, -self.window_left :].contiguous()
        return k_all, v_all
