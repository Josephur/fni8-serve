# SPDX-License-Identifier: MIT
"""Gated DeltaNet linear attention (Qwen3-Next / Qwen3.5/3.6 backbone, MiniMax
lightning) — Track-1 fp16 port.

Recurrence (Gated DeltaNet, Yang et al. ICLR'25), per head:

    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

with q,k L2-normalized per head, a short causal depthwise conv1d(k=4)+SiLU token
shift on q/k/v, a scalar decay gate alpha_t = exp(-softplus(dt)*exp(A_log)) and a
write gate beta_t = sigmoid(b_t), then a gated RMSNorm on the output. The linear
projections run on the fni8 dp4a GEMM (LinearW8A8); only the recurrence stays fp16.

This module is the CORRECT fp path (the numeric oracle). It uses the O(L) scalar
recurrence — unambiguous and exact. The chunked WY/UT parallel form + its int8
dp4a acceleration are Track 2 (fni8 csrc/): the chunk form needs the log-space
stabilization real kernels use, so it lives with that kernel, not here.
"""

from __future__ import annotations

import os

import fni8
import torch
import torch.nn as nn
import torch.nn.functional as F

from fni8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def recurrent_gated_delta_rule(q, k, v, beta, g, state=None):
    """Gated delta rule, scalar form. q,k: [B,H,L,Dk] (pre-L2-normed), v: [B,H,L,Dv],
    beta: [B,H,L] in (0,1], g: [B,H,L] = log(alpha) (<=0). `state`: optional
    [B,H,Dv,Dk] fp32 initial S (carried in from a prior call for decode); defaults
    to zero (prefill / stateless). Returns (o [B,H,L,Dv], state_final [B,H,Dv,Dk])
    so a caller can carry `S` across chunked/decode calls instead of losing it."""
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    S = (
        state
        if state is not None
        else torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=q.device)
    )
    alpha = g.exp().float()
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    out = torch.empty(B, H, L, Dv, dtype=torch.float32, device=q.device)
    for t in range(L):
        kt, vt, qt = kf[:, :, t], vf[:, :, t], qf[:, :, t]  # [B,H,D]
        at = alpha[:, :, t][..., None, None]  # [B,H,1,1]
        bt = bf[:, :, t][..., None]  # [B,H,1]
        Sk = torch.einsum("bhvk,bhk->bhv", S, kt)  # S_{t-1} k_t
        erase = at * (bt * Sk)[..., None] * kt[..., None, :]
        write = (bt * vt)[..., None] * kt[..., None, :]
        S = at * S - erase + write
        out[:, :, t] = torch.einsum("bhvk,bhk->bhv", S, qt)  # o_t = S_t q_t
    return out.to(v.dtype), S


# Kill-switch (default on); auto-degrades to the eager reference if the installed
# fni8 predates the decode kernel, so this stays correct on an older prebuilt fni8.
_DND_DECODE = os.environ.get("FNI8_DND_DECODE", "1") != "0" and hasattr(
    fni8, "deltanet_recurrent_decode"
)


def _gated_delta_rule(q, k, v, beta, g, state, L):
    """Dispatch the gated delta rule. For the single-token decode step (L==1, CUDA,
    Dk/Dv<=128) use the fused `fni8.deltanet_recurrent_decode` kernel: same math as
    the eager reference above (validated bit-close against it, cos 1.0), but it
    collapses the ~5 tiny eager ops into ONE launch AND is CUDA-graph-capturable
    (register state, no cudaFuncSetAttribute), so it replays inside the engine's
    graphed decode instead of forcing an eager fallback. Eager reference otherwise
    (prefill L>1, CPU, head dims > 128, or FNI8_DND_DECODE=0). The kernel takes
    alpha=exp(g)."""
    if _DND_DECODE and L == 1 and q.is_cuda and q.shape[-1] <= 128 and v.shape[-1] <= 128:
        o, state = fni8.deltanet_recurrent_decode(
            q.float(), k.float(), v.float(), g.exp().float(), beta.float(), initial_state=state
        )
        return o.to(v.dtype), state
    return recurrent_gated_delta_rule(q, k, v, beta, g, state=state)


class GatedDeltaNetAttention(nn.Module):
    """Linear-attention block. Projections on dp4a; recurrence fp16. GQA-style:
    `num_v_heads` value/output heads, `num_k_heads` query/key heads (broadcast).

    Weights: qkv_proj (merged Q|K|V), beta_proj + a/dt gate params (A_log, dt_bias),
    conv weight [width, kernel], out_proj, and the gated output RMSNorm gain.
    """

    is_recurrent = True  # carries per-slot decode state via ctx.lin_cache

    def __init__(
        self,
        cfg,
        *,
        qkv_proj: QTensor,
        out_proj: QTensor,
        conv_weight,
        a_log,
        dt_bias,
        beta_proj,
        gate_proj,
        norm_gain,
        num_k_heads,
        num_v_heads,
        key_dim,
        value_dim,
        conv_kernel=4,
        z_proj: QTensor | None = None,
    ):
        super().__init__()
        self.nk, self.nv = num_k_heads, num_v_heads
        self.kd, self.vd = key_dim, value_dim
        self.conv_kernel = conv_kernel
        self.qkv_proj = LinearW8A8(qkv_proj)
        self.out_proj = LinearW8A8(out_proj)
        self.beta_proj = LinearW8A8(beta_proj)
        self.gate_proj = LinearW8A8(gate_proj)
        if z_proj is not None:
            self.z_proj = LinearW8A8(z_proj)
        self.register_buffer("conv_weight", conv_weight, persistent=False)  # [Wc, K]
        self.A_log = nn.Parameter(a_log)
        self.dt_bias = nn.Parameter(dt_bias)
        self.norm = RMSNorm(value_dim, cfg.rms_norm_eps, norm_gain)

    def _conv(self, x, tail=None):
        """Causal depthwise conv1d(k) + SiLU. x: [B, L, Wc]. `tail`: optional
        [B, K-1, Wc] trailing raw (pre-conv) window from the previous call — carries
        decode history in instead of zero-padding, which would forget it every step.
        Returns (activated [B, L, Wc], new_tail [B, K-1, Wc])."""
        B, L, W = x.shape
        K = self.conv_kernel
        if tail is None:
            tail = x.new_zeros(B, K - 1, W)
        xt = torch.cat([tail, x], dim=1)  # [B,K-1+L,Wc]
        new_tail = xt[:, -(K - 1) :] if K > 1 else x.new_zeros(B, 0, W)
        xt = F.conv1d(xt.transpose(1, 2), self.conv_weight.unsqueeze(1), groups=W)
        return F.silu(xt.transpose(1, 2)), new_tail

    def forward(self, hidden, positions, ctx, layer_idx):
        B, L, _ = hidden.shape
        cache = ctx.lin_cache if ctx is not None else None
        conv_tail = cache.get_conv_tail(layer_idx) if cache is not None else None
        qkv, conv_tail = self._conv(self.qkv_proj(hidden), conv_tail)
        if cache is not None:
            cache.set_conv_tail(layer_idx, conv_tail)
        qk = self.nk * self.kd
        q, k, v = qkv.split([qk, qk, self.nv * self.vd], dim=-1)
        # HF gated-delta-rule scales the (l2-normed) query by 1/sqrt(head_k_dim) before
        # the readout (`query = query * scale`). Applied here (not inside the shared
        # recurrence, which stays a pure delta rule) since it is a per-model readout
        # scale. Omitting it inflates the pre-norm output ~sqrt(Dk)x and — via the gated
        # RMSNorm eps — rotates the normed output (cos ~0.84 vs HF instead of 1.0).
        q = (_l2norm(q.view(B, L, self.nk, self.kd)) * (self.kd**-0.5)).transpose(1, 2)
        k = _l2norm(k.view(B, L, self.nk, self.kd)).transpose(1, 2)
        v = v.view(B, L, self.nv, self.vd).transpose(1, 2)
        # broadcast k/q heads to value heads (GQA)
        rep = self.nv // self.nk
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
        beta = torch.sigmoid(self.beta_proj(hidden)).transpose(1, 2)  # [B,nv,L]...
        beta = beta.reshape(B, self.nv, L) if beta.dim() == 3 else beta
        dt = self.gate_proj(hidden).transpose(1, 2)
        g = -F.softplus(dt.float() + self.dt_bias.view(1, -1, 1)) * self.A_log.exp().view(1, -1, 1)
        state = cache.get_state(layer_idx) if cache is not None else None
        o, state = _gated_delta_rule(
            q, k, v, beta.reshape(B, self.nv, L), g.reshape(B, self.nv, L), state, L
        )
        if cache is not None:
            cache.set_state(layer_idx, state)
        # HF Qwen3_5RMSNormGated ("norm BEFORE gate"): a PER-HEAD RMS over head_v_dim,
        # then the per-head gain, THEN the z gate (silu) — in that order. Done in fp32
        # (the gated norm is numerically load-bearing, never quantized). The gain is
        # per-head (length vd), so normalize over the last (vd) axis, not nv*vd.
        o = o.transpose(1, 2).float()  # [B, L, nv, vd]
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm.eps)
        o = o * self.norm.weight.float()
        if hasattr(self, "z_proj"):
            gate = self.z_proj(hidden).view(B, L, self.nv, self.vd).float()
            o = o * F.silu(gate)
        o = o.reshape(B, L, self.nv * self.vd)
        return self.out_proj(o.to(hidden.dtype))


def lightning_attention(q, k, v, slopes, state=None):
    """MiniMax lightning attention (TransNormer): data-INDEPENDENT fixed decay, no
    delta correction. S_t = ratio_h S_{t-1} + k_t^T v_t ; o_t = q_t S_t, with per-head
    ratio = exp(-slope). q,k: [B,H,L,Dk], v: [B,H,L,Dv], slopes: [H]. Scalar oracle.
    `state`: optional [B,H,Dv,Dk] fp32 initial S (carried in from a prior call for
    decode); defaults to zero. Returns (o [B,H,L,Dv], state_final [B,H,Dv,Dk])."""
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    S = (
        state
        if state is not None
        else torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=q.device)
    )
    ratio = torch.exp(-slopes.float()).view(1, H, 1, 1)
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(B, H, L, Dv, dtype=torch.float32, device=q.device)
    for t in range(L):
        S = ratio * S + vf[:, :, t][..., :, None] * kf[:, :, t][..., None, :]  # k^T v outer
        out[:, :, t] = torch.einsum("bhvk,bhk->bhv", S, qf[:, :, t])
    return out.to(v.dtype), S


def lightning_slopes(num_heads: int, device="cpu") -> torch.Tensor:
    """ALiBi-style per-head decay slopes: 2^(-8*(h+1)/H)."""
    h = torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    return torch.pow(2.0, -8.0 * h / num_heads)


class ShortConv(nn.Module):
    """LFM2 double-gated causal depthwise short conv (LIV): out = out_proj(C * conv(B*x)),
    with (B,C,x) = in_proj(h).chunk(3). in_proj/out_proj on dp4a; conv is depthwise k=3."""

    is_recurrent = True  # carries per-slot decode conv-tail via ctx.lin_cache

    def __init__(self, dim: int, *, in_proj: QTensor, out_proj: QTensor, conv_weight, kernel=3):
        super().__init__()
        self.dim = dim
        self.kernel = kernel
        self.in_proj = LinearW8A8(in_proj)
        self.out_proj = LinearW8A8(out_proj)
        self.register_buffer("conv_weight", conv_weight, persistent=False)  # [dim, 1, k]

    def _conv(self, u, tail=None):
        """Causal depthwise conv1d(k) over the last dimension of `u` [B,D,L]. `tail`:
        optional [B,D,K-1] trailing window from the previous call — carries decode
        history in instead of zero-padding. Returns (activated [B,L,D], new_tail
        [B,D,K-1])."""
        B, D, L = u.shape
        K = self.kernel
        if tail is None:
            tail = u.new_zeros(B, D, K - 1)
        u_ext = torch.cat([tail, u], dim=-1)  # [B,D,K-1+L]
        new_tail = u_ext[:, :, -(K - 1) :] if K > 1 else u.new_zeros(B, D, 0)
        y = F.conv1d(u_ext, self.conv_weight, groups=D).transpose(1, 2)
        return y, new_tail

    def forward(self, x, positions=None, ctx=None, layer_idx=0):
        B, L, _ = x.shape
        bcx = self.in_proj(x)  # [B,L,3D]
        Bg, Cg, xg = bcx.chunk(3, dim=-1)
        u = (Bg * xg).transpose(1, 2)  # [B,D,L]
        cache = ctx.lin_cache if ctx is not None else None
        tail = cache.get_conv_tail(layer_idx) if cache is not None else None
        y, new_tail = self._conv(u, tail)
        if cache is not None:
            cache.set_conv_tail(layer_idx, new_tail)
        return self.out_proj(Cg * y)
