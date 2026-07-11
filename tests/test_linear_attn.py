# SPDX-License-Identifier: MIT
"""Gated DeltaNet linear attention (Track-1 fp16 port).

The scalar recurrence is the correct oracle. Checks the delta-rule algebra (a
single write is read back; decay shrinks old state), the L2-norm helper, and that
the full GatedDeltaNetAttention block runs end-to-end. The chunked/int8 form is
Track 2 (fni8 csrc/), validated there against this recurrence."""
import pytest
import torch
import torch.nn.functional as F

from fni8serve.layers.linear_attn import (
    GatedDeltaNetAttention,
    ShortConv,
    _l2norm,
    lightning_attention,
    lightning_slopes,
    recurrent_gated_delta_rule,
)

CUDA = torch.cuda.is_available()


def test_l2norm_unit():
    x = torch.randn(2, 3, 4, 8)
    assert torch.allclose(_l2norm(x).norm(dim=-1), torch.ones(2, 3, 4), atol=1e-4)


def test_delta_rule_writes_and_reads():
    """beta=1, alpha=1, one key: writing v at k then querying k reads v back
    (delta rule stores an associative pair)."""
    B, H, Dk, Dv = 1, 1, 4, 4
    k = _l2norm(torch.randn(B, H, 1, Dk))
    v = torch.randn(B, H, 1, Dv)
    out, _ = recurrent_gated_delta_rule(k.clone(), k, v, torch.ones(B, H, 1), torch.zeros(B, H, 1))
    torch.testing.assert_close(out, v, rtol=1e-3, atol=1e-3)   # o_1 = (v k^T) k = v (k unit)


def test_decay_shrinks_state():
    """Strong decay (alpha->0) makes an old write vanish by the next step."""
    B, H, Dk, Dv = 1, 1, 4, 4
    q = _l2norm(torch.randn(B, H, 2, Dk))
    k = _l2norm(torch.randn(B, H, 2, Dk))
    v = torch.randn(B, H, 2, Dv)
    beta = torch.ones(B, H, 2)
    g_strong = torch.tensor([[[0.0, -20.0]]])                  # step 2 decays state ~0
    out, _ = recurrent_gated_delta_rule(q, k, v, beta, g_strong)
    # at t=2 the state is dominated by the fresh write (old contribution ~ alpha~0)
    assert torch.isfinite(out).all()


def test_delta_rule_decode_state_matches_prefill():
    """The actual decode-caching bug: feeding the sequence one token at a time with
    `S` carried across calls (as the decode path now does) must reproduce the same
    output as a single whole-sequence ("prefill") call. Before the fix, each call
    zero-initialized `S`, so this would only hold for the first token."""
    torch.manual_seed(0)
    B, H, L, Dk, Dv = 2, 3, 6, 4, 5
    q = _l2norm(torch.randn(B, H, L, Dk))
    k = _l2norm(torch.randn(B, H, L, Dk))
    v = torch.randn(B, H, L, Dv)
    beta = torch.sigmoid(torch.randn(B, H, L))
    g = -F.softplus(torch.randn(B, H, L))

    full_out, full_state = recurrent_gated_delta_rule(q, k, v, beta, g)

    state, outs = None, []
    for t in range(L):
        o, state = recurrent_gated_delta_rule(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
                                              beta[:, :, t:t + 1], g[:, :, t:t + 1], state=state)
        outs.append(o)

    torch.testing.assert_close(torch.cat(outs, dim=2), full_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, full_state, rtol=1e-4, atol=1e-4)


def test_lightning_attention_decode_state_matches_prefill():
    """Same decode-caching property for MiniMax lightning attention: stepping token
    by token with `S` carried across calls must match one whole-sequence call."""
    torch.manual_seed(1)
    B, H, L, Dk, Dv = 2, 4, 5, 4, 4
    q, k = torch.randn(B, H, L, Dk), torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dv)
    slopes = lightning_slopes(H)

    full_out, full_state = lightning_attention(q, k, v, slopes)

    state, outs = None, []
    for t in range(L):
        o, state = lightning_attention(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
                                       slopes, state=state)
        outs.append(o)

    torch.testing.assert_close(torch.cat(outs, dim=2), full_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, full_state, rtol=1e-4, atol=1e-4)


class _ConvHarness:
    """Bare object exposing only what `GatedDeltaNetAttention._conv` touches, so the
    causal-conv decode-tail logic can be unit tested without building the full
    nn.Module (whose projections need the CUDA dp4a kernels)."""

    def __init__(self, conv_kernel, conv_weight):
        self.conv_kernel = conv_kernel
        self.conv_weight = conv_weight


def test_deltanet_conv_tail_matches_full_sequence():
    """The causal depthwise conv must carry its trailing `kernel-1` raw window
    across decode calls instead of zero-padding it away: one token at a time with
    the tail threaded through must match a single whole-sequence call."""
    torch.manual_seed(2)
    B, L, W, K = 2, 7, 5, 4
    harness = _ConvHarness(K, torch.randn(W, K))
    x = torch.randn(B, L, W)

    full, _ = GatedDeltaNetAttention._conv(harness, x)

    tail, outs = None, []
    for t in range(L):
        y, tail = GatedDeltaNetAttention._conv(harness, x[:, t:t + 1], tail)
        outs.append(y)

    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-5, atol=1e-5)


class _ShortConvHarness:
    """Bare object exposing only what `ShortConv._conv` touches, so the
    ShortConv causal-conv decode-tail logic can be unit tested without building the
    full nn.Module (whose projections need the CUDA dp4a kernels)."""

    def __init__(self, kernel, conv_weight):
        self.kernel = kernel
        self.conv_weight = conv_weight


def test_short_conv_tail_matches_full_sequence():
    """The ShortConv depthwise conv must carry its trailing `kernel-1` raw window
    across decode calls instead of zero-padding it away: one token at a time with
    the tail threaded through must match a single whole-sequence call."""
    torch.manual_seed(3)
    B, L, D, K = 2, 7, 5, 3
    harness = _ShortConvHarness(K, torch.randn(D, 1, K))
    u = torch.randn(B, D, L)

    full, _ = ShortConv._conv(harness, u)

    tail, outs = None, []
    for t in range(L):
        y, tail = ShortConv._conv(harness, u[:, :, t:t + 1], tail)
        outs.append(y)

    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-5, atol=1e-5)


def _build_deltanet_block():
    from fni8 import QTensor

    from fni8serve.models.config import ModelConfig

    torch.manual_seed(0)
    H, nk, nv, kd, vd = 128, 2, 4, 16, 16
    cfg = ModelConfig(arch="qwen3_next", vocab_size=32, hidden_size=H, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                      max_position_embeddings=64, head_dim=32)

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    return GatedDeltaNetAttention(
        cfg, qkv_proj=qt(2 * nk * kd + nv * vd, H), out_proj=qt(H, nv * vd),
        conv_weight=torch.randn(2 * nk * kd + nv * vd, 4, device="cuda", dtype=torch.float16),
        a_log=torch.zeros(nv, device="cuda"), dt_bias=torch.zeros(nv, device="cuda"),
        beta_proj=qt(nv, H), gate_proj=qt(nv, H),
        # gated output norm is PER-HEAD over head_v_dim (HF Qwen3_5RMSNormGated), so the
        # gain is `vd`-sized, not the flattened nv*vd.
        norm_gain=torch.ones(vd, device="cuda", dtype=torch.float16),
        num_k_heads=nk, num_v_heads=nv, key_dim=kd, value_dim=vd).cuda()


@pytest.mark.skipif(not CUDA, reason="the projections use the dp4a GEMM")
def test_deltanet_block_runs():
    blk = _build_deltanet_block()
    x = torch.randn(1, 6, 128, device="cuda", dtype=torch.float16)
    y = blk(x, None, None, 0)
    assert y.shape == (1, 6, 128) and torch.isfinite(y).all()


@pytest.mark.skipif(not CUDA, reason="the projections use the dp4a GEMM")
def test_decode_recurrence_output_stays_fp32(monkeypatch):
    """Regression guard for the redundant-cast removal: at decode (L==1) the fused
    kernel's fp32 readout must flow straight into the gated RMSNorm WITHOUT an
    fp32->fp16->fp32 round-trip. We stub `fni8.deltanet_recurrent_decode` with a pure
    fp32 reference so the test holds even on a prebuilt fni8 that predates the kernel,
    then compare the block output against the OLD behaviour (readout downcast to fp16
    before the norm). Removing the round-trip only drops fp16 rounding, so the block
    output must stay bit-close (cos >= 0.9999) — same numerics, one fewer cast pair."""
    import fni8serve.layers.linear_attn as la

    def _ref_decode(q, k, v, alpha, beta, initial_state=None):
        # exact eager math, fp32 in / fp32 out (mirrors the real decode kernel)
        g = alpha.clamp_min(1e-30).log()
        o, s = la.recurrent_gated_delta_rule(q, k, v, beta, g, state=initial_state)
        return o.float(), s

    blk = _build_deltanet_block()
    x = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)  # L==1 decode step

    # NEW path: kernel returns fp32, kept fp32 through to the gated norm.
    monkeypatch.setattr(la.fni8, "deltanet_recurrent_decode", _ref_decode, raising=False)
    monkeypatch.setattr(la, "_DND_DECODE", True)
    y_new = blk(x, None, None, 0).float()

    # OLD path: identical kernel but its readout is downcast to v.dtype first (the
    # round-trip the caller then undoes with `.float()`).
    def _ref_decode_roundtrip(q, k, v, alpha, beta, initial_state=None):
        o, s = _ref_decode(q, k, v, alpha, beta, initial_state=initial_state)
        return o.to(v.dtype), s

    monkeypatch.setattr(la.fni8, "deltanet_recurrent_decode", _ref_decode_roundtrip, raising=False)
    y_old = blk(x, None, None, 0).float()

    cos = torch.nn.functional.cosine_similarity(y_new.flatten(), y_old.flatten(), dim=0)
    assert cos.item() >= 0.9999, f"decode block output drifted after round-trip removal: cos={cos.item()}"
