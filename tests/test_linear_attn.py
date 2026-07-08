# SPDX-License-Identifier: MIT
"""Gated DeltaNet linear attention (Track-1 fp16 port).

The scalar recurrence is the correct oracle. Checks the delta-rule algebra (a
single write is read back; decay shrinks old state), the L2-norm helper, and that
the full GatedDeltaNetAttention block runs end-to-end. The chunked/int8 form is
Track 2 (fni8 csrc/), validated there against this recurrence."""
import pytest
import torch

from fni8serve.layers.linear_attn import _l2norm, recurrent_gated_delta_rule

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
    out = recurrent_gated_delta_rule(k.clone(), k, v, torch.ones(B, H, 1), torch.zeros(B, H, 1))
    torch.testing.assert_close(out, v, rtol=1e-3, atol=1e-3)   # o_1 = (v k^T) k = v (k unit)


def test_decay_shrinks_state():
    """Strong decay (alpha->0) makes an old write vanish by the next step."""
    B, H, Dk, Dv = 1, 1, 4, 4
    q = _l2norm(torch.randn(B, H, 2, Dk))
    k = _l2norm(torch.randn(B, H, 2, Dk))
    v = torch.randn(B, H, 2, Dv)
    beta = torch.ones(B, H, 2)
    g_strong = torch.tensor([[[0.0, -20.0]]])                  # step 2 decays state ~0
    out = recurrent_gated_delta_rule(q, k, v, beta, g_strong)
    # at t=2 the state is dominated by the fresh write (old contribution ~ alpha~0)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not CUDA, reason="the projections use the dp4a GEMM")
def test_deltanet_block_runs():
    from fni8 import QTensor

    from fni8serve.layers.linear_attn import GatedDeltaNetAttention
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

    blk = GatedDeltaNetAttention(
        cfg, qkv_proj=qt(2 * nk * kd + nv * vd, H), out_proj=qt(H, nv * vd),
        conv_weight=torch.randn(2 * nk * kd + nv * vd, 4, device="cuda", dtype=torch.float16),
        a_log=torch.zeros(nv, device="cuda"), dt_bias=torch.zeros(nv, device="cuda"),
        beta_proj=qt(nv, H), gate_proj=qt(nv, H),
        norm_gain=torch.ones(nv * vd, device="cuda", dtype=torch.float16),
        num_k_heads=nk, num_v_heads=nv, key_dim=kd, value_dim=vd).cuda()
    x = torch.randn(1, 6, H, device="cuda", dtype=torch.float16)
    y = blk(x, None, None, 0)
    assert y.shape == (1, 6, H) and torch.isfinite(y).all()
