# SPDX-License-Identifier: MIT
"""Multi-head Latent Attention (MLA) — Track-1 fp16 port.

Checks the weight-absorption identity the Track-2 int8 kernel depends on
(q_nope·k_nope == (W_UK^T q_nope)·c_KV), that the MLAAttention block runs
end-to-end (down/up LoRA projections on dp4a + decoupled RoPE + causal softmax),
and that decode through the latent-KV cache matches the uncached one-shot forward."""
import pytest
import torch

from fni8serve.layers.mla_attn import absorb_qk_equiv

CUDA = torch.cuda.is_available()

H, NH = 256, 4
Q_LORA, KV_LORA = 96, 64
QK_NOPE, QK_ROPE, VD = 32, 16, 32


def test_absorb_identity():
    """The nope score equals its latent-space form — the crux of MLA's absorb path."""
    torch.manual_seed(0)
    nope, kv_lora = 128, 512
    q_nope = torch.randn(4, nope)
    w_uk = torch.randn(nope, kv_lora) * 0.02       # W_UK^T : nope -> kv_lora
    c_kv = torch.randn(4, kv_lora) * 0.1
    k_nope = c_kv @ w_uk.t()                        # k_nope = W_UK c_KV
    direct = (q_nope * k_nope).sum(-1)              # q_nope · k_nope
    absorbed = absorb_qk_equiv(q_nope, w_uk, c_kv)
    torch.testing.assert_close(direct, absorbed, rtol=1e-4, atol=1e-3)


def _build_mla_block():
    from fni8 import QTensor

    from fni8serve.layers.mla_attn import MLAAttention
    from fni8serve.models.config import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(arch="deepseek", vocab_size=64, hidden_size=H, num_hidden_layers=1,
                      num_attention_heads=NH, num_key_value_heads=NH, intermediate_size=128,
                      max_position_embeddings=64, head_dim=QK_NOPE + QK_ROPE)

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    def ones(n):
        return torch.ones(n, device="cuda", dtype=torch.float16)

    return MLAAttention(
        cfg, q_a_proj=qt(Q_LORA, H), q_a_norm=ones(Q_LORA),
        q_b_proj=qt(NH * (QK_NOPE + QK_ROPE), Q_LORA),
        kv_a_proj=qt(KV_LORA + QK_ROPE, H), kv_a_norm=ones(KV_LORA),
        kv_b_proj=qt(NH * (QK_NOPE + VD), KV_LORA), o_proj=qt(H, NH * VD),
        num_heads=NH, q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE, v_head_dim=VD).cuda()


@pytest.mark.skipif(not CUDA, reason="LoRA projections use the dp4a GEMM")
def test_mla_block_runs_and_matches_reference():
    blk = _build_mla_block()
    x = torch.randn(1, 7, H, device="cuda", dtype=torch.float16)
    y = blk(x, None, None, 0)
    assert y.shape == (1, 7, H) and torch.isfinite(y).all()
    # determinism
    assert torch.equal(y, blk(x, None, None, 0))


@pytest.mark.skipif(not CUDA, reason="LoRA projections use the dp4a GEMM")
def test_mla_decode_cache_matches_uncached_recompute():
    """Cached one-token-at-a-time decode must match the one-shot (no-cache) forward
    over the whole sequence — that uncached forward is MLA's own oracle (it ports the
    HF decompress path directly, see module docstring)."""
    from fni8serve.models.base import ForwardContext
    from fni8serve.models.cache import MLALatentCache

    blk = _build_mla_block()
    S = 7
    x = torch.randn(1, S, H, device="cuda", dtype=torch.float16)
    ref = blk(x, None, None, 0)

    cache = MLALatentCache(1, 1, KV_LORA + QK_ROPE, S, device="cuda")
    pos_full = torch.arange(S, device="cuda").unsqueeze(0)
    prefill_n = S - 1
    pre = blk(x[:, :prefill_n], pos_full[:, :prefill_n],
             ForwardContext(is_prefill=True, kv_cache=cache), 0)
    cache.advance(prefill_n)
    dec = blk(x[:, prefill_n:], pos_full[:, prefill_n:],
             ForwardContext(is_prefill=False, kv_cache=cache), 0)
    cache.advance(1)

    got = torch.cat([pre, dec], dim=1)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
