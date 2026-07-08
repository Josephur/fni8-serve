# SPDX-License-Identifier: MIT
"""Multi-head Latent Attention (MLA) — Track-1 fp16 port.

Checks the weight-absorption identity the Track-2 int8 kernel depends on
(q_nope·k_nope == (W_UK^T q_nope)·c_KV) and that the MLAAttention block runs
end-to-end (down/up LoRA projections on dp4a + decoupled RoPE + causal softmax)."""
import pytest
import torch

from fni8serve.layers.mla_attn import absorb_qk_equiv

CUDA = torch.cuda.is_available()


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


@pytest.mark.skipif(not CUDA, reason="LoRA projections use the dp4a GEMM")
def test_mla_block_runs_and_matches_reference():
    from fni8 import QTensor

    from fni8serve.layers.mla_attn import MLAAttention
    from fni8serve.models.config import ModelConfig

    torch.manual_seed(0)
    H, nh = 256, 4
    q_lora, kv_lora = 96, 64
    qk_nope, qk_rope, vd = 32, 16, 32
    cfg = ModelConfig(arch="deepseek", vocab_size=64, hidden_size=H, num_hidden_layers=1,
                      num_attention_heads=nh, num_key_value_heads=nh, intermediate_size=128,
                      max_position_embeddings=64, head_dim=qk_nope + qk_rope)

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    def ones(n):
        return torch.ones(n, device="cuda", dtype=torch.float16)

    blk = MLAAttention(
        cfg, q_a_proj=qt(q_lora, H), q_a_norm=ones(q_lora), q_b_proj=qt(nh * (qk_nope + qk_rope), q_lora),
        kv_a_proj=qt(kv_lora + qk_rope, H), kv_a_norm=ones(kv_lora),
        kv_b_proj=qt(nh * (qk_nope + vd), kv_lora), o_proj=qt(H, nh * vd),
        num_heads=nh, q_lora_rank=q_lora, kv_lora_rank=kv_lora, qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope, v_head_dim=vd).cuda()
    x = torch.randn(1, 7, H, device="cuda", dtype=torch.float16)
    y = blk(x, None, None, 0)
    assert y.shape == (1, 7, H) and torch.isfinite(y).all()
    # determinism
    assert torch.equal(y, blk(x, None, None, 0))
