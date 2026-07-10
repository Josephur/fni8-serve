# SPDX-License-Identifier: MIT
"""Engine tests: continuous batching over concurrent requests of DIFFERENT prompt
lengths through a small random Qwen3, plus determinism vs the single-sequence
runner. Proves the scheduler + slot cache + ragged decode produce correct, stable
output. Needs CUDA."""

import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.engine import LLMEngine, SamplingParams
from fni8serve.models import ModelConfig, ModelRunner, build_model

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="engine needs the CUDA fni8 kernels")


def _cfg():
    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
    )


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        sd[f"{p}.self_attn.q_proj.weight"] = r(nh * hd, H)
        sd[f"{p}.self_attn.k_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.v_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.o_proj.weight"] = r(H, nh * hd)
        sd[f"{p}.self_attn.q_norm.weight"] = r(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = r(hd)
        sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_engine_generates_for_concurrent_requests():
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11, 12, 13]]  # different lengths
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6))
    assert len(outs) == 4
    for o in outs:
        assert len(o) == 6 and all(0 <= t < cfg.vocab_size for t in o)


def test_engine_matches_single_sequence_runner():
    """Greedy engine output for one request must closely track the standalone
    runner. Not bit-exact any more: the engine's decode KV cache is now paged
    int8 (quantize-on-write), while the standalone runner stays fp16
    contiguous, so a token can occasionally flip on a near-tied logit -- see
    `tests/test_paged_engine.py` for the direct cosine-similarity check of the
    two paths' logits."""
    torch.manual_seed(1)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt = [3, 1, 4, 1, 5]

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    eng_out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))[0]

    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref = runner.generate_greedy(torch.tensor([prompt], device="cuda"), max_new_tokens=5)[
        0
    ].tolist()

    assert eng_out[0] == ref[0]
    agree = sum(1 for a, b in zip(eng_out, ref) if a == b) / len(ref)
    assert agree >= 0.8, f"paged vs contiguous greedy diverged too much: {eng_out} vs {ref}"


def test_engine_respects_eos():
    torch.manual_seed(2)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64, eos_id=None)
    out = eng.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=4))[0]
    assert len(out) == 4


def test_engine_applies_logit_processors():
    """A per-request logit processor can force a specific token every step, proving
    the hook (issue #38) threads from SamplingParams through the engine's sampler."""
    torch.manual_seed(3)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64)

    def force_token_5(input_ids, logits):
        logits = logits.clone()
        logits[:] = float("-inf")
        logits[5] = 0.0
        return logits

    params = SamplingParams(temperature=0.0, max_tokens=4, logit_processors=[force_token_5])
    out = eng.generate([[1, 2, 3]], params)[0]
    assert out == [5, 5, 5, 5]


def test_varlen_prefill_matches_individual():
    """Multiple sequences of DIFFERENT lengths packed into one varlen forward pass
    must produce identical first-token output (per sequence) as individual
    single-sequence prefills. Proves the packed attention kernel, cumulative
    sequence lengths, and paged KV slot_mapping all thread correctly."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompts = [
        [1, 2, 3, 4],
        [5, 6],
        [7, 8, 9, 10, 11, 12, 13],
    ]
    params = SamplingParams(temperature=0.0, max_tokens=1)
    kv_cfg = dict(device="cuda", max_num_seqs=8, max_len=64, max_batch_tokens=8192)

    eng = LLMEngine(cfg, sd, **kv_cfg)
    outs_varlen = eng.generate(prompts, params)

    outs_ref: list[list[int]] = []
    for p in prompts:
        eng2 = LLMEngine(cfg, sd, **kv_cfg)
        outs_ref.append(eng2.generate([p], params)[0])

    for i, (v, ref) in enumerate(zip(outs_varlen, outs_ref)):
        assert v == ref, f"seq {i} (len={len(prompts[i])}): varlen {v} != individual {ref}"


def test_varlen_prefill_same_length():
    """Same-length sequences also hit the varlen path when batch > 1 and should
    still match individual prefills."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    params = SamplingParams(temperature=0.0, max_tokens=1)
    kv_cfg = dict(device="cuda", max_num_seqs=8, max_len=64, max_batch_tokens=8192)

    eng = LLMEngine(cfg, sd, **kv_cfg)
    outs_varlen = eng.generate(prompts, params)

    outs_ref: list[list[int]] = []
    for p in prompts:
        eng2 = LLMEngine(cfg, sd, **kv_cfg)
        outs_ref.append(eng2.generate([p], params)[0])

    for i, (v, ref) in enumerate(zip(outs_varlen, outs_ref)):
        assert v == ref, f"seq {i}: varlen {v} != individual {ref}"


def test_varlen_prefill_and_decode():
    """Varlen prefill followed by decode must produce the same full multi-token
    output as individual prefills + decodes."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompts = [
        [1, 2, 3, 4],
        [5, 6, 7],
        [8, 9, 10, 11, 12],
    ]
    params = SamplingParams(temperature=0.0, max_tokens=4)
    kv_cfg = dict(device="cuda", max_num_seqs=8, max_len=64, max_batch_tokens=8192)

    eng = LLMEngine(cfg, sd, **kv_cfg)
    outs_varlen = eng.generate(prompts, params)

    outs_ref: list[list[int]] = []
    for p in prompts:
        eng2 = LLMEngine(cfg, sd, **kv_cfg)
        outs_ref.append(eng2.generate([p], params)[0])

    for i, (v, ref) in enumerate(zip(outs_varlen, outs_ref)):
        assert v == ref, f"seq {i} (prompt len={len(prompts[i])}): varlen {v} != individual {ref}"


def test_preempt_and_resume():
    """When load exceeds max_num_seqs, the scheduler preempts running sequences
    (evict/recompute) instead of blocking; every request completes with the
    correct number of output tokens."""
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=2, max_len=64)
    # 5 prompts with max_num_seqs=2 -> slot pressure forces preemption
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11], [12, 13, 14]]
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
    assert len(outs) == 5
    for o in outs:
        assert len(o) == 8 and all(0 <= t < cfg.vocab_size for t in o)


def _qwen3_next_hybrid_cfg_sd():
    x = dict(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    cfg = ModelConfig(
        arch="qwen3_next",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        linear_attention=True,
        full_attention_interval=2,
        shared_expert_intermediate_size=64,
        extra=x,
    )
    H, nh, nkv, hd = 128, 4, 2, 32
    nk, nv, kd, vd = 2, 4, 16, 16
    qkv_lin = 2 * nk * kd + nv * vd

    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        if (i + 1) % 2 == 0:  # full attn
            sd[f"{p}.self_attn.q_proj.weight"] = r(nh * hd, H)
            sd[f"{p}.self_attn.k_proj.weight"] = r(nkv * hd, H)
            sd[f"{p}.self_attn.v_proj.weight"] = r(nkv * hd, H)
            sd[f"{p}.self_attn.o_proj.weight"] = r(H, nh * hd)
            sd[f"{p}.self_attn.q_norm.weight"] = r(hd)
            sd[f"{p}.self_attn.k_norm.weight"] = r(hd)
        else:  # linear (DeltaNet)
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = r(qkv_lin, H)
            sd[f"{la}.out_proj.weight"] = r(H, nv * vd)
            sd[f"{la}.conv_weight"] = r(qkv_lin, 4)
            sd[f"{la}.A_log"] = r(nv).float()
            sd[f"{la}.dt_bias"] = r(nv).float()
            sd[f"{la}.beta_proj.weight"] = r(nv, H)
            sd[f"{la}.dt_proj.weight"] = r(nv, H)
            sd[f"{la}.z_proj.weight"] = r(nv * vd, H)
            sd[f"{la}.norm.weight"] = r(nv * vd)
        for w, d in (("gate", cfg.num_experts),):
            sd[f"{p}.mlp.gate.weight"] = r(d, H)
        for e in range(cfg.num_experts):
            sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(cfg.moe_intermediate_size, H)
            sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(cfg.moe_intermediate_size, H)
            sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
        sd[f"{p}.mlp.shared_expert.gate_proj.weight"] = r(cfg.moe_intermediate_size, H)
        sd[f"{p}.mlp.shared_expert.up_proj.weight"] = r(cfg.moe_intermediate_size, H)
        sd[f"{p}.mlp.shared_expert.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
    return cfg, sd


def test_engine_qwen3_next_hybrid_matches_runner():
    """A hybrid (DeltaNet + full-attn) model decoded through the engine with
    recurrent-state carry must match the standalone ModelRunner output at the same
    token-agreement threshold as the pure-attention path."""
    torch.manual_seed(42)
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    prompt = [3, 1, 4, 1, 5]

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    eng_out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))[0]

    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref = runner.generate_greedy(torch.tensor([prompt], device="cuda"), max_new_tokens=5)[
        0
    ].tolist()

    assert eng_out[0] == ref[0], f"first token mismatch: {eng_out} vs {ref}"
    agree = sum(1 for a, b in zip(eng_out, ref) if a == b) / len(ref)
    assert agree >= 0.8, f"engine vs runner hybrid decode diverged too much: {eng_out} vs {ref}"
