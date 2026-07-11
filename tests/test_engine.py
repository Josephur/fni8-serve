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
    """When load exceeds max_num_seqs, excess requests WAIT in FIFO and are admitted
    as running sequences finish (no count-cap preemption / recompute thrash); every
    request still completes with the correct number of output tokens."""
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=2, max_len=64)
    # 5 prompts with max_num_seqs=2 -> slot pressure; the extra 3 queue and drain
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


def test_engine_qwen3_next_concurrent_recurrent_decode():
    """Continuous batching for a recurrent (hybrid DeltaNet) family: two sequences
    decoded CONCURRENTLY through one engine must produce exactly what each produces
    when run alone. Before per-slot recurrent state this crashed outright — the
    conv-tail cached from the last prefilled sequence (batch 1) was `torch.cat`-ed
    against a batch-2 decode input — and, absent the crash, the two sequences shared
    one global recurrent state and corrupted each other. This is the property that
    makes divergent families actually SERVE (not just single-request generate)."""
    torch.manual_seed(42)
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    pA = [3, 1, 4, 1, 5, 9, 2, 6]
    pB = [7, 2, 7, 1, 8, 2, 8]  # different length on purpose (ragged batch)
    params = SamplingParams(temperature=0.0, max_tokens=6)

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    both = eng.generate([pA, pB], params)

    solo_a = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64).generate([pA], params)[0]
    solo_b = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64).generate([pB], params)[0]

    assert both[0] == solo_a, f"seq A corrupted by concurrent decode: {both[0]} vs {solo_a}"
    assert both[1] == solo_b, f"seq B corrupted by concurrent decode: {both[1]} vs {solo_b}"


# ── MTP speculative-decode tests ──────────────────────────────────────────


def _cfg_mtp():
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
        num_mtp_layers=1,  # single MTP depth
    )


def _sd_mtp(cfg):
    def r(*s):
        # Larger weight scale than the other engine tests on purpose: it makes the
        # tiny random model's greedy decode genuinely CONTEXT-sensitive (a varied,
        # non-cyclic token stream) instead of collapsing to a 2-token limit cycle.
        # That sensitivity is what lets the spec-decode bit-identical test below
        # actually catch a corrupted / missing accepted-token KV commit — with a
        # degenerate cyclic model, a KV hole leaves the output unchanged.
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.5

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
    # MTP depth 1
    mp = "model.mtp.1"
    sd[f"{mp}.fc.weight"] = r(H, 2 * H)
    sd[f"{mp}.pre_fc_norm_hidden.weight"] = r(H)
    sd[f"{mp}.pre_fc_norm_embedding.weight"] = r(H)
    sd[f"{mp}.self_attn.q_proj.weight"] = r(nh * hd, H)
    sd[f"{mp}.self_attn.k_proj.weight"] = r(nkv * hd, H)
    sd[f"{mp}.self_attn.v_proj.weight"] = r(nkv * hd, H)
    sd[f"{mp}.self_attn.o_proj.weight"] = r(H, nh * hd)
    sd[f"{mp}.self_attn.q_norm.weight"] = r(hd)
    sd[f"{mp}.self_attn.k_norm.weight"] = r(hd)
    sd[f"{mp}.input_layernorm.weight"] = r(H)
    sd[f"{mp}.post_attention_layernorm.weight"] = r(H)
    sd[f"{mp}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
    sd[f"{mp}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
    sd[f"{mp}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_mtp_spec_decode_runs():
    """MTP spec-decode completes without error (single seq, greedy)."""
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=2, max_len=64)
    prompt = [1, 2, 3, 4]
    out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
    assert len(out) == 1
    assert len(out[0]) == 8
    assert all(0 <= t < cfg.vocab_size for t in out[0])


def _no_mtp_cfg(cfg):
    """The same ModelConfig with the MTP head stripped — the non-spec greedy
    reference engine (plain autoregressive decode, one token per step)."""
    return ModelConfig(
        **{k: v for k, v in vars(cfg).items() if k not in ("num_mtp_layers", "extra")},
        extra=cfg.extra,
    )


def _force_full_acceptance(eng_spec, ref_by_prompt):
    """Wire a spec-decode engine so its MTP draft head ALWAYS proposes the true
    greedy continuation, forcing every draft to be accepted (n_acc >= 2).

    Random-init MTP heads have ~0 draft acceptance, so the accept/commit path
    (the part with the corrupting bugs) would never run — every step would fall
    back to n_acc == 1, which is trivially equal to non-spec greedy and hides
    both the double-o_proj verify bug and the missing accepted-token KV commit.
    We instead monkeypatch ``draft_greedy`` to emit, per sequence, the very token
    the target model will greedily verify next (looked up from a precomputed
    non-spec reference), so acceptance is guaranteed and the real commit path is
    exercised. Returns a list that accumulates the per-step n_acc actually taken.

    ``ref_by_prompt`` maps ``tuple(prompt_ids) -> (prompt_len, reference_output)``.
    """
    holder: dict = {}
    naccs: list[int] = []

    def forced_draft(last_hidden, last_token, positions, ctx):
        batch = holder["batch"]
        toks = []
        for b in range(positions.shape[0]):
            P, ref = ref_by_prompt[tuple(batch[b].prompt_ids)]
            # positions[b, 0] is this sequence's current KV length L; after prefill
            # L == P and len(output_ids) == 1, and both advance in lockstep, so the
            # token that should follow the base token is ref[L - P + 2].
            idx = int(positions[b, 0].item()) - P + 2
            toks.append(ref[idx] if 0 <= idx < len(ref) else 0)
        return [torch.tensor(toks, device=last_hidden.device, dtype=torch.long).view(-1, 1)]

    eng_spec.model.mtp.draft_greedy = forced_draft

    runner = eng_spec.runner
    orig = runner._spec_decode_eager

    def wrapped(batch, mtp):
        holder["batch"] = batch  # so forced_draft can map row -> sequence
        before = [len(s.output_ids) for s in batch]
        result = orig(batch, mtp)
        naccs.extend(len(s.output_ids) - b for s, b in zip(batch, before))
        return result

    runner._spec_decode_eager = wrapped
    return naccs


def test_mtp_spec_decode_bit_identical_greedy():
    """Greedy spec-decode MUST be bit-identical to plain greedy decode.

    This is the load-bearing correctness contract: with temperature 0 the
    accept-longest-greedy-prefix rule can only ever emit tokens the target model
    would have emitted anyway, so the token stream must match non-spec greedy
    EXACTLY (not "mostly" — the old >= 0.8 agreement bar passed even with both
    silent-corruption bugs present). We force full draft acceptance so several
    tokens are committed per step, which:
      * exercises the accepted-token KV commit — if the intermediate accepted
        tokens' K/V are not written to the paged cache, the next step reads a
        hole and the output diverges (caught by the bit-identical assert);
      * exercises the verify o_proj path — if ``_verify_batched``'s output is
        o_proj'd twice, the verify logits are garbage, no forced draft is ever
        accepted, and n_acc collapses to 1 (caught by the n_acc >= 2 assert).
    """
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    params = SamplingParams(temperature=0.0, max_tokens=24)

    # Non-spec greedy reference.
    eng_ref = LLMEngine(_no_mtp_cfg(cfg), sd, device="cuda", max_num_seqs=2,
                        max_len=64, enable_cuda_graph=False)
    out_ref = eng_ref.generate([prompt], params)[0]

    # Spec-decode engine with forced full acceptance.
    eng_spec = LLMEngine(cfg, sd, device="cuda", max_num_seqs=2, max_len=64,
                         enable_cuda_graph=False)
    naccs = _force_full_acceptance(eng_spec, {tuple(prompt): (len(prompt), out_ref)})
    out_spec = eng_spec.generate([prompt], params)[0]

    assert max(naccs) >= 2, (
        f"forced draft was never accepted (n_acc stayed 1: {naccs}); verify logits "
        f"are wrong — check for double o_proj in _verify_batched"
    )
    assert out_spec == out_ref, (
        f"spec-decode NOT bit-identical to non-spec greedy:\n  spec={out_spec}\n   ref={out_ref}\n"
        f"  (accepted-token KV likely not committed to the paged cache)"
    )


def test_mtp_spec_decode_bit_identical_greedy_concurrent():
    """Same bit-identical contract as above, but with THREE sequences of
    different lengths decoded concurrently through one engine — the accepted-token
    KV commit and verify o_proj must be correct per row in a ragged batch, not
    just for a single sequence."""
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6], [7, 2, 7, 1, 8], [1, 6, 1, 8, 0, 3]]
    params = SamplingParams(temperature=0.0, max_tokens=20)

    eng_ref = LLMEngine(_no_mtp_cfg(cfg), sd, device="cuda", max_num_seqs=4,
                        max_len=64, enable_cuda_graph=False)
    refs = eng_ref.generate(prompts, params)

    eng_spec = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64,
                         enable_cuda_graph=False)
    ref_by_prompt = {tuple(p): (len(p), r) for p, r in zip(prompts, refs)}
    naccs = _force_full_acceptance(eng_spec, ref_by_prompt)
    outs = eng_spec.generate(prompts, params)

    assert max(naccs) >= 2, f"forced draft never accepted in the concurrent batch: {naccs}"
    for i, (spec, ref) in enumerate(zip(outs, refs)):
        assert spec == ref, (
            f"seq {i}: spec-decode NOT bit-identical to non-spec greedy:\n"
            f"  spec={spec}\n   ref={ref}"
        )
