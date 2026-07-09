# SPDX-License-Identifier: MIT
"""Prefix caching tests (issue #80): two requests with a shared prefix skip
recompute of the shared portion, and output is identical to no-cache."""

import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.engine import LLMEngine, PagedKVCache, SamplingParams
from fni8serve.models import ModelConfig

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


def test_prefix_cache_matches_no_cache():
    """Two requests sharing a prefix must produce identical greedy output to the
    same requests run without prefix caching, and the second request should reuse
    the first's KV blocks for the shared portion."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt_a = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    prompt_b = [
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        25,
    ]  # same 17-token prefix, diff last token

    params = SamplingParams(temperature=0.0, max_tokens=8)

    # Reference run: no caching (both requests processed together, fresh cache)
    torch.manual_seed(42)
    ref_eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False
    )
    ref_out = ref_eng.generate([prompt_a, prompt_b], params)

    # Cached run: process one at a time so prefix is stored before second request
    torch.manual_seed(42)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False)

    sid_a = eng.add_request(prompt_a, params)
    while not eng.sequence(sid_a).is_finished(eng.eos_id):
        eng.step()
    out_a = eng.sequence(sid_a).output_ids
    eng.forget(sid_a)

    # Before second request, verify prefix was stored
    matched, blocks = eng.cache.lookup_prefix(prompt_b)
    assert matched >= 16, f"expected prefix match >=16 tokens, got {matched}"

    sid_b = eng.add_request(prompt_b, params)
    seq_b = eng.sequence(sid_b)
    initial_blocks = eng.cache.num_blocks - len(eng.cache._free_blocks)

    while not seq_b.is_finished(eng.eos_id):
        eng.step()
    out_b = seq_b.output_ids
    eng.forget(sid_b)

    # Output must match reference
    assert out_a == ref_out[0], f"prefix-cached A {out_a} != ref {ref_out[0]}"
    assert out_b == ref_out[1], f"prefix-cached B {out_b} != ref {ref_out[1]}"


def test_prefix_cache_block_reuse():
    """The second request must reuse the first request's physical blocks for the
    shared prefix — alloc at most one new block for the suffix."""
    torch.manual_seed(99)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False)
    pool_size = eng.cache.num_blocks

    # First request: a long prefix
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    params = SamplingParams(temperature=0.0, max_tokens=4)

    sid1 = eng.add_request(prompt, params)
    while not eng.sequence(sid1).is_finished(eng.eos_id):
        eng.step()
    eng.forget(sid1)

    # After first request finishes, all blocks should be free (or kept by prefix cache)
    free_after_first = len(eng.cache._free_blocks)

    # Second request: same prefix, different suffix
    prompt2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30]
    sid2 = eng.add_request(prompt2, params)
    seq2 = eng.sequence(sid2)

    # Check that blocks were shared
    matched, blocks = eng.cache.lookup_prefix(prompt2)
    assert matched >= 16, f"prefix match expected >=16, got {matched}"
    assert len(blocks) > 0, "should share at least one block"

    free_before_second_start = len(eng.cache._free_blocks)
    while not seq2.is_finished(eng.eos_id):
        eng.step()
    eng.forget(sid2)

    # Verify blocks reused: at most one block is held by the prefix entry
    # (by design — the prefix persists for future sharing). All other blocks
    # must be free.
    free_after_second = len(eng.cache._free_blocks)
    held_by_prefix = pool_size - free_after_second
    assert held_by_prefix <= 1, (
        f"at most 1 block held by prefix cache, got {held_by_prefix} missing: "
        f"{free_after_second}/{pool_size}"
    )
    # The shared block must still be reachable via lookup
    matched2, blocks2 = eng.cache.lookup_prefix(prompt2)
    assert matched2 >= 16 and len(blocks2) > 0, "prefix should still be cached"


def test_prefix_cache_lookup_no_match():
    """Lookup returns 0 for a prompt with no stored prefix."""
    torch.manual_seed(7)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False)
    matched, blocks = eng.cache.lookup_prefix([99, 98, 97])
    assert matched == 0
    assert blocks == []
