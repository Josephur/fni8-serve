# SPDX-License-Identifier: MIT
"""CUDA-graph decode (issue #42): capturing the whole decode step (model forward
+ logits) into a CUDA graph collapses per-step dispatch (~3,200 cudaLaunchKernel
launches on a 28-layer 0.6B model) down to one `cudaGraphLaunch` on replay.

The one thing that MUST hold for this to be safe: graphed decode and eager decode
produce the SAME token ids for the same seed/prompts, at every batch-size bucket
boundary (unpadded, exactly-at-bucket, and padded) and across a context-length
bucket re-capture. `test_paged_engine.py` / `test_engine.py` already cover the
eager path's own numerics against the non-paged reference runner; this file only
checks graphed-vs-eager agreement plus the capture/fallback bookkeeping.
"""
import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.engine import LLMEngine, PagedKVCache, SamplingParams
from fni8serve.engine.cuda_graph import GraphedDecode
from fni8serve.models import ModelConfig

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="cuda-graph decode needs a real CUDA device")


def _cfg(**overrides):
    base = dict(arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=3,
               num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
               max_position_embeddings=512, head_dim=32, qk_norm=True,
               tie_word_embeddings=True)
    base.update(overrides)
    return ModelConfig(**base)


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
    hd, nh, nkv, H = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
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


def _prompts(num_seqs):
    # Different lengths per sequence, like the ragged-batch engine tests.
    return [[(i * 7 + j) % 200 + 1 for j in range(3 + i % 3)] for i in range(num_seqs)]


def _run(num_seqs, *, enable_cuda_graph, seed=0, max_tokens=6, max_num_seqs=16, max_len=64):
    torch.manual_seed(seed)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=max_num_seqs, max_len=max_len,
                    enable_cuda_graph=enable_cuda_graph)
    outs = eng.generate(_prompts(num_seqs),
                        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
    return outs, eng


@pytest.mark.parametrize("num_seqs", [1, 3, 8])
def test_graphed_matches_eager_token_ids(num_seqs):
    """Same seed/prompts, greedy sampling: graphed decode (batch padded up to the
    1/2/4/8/16/... bucket -- num_seqs=3 pads to 4, num_seqs=8 is an exact bucket)
    must produce BIT-IDENTICAL token ids to eager decode."""
    eager, _ = _run(num_seqs, enable_cuda_graph=False)
    graphed, eng = _run(num_seqs, enable_cuda_graph=True)
    assert eng.runner.graphed is not None and eng.runner.graphed.supported
    assert graphed == eager


@pytest.mark.parametrize("num_seqs", [1, 8])
def test_graphed_decode_is_deterministic(num_seqs):
    """The same seed run 3x through the graphed path must produce the exact same
    tokens every time -- catches replay reading stale/uninitialized buffer memory
    (e.g. a pad-row slot that was never re-copied before a later real use)."""
    runs = [_run(num_seqs, enable_cuda_graph=True)[0] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_graph_recaptures_on_context_bucket_growth():
    """A long enough generation crosses a context-length bucket boundary, forcing
    a second capture at the new bucket -- must still match eager token-for-token
    (proves re-capture-on-growth is correct, not just the first bucket)."""
    cfg = _cfg()

    torch.manual_seed(7)
    sd_eager = _sd(cfg)
    eng_eager = LLMEngine(cfg, sd_eager, device="cuda", max_num_seqs=4, max_len=256,
                          enable_cuda_graph=False)
    out_eager = eng_eager.generate([[3, 1, 4, 1, 5, 9, 2, 6]],
                                   SamplingParams(temperature=0.0, max_tokens=80, ignore_eos=True))[0]

    torch.manual_seed(7)
    sd_graph = _sd(cfg)
    eng_graph = LLMEngine(cfg, sd_graph, device="cuda", max_num_seqs=4, max_len=256,
                          enable_cuda_graph=True)
    eng_graph.runner.graphed.context_bucket_size = 16   # force a couple of re-captures
    out_graph = eng_graph.generate([[3, 1, 4, 1, 5, 9, 2, 6]],
                                   SamplingParams(temperature=0.0, max_tokens=80, ignore_eos=True))[0]

    assert out_graph == out_eager
    assert len(eng_graph.runner.graphed._graphs) > 1, \
        "expected more than one context bucket to have been captured"


def test_graphed_decode_does_not_shrink_real_concurrency():
    """GraphedDecode's scratch pad row (engine/cuda_graph.py `_scratch`) pins one
    cache slot forever -- it must come from a slot dedicated to that purpose, not
    one of the `max_num_seqs` the caller was promised. Fill every real slot
    concurrently and confirm all `max_num_seqs` requests actually ran together."""
    torch.manual_seed(3)
    cfg = _cfg()
    sd = _sd(cfg)
    max_num_seqs = 4
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=max_num_seqs, max_len=64,
                    enable_cuda_graph=True)
    outs = eng.generate(_prompts(max_num_seqs),
                        SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))
    assert len(outs) == max_num_seqs
    assert all(len(o) == 6 for o in outs)
    # The scheduler must have been able to admit all `max_num_seqs` at once (not
    # max_num_seqs - 1), i.e. the scratch slot came from outside its pool.
    assert eng.cache.num_slots == max_num_seqs + 1


# -- recurrent (LFM2 short-conv) graphed decode ---------------------------------
def _lfm2_cfg(**overrides):
    base = dict(arch="lfm2", vocab_size=256, hidden_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                max_position_embeddings=512, head_dim=32, qk_norm=True,
                tie_word_embeddings=True, rms_norm_eps=1e-5,
                extra=dict(full_attn_idxs=[1], conv_L_cache=3))
    base.update(overrides)
    return ModelConfig(**base)


def _lfm2_sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    hd, nh, nkv = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads
    attn_idxs = set(cfg.extra["full_attn_idxs"])
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = r(H)
        sd[f"{p}.ffn_norm.weight"] = r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = r(d, H if w != "w2" else cfg.intermediate_size)
        if i in attn_idxs:
            a = f"{p}.self_attn"
            sd[f"{a}.q_proj.weight"] = r(nh * hd, H)
            sd[f"{a}.k_proj.weight"] = r(nkv * hd, H)
            sd[f"{a}.v_proj.weight"] = r(nkv * hd, H)
            sd[f"{a}.out_proj.weight"] = r(H, nh * hd)
            sd[f"{a}.q_layernorm.weight"] = r(hd)
            sd[f"{a}.k_layernorm.weight"] = r(hd)
        else:
            c = f"{p}.conv"
            sd[f"{c}.in_proj.weight"] = r(3 * H, H)
            sd[f"{c}.out_proj.weight"] = r(H, H)
            sd[f"{c}.conv.weight"] = r(H, 1, 3)
    return sd


def _run_lfm2(num_seqs, *, enable_cuda_graph, seed=0, max_tokens=6, max_num_seqs=16, max_len=64):
    torch.manual_seed(seed)
    cfg = _lfm2_cfg()
    sd = _lfm2_sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=max_num_seqs, max_len=max_len,
                    enable_cuda_graph=enable_cuda_graph)
    outs = eng.generate(_prompts(num_seqs),
                        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))
    return outs, eng


def test_graphed_decode_accepts_recurrent_lfm2():
    """A recurrent (LFM2 short-conv) model must now capture: with fixed-address
    per-slot recurrent-state buffers the decode step is static, so GraphedDecode
    should report supported instead of declining 'recurrent ... not capturable'."""
    from fni8serve.models import build_model
    cfg = _lfm2_cfg()
    model = build_model(cfg, _lfm2_sd(cfg)).cuda().eval()
    cache = PagedKVCache(cfg.num_hidden_layers, 4, cfg.num_key_value_heads, 64,
                         cfg.resolved_head_dim(), device="cuda")
    g = GraphedDecode(model, cache, device="cuda")
    assert g.supported, g._unsupported_reason


@pytest.mark.parametrize("num_seqs", [1, 3, 8])
def test_graphed_matches_eager_lfm2(num_seqs):
    """Recurrent graphed decode must produce BIT-IDENTICAL token ids to eager."""
    eager, _ = _run_lfm2(num_seqs, enable_cuda_graph=False)
    graphed, eng = _run_lfm2(num_seqs, enable_cuda_graph=True)
    assert eng.runner.graphed is not None and eng.runner.graphed.supported
    assert graphed == eager


@pytest.mark.parametrize("num_seqs", [1, 8])
def test_graphed_recurrent_decode_is_deterministic(num_seqs):
    runs = [_run_lfm2(num_seqs, enable_cuda_graph=True)[0] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_graphed_recurrent_concurrent_independent():
    """Two recurrent sequences decoded concurrently under CUDA graphs must each
    match their solo (single-stream) graphed decode -- per-slot state stays isolated
    across the shared fixed-address buffers."""
    p = _prompts(2)
    both, _ = _run_lfm2(2, enable_cuda_graph=True)
    solo0 = _run_lfm2(1, enable_cuda_graph=True)[0][0]
    # re-run seq 1 alone by building an engine with just the second prompt
    torch.manual_seed(0)
    cfg = _lfm2_cfg()
    sd = _lfm2_sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=16, max_len=64, enable_cuda_graph=True)
    solo1 = eng.generate([p[1]], SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))[0]
    assert both[0] == solo0
    assert both[1] == solo1


class _FakeModel:
    """`GraphedDecode._check_supported` only reads `model.config` -- stub the
    rest so the capability check can be tested without building full weights."""
    def __init__(self, cfg):
        self.config = cfg
        self._mods = []

    def modules(self):
        return iter(self._mods)


def _tiny_cache():
    return PagedKVCache(1, 2, 2, 32, 16, device="cuda")


def test_graphed_decode_declines_moe():
    """MoE expert routing (`mask.nonzero()`) is data-dependent control flow a
    CUDA graph can't represent -- GraphedDecode must detect and refuse it
    up front rather than silently capturing something wrong."""
    cfg = _cfg(arch="qwen3_moe", num_experts=4, num_experts_per_tok=2,
              moe_intermediate_size=64)
    g = GraphedDecode(_FakeModel(cfg), _tiny_cache(), device="cuda")
    assert not g.supported


def test_graphed_decode_declines_sliding_window():
    """The sliding-window decode path falls back to a per-slot dequant + python
    loop (see GQAAttention._decode_batched) -- also not graph-capturable."""
    cfg = _cfg(sliding_window=4, sliding_window_pattern=2)
    g = GraphedDecode(_FakeModel(cfg), _tiny_cache(), device="cuda")
    assert not g.supported


def test_graphed_decode_accepts_plain_dense_qwen3():
    cfg = _cfg()
    g = GraphedDecode(_FakeModel(cfg), _tiny_cache(), device="cuda")
    assert g.supported


def test_batch_larger_than_widest_bucket_falls_back_to_eager():
    """A batch bigger than every configured bucket must return None (eager
    fallback), not raise or silently truncate the batch."""
    cfg = _cfg()
    sd = _sd(cfg)
    from fni8serve.models import build_model
    model = build_model(cfg, sd).cuda().eval()
    cache = PagedKVCache(cfg.num_hidden_layers, 4, cfg.num_key_value_heads, 64,
                         cfg.resolved_head_dim(), device="cuda")
    g = GraphedDecode(model, cache, device="cuda", batch_buckets=(1, 2))
    from fni8serve.engine.sequence import SamplingParams as SP
    from fni8serve.engine.sequence import Sequence
    seqs = [Sequence(i, [1, 2, 3], SP()) for i in range(3)]
    for s in seqs:
        s.slot = cache.alloc()
        s.length = 3
    assert g.try_decode(seqs) is None
