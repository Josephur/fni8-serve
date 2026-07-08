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
    return ModelConfig(arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=256, head_dim=32, qk_norm=True,
                       tie_word_embeddings=True)


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


def test_engine_generates_for_concurrent_requests():
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11, 12, 13]]   # different lengths
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6))
    assert len(outs) == 4
    for o in outs:
        assert len(o) == 6 and all(0 <= t < cfg.vocab_size for t in o)


def test_engine_matches_single_sequence_runner():
    """Greedy engine output for one request must equal the standalone runner (the
    ragged-decode path is numerically identical to the uniform path)."""
    torch.manual_seed(1)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt = [3, 1, 4, 1, 5]

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    eng_out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))[0]

    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref = runner.generate_greedy(torch.tensor([prompt], device="cuda"), max_new_tokens=5)[0].tolist()

    assert eng_out == ref


def test_engine_respects_eos():
    torch.manual_seed(2)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64, eos_id=None)
    out = eng.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=4))[0]
    assert len(out) == 4
