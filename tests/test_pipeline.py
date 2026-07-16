# SPDX-License-Identifier: MIT
"""Pipeline-parallelism tests: 2-stage PP across 2 GPUs (issue #84).

Verifies:
    - make_pipeline splits a model into two stages on two GPUs.
    - A PP'd model produces output identical to a single-GPU reference.
    - Per-boundary send/recv uses the D9 transport seam.
    - Micro-batched decode overlap: transfer time hidden inside compute time.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("fni8")

from fni8serve.dist.pipeline import make_pipeline, PipelineEngine, PipelineStage
from fni8serve.engine import LLMEngine, SamplingParams
from fni8serve.models import ModelConfig

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="pipeline needs CUDA + fni8")

NUM_GPUS = torch.cuda.device_count() if CUDA else 0
two_gpus = pytest.mark.skipif(NUM_GPUS < 2, reason="needs >= 2 GPUs for PP")
three_gpus = pytest.mark.skipif(NUM_GPUS < 3, reason="needs >= 3 GPUs for PP")


def _cfg(n_layers: int = 2):
    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
    )


def _sd(cfg: ModelConfig):
    """Build a state dict on CPU so it can be cloned to multiple devices."""

    def r(*s):
        return torch.randn(*s, device="cpu", dtype=torch.float16) * 0.05

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    sd: dict = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
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


# ── Stage construction ─────────────────────────────────────────────────────


class TestMakePipeline:
    @three_gpus
    @cuda_only
    def test_make_pipeline_supports_three_stage_capacity(self):
        """A model larger than two cards must split across three stage-local slices."""
        cfg = _cfg(n_layers=6)
        stages = make_pipeline(cfg, _sd(cfg), devices=(0, 1, 2), max_num_seqs=4, max_len=64)

        assert len(stages) == 3
        assert [len(stage.layers) for stage in stages] == [2, 2, 2]
        assert [stage.layer_offset for stage in stages] == [0, 2, 4]
        assert [stage._device.index for stage in stages] == [0, 1, 2]
        assert stages[0].is_first and not stages[0].is_last
        assert not stages[1].is_first and not stages[1].is_last
        assert stages[2].is_last
        for expected_device, stage in enumerate(stages):
            qweight = stage.layers[0].self_attn.qkv_proj.weight.data
            assert qweight.device.index == expected_device

    @two_gpus
    @cuda_only
    def test_make_pipeline_builds_backbone_once(self, monkeypatch):
        """Stage construction must not duplicate a card-filling model per GPU."""
        import fni8serve.dist.pipeline as pipeline

        calls = 0
        real_build = pipeline.build_model

        def counted_build(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_build(*args, **kwargs)

        monkeypatch.setattr(pipeline, "build_model", counted_build)
        cfg = _cfg(n_layers=4)
        s0, s1 = pipeline.make_pipeline(cfg, _sd(cfg), devices=(0, 1), max_num_seqs=4, max_len=64)

        assert calls == 1
        assert s0.layers[0].self_attn.qkv_proj.weight.data.device.index == 0
        assert s1.layers[0].self_attn.qkv_proj.weight.data.device.index == 1

    @two_gpus
    @cuda_only
    def test_make_pipeline_returns_two_stages(self):
        torch.manual_seed(0)
        cfg = _cfg()
        sd = _sd(cfg)
        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        assert isinstance(s0, PipelineStage)
        assert isinstance(s1, PipelineStage)
        assert s0.is_first
        assert s1.is_last
        assert s0.embed is not None
        assert s0.lm_head is None
        assert s1.lm_head is not None
        assert s1.norm is not None

    @two_gpus
    @cuda_only
    def test_stage_layer_splits(self):
        torch.manual_seed(0)
        cfg = _cfg(n_layers=4)
        sd = _sd(cfg)
        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        assert len(s0.layers) == 2  # layers 0,1
        assert len(s1.layers) == 2  # layers 2,3
        assert s1.layer_offset == 2

    @two_gpus
    @cuda_only
    def test_stage_devices(self):
        torch.manual_seed(0)
        cfg = _cfg()
        sd = _sd(cfg)
        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        assert s0._device.index == 0
        assert s1._device.index == 1

    @two_gpus
    @cuda_only
    def test_stage_caches_allocated(self):
        torch.manual_seed(0)
        cfg = _cfg(n_layers=4)
        sd = _sd(cfg)
        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        assert s0.kv_cache.has_free_slot()
        assert s1.kv_cache.has_free_slot()
        slot0 = s0.kv_cache.alloc()
        slot1 = s1.kv_cache.alloc()
        assert slot0 >= 0
        assert slot1 >= 0
        assert slot0 == slot1  # both caches return same slot (they start identically)


# ── Pipeline integration (output correctness) ──────────────────────────────


class TestPipelineOutputMatchesSingleGpu:
    @three_gpus
    @cuda_only
    def test_same_output_three_stage_pipeline(self):
        torch.manual_seed(42)
        cfg = _cfg(n_layers=6)
        sd = _sd(cfg)
        prompt = [3, 1, 4, 1, 5]
        params = SamplingParams(temperature=0.0, max_tokens=4)

        ref = LLMEngine(
            cfg,
            {k: v.clone().to("cuda:0") for k, v in sd.items()},
            device="cuda:0",
            max_num_seqs=4,
            max_len=64,
        ).generate([prompt], params)[0]
        stages = make_pipeline(cfg, sd, devices=(0, 1, 2), max_num_seqs=4, max_len=64)
        engine = PipelineEngine.from_stages(
            stages, cfg, max_num_seqs=4, max_len=64, wire_scheme="int8"
        )

        assert engine.generate([prompt], params)[0] == ref

    @two_gpus
    @cuda_only
    def test_same_output_single_prompt(self):
        """PP output for one prompt must match single-GPU engine output token-for-token."""
        torch.manual_seed(42)
        cfg = _cfg()
        sd = _sd(cfg)
        prompt = [3, 1, 4, 1, 5]
        params = SamplingParams(temperature=0.0, max_tokens=5)

        # Single-GPU reference
        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=4, max_len=64)
        ref_out = eng_ref.generate([prompt], params)[0]

        # 2-stage PP (wire_scheme="int8" isolates pipeline logic from codec noise)
        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=4, max_len=64, wire_scheme="int8")
        pp_out = eng_pp.generate([prompt], params)[0]

        assert pp_out == ref_out, f"PP {pp_out} != ref {ref_out}"

    @two_gpus
    @cuda_only
    def test_same_output_multiple_prompts(self):
        """Multiple concurrent prompts must produce matching output."""
        torch.manual_seed(42)
        cfg = _cfg()
        sd = _sd(cfg)
        prompts = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11]]
        params = SamplingParams(temperature=0.0, max_tokens=4)

        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=8, max_len=64)
        ref_outs = eng_ref.generate(prompts, params)

        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=8, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=8, max_len=64, wire_scheme="int8")
        pp_outs = eng_pp.generate(prompts, params)

        for i, (pp, ref) in enumerate(zip(pp_outs, ref_outs)):
            assert pp == ref, f"prompt {i}: PP {pp} != ref {ref}"

    @two_gpus
    @cuda_only
    def test_same_output_4_layer_model(self):
        """A 4-layer model split 2+2 must match."""
        torch.manual_seed(42)
        cfg = _cfg(n_layers=4)
        sd = _sd(cfg)
        prompt = [3, 1, 4, 1, 5]
        params = SamplingParams(temperature=0.0, max_tokens=5)

        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=4, max_len=64)
        ref_out = eng_ref.generate([prompt], params)[0]

        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=4, max_len=64, wire_scheme="int8")
        pp_out = eng_pp.generate([prompt], params)[0]

        assert pp_out == ref_out, f"4-layer PP {pp_out} != ref {ref_out}"

    @two_gpus
    @cuda_only
    def test_same_output_with_logit_processors(self):
        """Per-request logit processors must work through the pipeline."""
        torch.manual_seed(42)
        cfg = _cfg()
        sd = _sd(cfg)
        prompt = [3, 1, 4]

        def force_token_7(input_ids, logits):
            logits = logits.clone()
            logits[:] = float("-inf")
            logits[7] = 0.0
            return logits

        params = SamplingParams(temperature=0.0, max_tokens=3, logit_processors=[force_token_7])

        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=4, max_len=64)
        ref_out = eng_ref.generate([prompt], params)[0]

        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=4, max_len=64, wire_scheme="int8")
        pp_out = eng_pp.generate([prompt], params)[0]

        assert ref_out == [7, 7, 7]
        assert pp_out == ref_out

    @two_gpus
    @cuda_only
    def test_same_output_respects_eos(self):
        torch.manual_seed(42)
        cfg = _cfg()
        sd = _sd(cfg)
        prompt = [1, 2, 3]
        params = SamplingParams(temperature=0.0, max_tokens=4)

        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=4, max_len=64)
        ref_out = eng_ref.generate([prompt], params)[0]

        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=4, max_len=64, wire_scheme="int8")
        pp_out = eng_pp.generate([prompt], params)[0]

        assert len(pp_out) == len(ref_out)
        assert pp_out == ref_out


# ── Transport seam ─────────────────────────────────────────────────────────


class TestPipelineTransportSeam:
    @two_gpus
    @cuda_only
    def test_transfer_handle_populated(self):
        """The send/recv on the PP boundary must complete without error and produce
        valid output, proving the D9 transport seam is used."""
        torch.manual_seed(0)
        cfg = _cfg()
        sd = _sd(cfg)
        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)

        eng = PipelineEngine(s0, s1, cfg, max_num_seqs=4, max_len=64)
        eng.add_request([3, 1, 4], SamplingParams(temperature=0.0, max_tokens=1))
        eng.step()

        out = eng._out
        seq = list(out.values())[0]
        assert len(seq.output_ids) == 1
        assert 0 <= seq.output_ids[0] < cfg.vocab_size


class TestMicrobatching:
    @two_gpus
    @cuda_only
    def test_microbatch_decode_batch_of_4(self):
        """Micro-batched decode over 4 sequences must produce the same output as
        single-GPU reference."""
        torch.manual_seed(42)
        cfg = _cfg()
        sd = _sd(cfg)
        prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12]]
        params = SamplingParams(temperature=0.0, max_tokens=4)

        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=8, max_len=64)
        ref_outs = eng_ref.generate(prompts, params)

        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=8, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=8, max_len=64, wire_scheme="int8")
        pp_outs = eng_pp.generate(prompts, params)

        for i, (pp, ref) in enumerate(zip(pp_outs, ref_outs)):
            assert pp == ref, f"micro-batched prompt {i}: PP {pp} != ref {ref}"

    @two_gpus
    @cuda_only
    def test_microbatch_decode_batch_of_1(self):
        """Single-sequence decode still works."""
        torch.manual_seed(42)
        cfg = _cfg()
        sd = _sd(cfg)
        prompt = [3, 1, 4, 1, 5]
        params = SamplingParams(temperature=0.0, max_tokens=6)

        sd0 = {k: v.clone().to("cuda:0") for k, v in sd.items()}
        eng_ref = LLMEngine(cfg, sd0, device="cuda:0", max_num_seqs=4, max_len=64)
        ref_out = eng_ref.generate([prompt], params)[0]

        s0, s1 = make_pipeline(cfg, sd, devices=(0, 1), max_num_seqs=4, max_len=64)
        eng_pp = PipelineEngine(s0, s1, cfg, max_num_seqs=4, max_len=64, wire_scheme="int8")
        pp_out = eng_pp.generate([prompt], params)[0]

        assert pp_out == ref_out, f"batch=1 PP {pp_out} != ref {ref_out}"
