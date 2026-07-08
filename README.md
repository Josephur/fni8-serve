# fni8-serve

A minimal INT8 (W8A8) inference server for the NVIDIA Volta / CMP 100-210 fleet, built
as a serving layer over the [`fni8`](https://github.com/jajmangold/fni8)
FlashAttention-2 dp4a kernels.

The scheduler, paged-KV, and continuous-batching design follow
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT). The compute path swaps
fp16 flash-attention for `fni8`'s int8 dp4a kernels, because on this hardware the fp16
tensor cores are firmware-disabled and dp4a is the fast path.

## Contents

- [Why this exists](#why-this-exists)
- [Pre-quantized models on Hugging Face](#pre-quantized-models-on-hugging-face)
- [Architecture](#architecture)
- [The `.fni8` weight format](#the-fni8-weight-format)
- [Quickstart](#quickstart)
- [Model support](#model-support)
- [Status](#status)
- [The fni8 family](#the-fni8-family)
- [License](#license)

## Why this exists

The deployment fleet is the CMP 100-210 (GV100 silicon, 16 GB HBM2 at 829 GB/s, PCIe 1.0
x1 at roughly 250 MB/s). Two hardware facts drive every design choice.

First, the fp16/TF32 tensor cores are firmware-limited to about 6.9 TFLOP/s, while INT8
`__dp4a` on the CUDA cores is healthy at about 46 TOP/s. So this server runs in W8A8, not
fp16, which is the opposite of a normal GPU's verdict. `fni8` is the kernel that makes
that work.

Second, the interconnect is roughly 3,300x slower than HBM. Tensor and FSDP parallelism
are unusable here (seconds per token); only pipeline and MoE-expert parallelism survive,
with the tokens on the wire compressed by `fni8.transport`. See
`fused_ni8/utils/docs/transport-compression.md`.

## Pre-quantized models on Hugging Face

Ready-to-serve `.fni8` weights (int8 and int4 in each repo) live under
[huggingface.co/jajmangold](https://huggingface.co/jajmangold?search=fni8). Each is
published as a linked quantization, so it also appears under its source model's
Quantizations tab. Produce more with `tools/forge.sh` (download, quantize,
`forge publish`).

## Architecture

The engine is ported from nano-vllm; the compute layers are swapped for `fni8`.

| nano-vllm piece | here |
| --- | --- |
| engine: scheduler, block_manager, sequence, runner | keep (port), the serving loop |
| `layers/attention.py` (flash-attn fp16) | swap to `fni8` int8 prefill + decode, int8 paged KV |
| `store_kvcache` Triton kernel | adapt to quantize-on-write into an int8 paged cache |
| `layers/linear.py` (fp16) | swap to int8 / W4A8 dp4a GEMM (via `fni8`) |
| layernorm / rotary / activation / embed / sampler | keep (torch) |
| tensor parallelism (torch.distributed) | replace with PP + MoE-EP + `fni8.transport` |
| GGUF / safetensors loader | replace with the `.fni8` zero-transform mmap loader |

## The `.fni8` weight format

Weights load through `fni8`'s `.fni8` container: the on-disk bytes are the resident dp4a
layout, so loading is `mmap + cudaMemcpy` with no dequant or repack. The container holds
int8 (`per_row_i8`) and 4-bit (`per_group_i4`) weights, fp32 scales, Hadamard/smoothing
flags, and a shard index for rank-local partial loads. 4-bit weights halve the footprint
(2x model capacity in 16 GB) and speed weight-bandwidth-bound decode; they unpack to int8
in-kernel, since sm_70 has no int4 matmul.

## Quickstart

Convert an HF checkpoint, then serve it:

```bash
# 1. Convert to .fni8 (int8 or 4-bit weights, resident dp4a layout)
python -m fni8serve.convert  /path/to/Qwen3-8B  qwen3-8b.fni8  --bits 8
```

```python
# 2. Serve it. Architecture is auto-detected from the checkpoint; continuous batching.
from fni8serve import ModelConfig, LLMEngine, SamplingParams, load_fni8_state_dict, checkpoint_info

cfg = ModelConfig.from_hf(checkpoint_info("qwen3-8b.fni8")["meta"]["config"])
engine = LLMEngine(cfg, load_fni8_state_dict("qwen3-8b.fni8"))
out = engine.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=32))
```

## OpenAI-compatible API server

`fni8serve.api` puts an OpenAI-compatible HTTP surface over `LLMEngine`:
`/v1/chat/completions`, `/v1/completions`, `/v1/models`, both streaming (SSE) and
non-streaming. Chat formatting uses the model's own HF `apply_chat_template` (not a
hardcoded template), so tool/template formatting comes free per model. Needs the
`serve` extra: `pip install -e ".[serve]"`.

```bash
# The .fni8 file carries weights + config, not tokenizer files -- point --tokenizer
# at the base model repo (or a fni8 quant repo that mirrors one).
python -m fni8serve.api.server --model qwen3-8b.fni8 --tokenizer Qwen/Qwen3-8B
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-8b.fni8",
    messages=[{"role": "user", "content": "Say hi in five words."}],
)
print(resp.choices[0].message.content)
```

A per-step **logit-processor hook** sits in the sampler (`fni8serve.layers.sampler`,
wired through `SamplingParams.logit_processors`): a list of
`(input_ids, logits) -> logits` callables applied before sampling, per sequence.
This is the seam structured outputs / grammars plug into.

### Structured outputs / grammars

`fni8serve.structured` plugs [XGrammar](https://github.com/mlc-ai/xgrammar) into that
logit-processor hook: it compiles a JSON schema or a grammar into a token-mask
matcher, masks disallowed tokens each step, and advances the matcher on the token
actually sampled. Pure Python/CPU -- no CUDA needed for the mask itself. Needs the
`structured` extra: `pip install -e ".[structured]"`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-8b.fni8",
    messages=[{"role": "user", "content": "Extract the name and age as JSON."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "person", "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }},
    },
)
```

A `grammar` request field is the same mechanism for a raw grammar (GBNF/EBNF)
instead of a JSON schema -- and wins if both are set:

```python
resp = client.chat.completions.create(
    model="qwen3-8b.fni8",
    messages=[{"role": "user", "content": "Yes or no: is the sky blue?"}],
    extra_body={"grammar": 'root ::= "yes" | "no"'},
)
```

`fni8serve.structured.GrammarCompilerCache` caches the (vocab-walk-expensive)
`xgrammar.GrammarCompiler` per tokenizer; XGrammar itself caches compiled
grammars/schemas, so repeat schemas across requests compile once.

`--chat-template <file>` overrides the tokenizer's own embedded template with a
Jinja source file (matches vLLM's `--chat-template` flag). The override still
renders through `apply_chat_template`, so it gets the same sandboxed Jinja
environment (`jinja2.sandbox.ImmutableSandboxedEnvironment`, via `transformers`) as
the model's own template -- a custom template is never given more than that.

## Model support

Model support is a registry: a family is a `ModelConfig` plus a thin `models/<family>.py`
over shared layers, registered with `@register_model`. The engine never changes. See
[`fni8serve/models/COVERAGE.md`](fni8serve/models/COVERAGE.md) for current status.

- Full decode through the dp4a kernels: Qwen3, Qwen3-MoE, Gemma3 (GQA with QK-norm and
  sliding window).
- Divergent-attention families on ported fp16 backends (int8 acceleration is the
  DeltaNet/MLA kernel track in `fni8`): DeepSeek (MLA), Qwen3-Next / 3.5 / 3.6 (hybrid
  Gated-DeltaNet plus full attention), LFM2 (short-conv plus attention), GLM-4.5/4.6,
  Hunyuan, MiniMax (lightning).

## Status

- [x] `.fni8` zero-transform weight loader plus HF-to-`.fni8` converter
      (`fni8serve.convert`), including FP8-source models (DeepSeek, Hy3)
- [x] int8 / W4A8 dp4a GEMM in `fni8`; the Linear seam runs it (fp16 fallback for NF4/CPU)
- [x] Continuous-batching engine (scheduler, slot KV, `LLMEngine.generate()`)
- [x] Modular model registry, 8+ families (build + prefill validated)
- [x] Recurrent-state and latent decode caching for the linear/DeltaNet/MLA families
- [x] Paged-KV block-table with int8 quantize-on-write, wired into the engine
- [x] OpenAI-compatible API server (`fni8serve.api`) with SSE streaming, plus a
      per-step logit-processor hook in the sampler for structured outputs / grammars
- [x] Structured outputs / grammars (`fni8serve.structured`): XGrammar behind the
      logit-processor hook, `response_format={type: json_schema}` + a `grammar`
      (GBNF/EBNF) extension
- [ ] int8 dp4a DeltaNet and MLA kernels in `fni8` (the divergent-attention acceleration)
- [ ] multi-GPU PP + MoE-EP with `fni8.transport`

## The fni8 family

Part of a three-repo stack on the same sm_70 fleet:

- **[fni8](https://github.com/jajmangold/fni8)**: the dp4a kernels and `.fni8` format.
- **fni8-serve** (this repo): LLM serving.
- **[ComfyUI-fni8](https://github.com/jajmangold/ComfyUI-fni8)**: diffusion DiTs in ComfyUI.

## License

MIT (see `LICENSE`). This project adapts the nano-vllm design (MIT). It depends on
`fni8`, which is BSD-3-Clause. See `NOTICE`.
