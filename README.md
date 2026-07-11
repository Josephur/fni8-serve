# fni8-serve

A minimal INT8 (W8A8) inference server for the NVIDIA Volta / CMP 100-210 fleet, built
as a serving layer over the [`fni8`](https://github.com/jajmangold/fni8)
FlashAttention-2 dp4a kernels.

The scheduler, paged-KV, and continuous-batching design follow
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT). The compute path swaps
fp16 flash-attention for `fni8`'s int8 dp4a kernels, because on this hardware the fp16
tensor cores are firmware-disabled and dp4a is the fast path.

## What's feature-complete vs in-progress

**Supported for generation (decodes int8 end-to-end today):** **Qwen3** dense/untied,
**Qwen3-MoE**, and **Gemma3** (text) — full prefill + autoregressive decode on the dp4a
kernels, behind the OpenAI-compatible server. Only these three families are usable for
generation. Everything else is partial:

- **DeepSeek-V3 (MLA)** decodes correctly but in **fp16** (via `MLALatentCache`); the int8
  dp4a absorb kernel that would accelerate it is in progress (Track 2).
- **Qwen3-Next / 3.5 / 3.6, MiniMax, LFM2, GLM-4.5/4.6, Hunyuan** are **prefill only** —
  they build and run a prefill forward but **cannot yet generate**; full decode is pending
  the recurrent-state / linear-attention decode cache (in progress).
- **Multi-GPU (PP + MoE-EP)** and **MTP speculative decode** are scaffolded, not yet wired
  end-to-end.

See [Model support](#model-support) and the authoritative
[`fni8serve/models/COVERAGE.md`](fni8serve/models/COVERAGE.md) matrix.

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

### Run the server as a Docker image

Released as `ghcr.io/jajmangold/fni8-serve` (a lean runtime image built on the
prebuilt `fni8-built:sm70` base — fni8 kernels already baked in, no CUDA recompile).
The image ships **no weights**: mount your own `.fni8` checkpoint + tokenizer and point
`FNI8_MODEL`/`FNI8_TOKENIZER` at them (or pass the server's CLI flags directly).

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e FNI8_MODEL=/models/Qwen3-0.6B-fni8/Qwen__Qwen3-0.6B.b8.fni8 \
  -e FNI8_TOKENIZER=/models/Qwen3-0.6B-tok \
  -e FNI8_SERVED_MODEL_NAME=Qwen3-0.6B \
  ghcr.io/jajmangold/fni8-serve:latest
# equivalently, pass CLI flags after the image name (they override the env-var mode):
#   docker run ... ghcr.io/jajmangold/fni8-serve:latest \
#     --model /models/....fni8 --tokenizer /models/tok --served-model-name Qwen3-0.6B --port 8000
```

Then `curl http://localhost:8000/v1/models`. Build it locally with
`docker build -f docker/Dockerfile.runtime -t fni8-serve:local .`.

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

### Tools / tool_choice

`tools` is threaded straight into `apply_chat_template(tools=...)`, so formatting
comes from the model's own tools-aware chat template (same free-per-model
mechanism as the rest of chat formatting) -- nothing fni8-serve-specific happens
to the prompt.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]
resp = client.chat.completions.create(
    model="qwen3-8b.fni8",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
)
print(resp.choices[0].message.tool_calls[0].function)
```

With the default `tool_choice="auto"`, the model free-generates and its
raw-text tool-call emission is parsed by a per-model `fni8serve.tool_calls`
parser (`--tool-parser`, default `hermes` -- Qwen 2.5/3's native
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` tag convention).
`tool_choice="required"`, or a named choice
(`{"type": "function", "function": {"name": "get_weather"}}`), skips the parser
entirely: it builds a JSON schema from the tool's `parameters` and constrains
generation through the same XGrammar structured-output backend `response_format`
uses, so the result is guaranteed schema-valid.

## Offline batch inference

`fni8serve.batch.generate_batch(engine, requests)` runs a list of `BatchRequest`
(prompt ids, optionally its own `SamplingParams`, an optional `custom_id`) to
completion over `LLMEngine`'s continuous-batching scheduler, one call -- unlike
`LLMEngine.generate()`, each request may carry its own `SamplingParams`. Results
come back as a list of `BatchResult`, always in request order. Good for eval or
dataset runs without holding a connection open per request:

```python
from fni8serve import BatchRequest, generate_batch, SamplingParams

requests = [
    BatchRequest(tokenizer.encode(p), SamplingParams(max_tokens=64), custom_id=str(i))
    for i, p in enumerate(prompts)
]
for result in generate_batch(engine, requests):
    print(result.custom_id, tokenizer.decode(result.output_ids))
```

The API server exposes the same offline-batch idea over OpenAI's `/v1/batches`
JSONL-in/JSONL-out shape: upload a JSONL file of `/v1/chat/completions` or
`/v1/completions` request bodies via `/v1/files`, `POST /v1/batches` to run it,
then read the output file back. Every line -- including one carrying `tools` /
`tool_choice` -- is driven through the SAME `EngineWorker` continuous-batching
loop and the same request-building/tool-call helpers the streaming endpoints use
(`fni8serve.api.request_helpers`), so batch and live traffic share the engine
instead of contending for it, and get identical tool-calling behavior. Unlike
OpenAI's real (up-to-24h) batch window, this server has no background job queue --
`POST /v1/batches` runs the file to completion before returning, which is the
point for a short eval/dataset job, not a million-line one. Needs the `serve`
extra's `python-multipart` (for `/v1/files`' multipart upload).

```python
batch_input = client.files.create(file=open("prompts.jsonl", "rb"), purpose="batch")
batch = client.batches.create(input_file_id=batch_input.id, endpoint="/v1/completions",
                              completion_window="24h")
output = client.files.content(batch.output_file_id).text   # JSONL, one line per request
```

## Model support

Model support is a registry: a family is a `ModelConfig` plus a thin `models/<family>.py`
over shared layers, registered with `@register_model`. The engine never changes. See
[`fni8serve/models/COVERAGE.md`](fni8serve/models/COVERAGE.md) for current status.

- **Decodes int8 end-to-end** (full generation on the dp4a kernels — the supported set):
  Qwen3 dense/untied, Qwen3-MoE, Gemma3 (GQA with QK-norm and sliding window).
- **Decodes, but fp16 (not int8-accelerated yet):** DeepSeek-V3 (MLA). Full prefill +
  autoregressive decode run through the fp16 decompress-MLA path and `MLALatentCache`; the
  int8 dp4a absorb kernel is in progress (Track 2 in `fni8`).
- **Prefill only — cannot decode yet:** Qwen3-Next / 3.5 / 3.6 (hybrid Gated-DeltaNet plus
  full attention), MiniMax (lightning), LFM2 (short-conv plus attention), GLM-4.5/4.6,
  Hunyuan. These build through the registry and run a prefill forward on a ported fp16
  backend, but full autoregressive generation is pending the recurrent-state / linear-
  attention decode cache (in progress). Do not treat them as usable for generation yet.

## Status

- [x] `.fni8` zero-transform weight loader plus HF-to-`.fni8` converter
      (`fni8serve.convert`), including FP8-source models (DeepSeek, Hy3)
- [x] int8 / W4A8 dp4a GEMM in `fni8`; the Linear seam runs it (fp16 fallback for NF4/CPU)
- [x] Continuous-batching engine (scheduler, slot KV, `LLMEngine.generate()`)
- [x] Modular model registry, 8+ families build + prefill-validated (full int8 decode is
      Qwen3 / Qwen3-MoE / Gemma3 only; DeepSeek decodes in fp16; the rest are prefill-only)
- [x] Latent-KV decode cache (`MLALatentCache`) for DeepSeek MLA — fp16 decode
- [ ] Recurrent-state decode cache for the linear/DeltaNet/conv/lightning families
      (Qwen3-Next/3.5/3.6, LFM2, GLM, Hunyuan, MiniMax); until it lands they are prefill-only
- [x] Paged-KV block-table with int8 quantize-on-write, wired into the engine
- [x] OpenAI-compatible API server (`fni8serve.api`) with SSE streaming, plus a
      per-step logit-processor hook in the sampler for structured outputs / grammars
- [x] Structured outputs / grammars (`fni8serve.structured`): XGrammar behind the
      logit-processor hook, `response_format={type: json_schema}` + a `grammar`
      (GBNF/EBNF) extension
- [x] `tools`/`tool_choice` (`fni8serve.tool_calls`): tools formatted into the
      prompt via the chat template; `auto` parsed with a per-model text parser
      (`hermes`), `required`/named forced through the structured-output backend
- [x] Offline batch inference (`fni8serve.batch.generate_batch`) plus OpenAI
      `/v1/batches` + `/v1/files` (JSONL in/out), sharing the API server's
      continuous-batching `EngineWorker` and tool-calling request path
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
