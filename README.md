# fni8-serve

An INT8 (W8A8) LLM inference server for the NVIDIA Volta / CMP 100-210 fleet, built over
the [`fni8`](https://github.com/jajmangold/fni8) dp4a kernels.

On this hardware the fp16 tensor cores are firmware-dead (~6.9 TFLOP/s) while INT8
`__dp4a` is healthy (~46 TOP/s), so the server runs in W8A8 rather than fp16 — the
opposite of a normal GPU's verdict. The engine (scheduler, paged KV, continuous
batching) follows [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm); the compute
path swaps fp16 flash-attention for `fni8`'s int8 kernels.

## Quickstart

```bash
pip install -e ".[serve]"
python -m fni8serve.api.server --model qwen3-8b.fni8 --tokenizer Qwen/Qwen3-8B
```

It speaks the OpenAI API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen3-8b", messages=[{"role": "user", "content": "hi"}])
```

Convert any HF checkpoint to `.fni8` with `tools/forge.sh`, or pull a pre-quantized one
from [huggingface.co/jajmangold](https://huggingface.co/jajmangold?search=fni8). On
DiT/prefill GEMM shapes the int8 path runs **1.4–2.4× faster** than fp16 here — a
fleet-specific number that doesn't transfer to a real V100.

## Model support

The engine is a registry — a family is a `ModelConfig` plus a thin `models/<family>.py`.
[`fni8serve/models/COVERAGE.md`](fni8serve/models/COVERAGE.md) is the source of truth;
the short version:

| Family | int8 decode (end-to-end) |
| --- | --- |
| Qwen3 dense/MoE, Gemma3 | ✅ full prefill + decode |
| Qwen3-Next / 3.5 (Gated-DeltaNet hybrid) | ✅ decode landed in fni8 v0.1.0-rc3 (fused graph-capturable kernels) |
| DeepSeek-V3 (MLA) | decodes, but in **fp16** — int8 absorb kernel pending |
| MiniMax, LFM2, GLM-4.5/4.6, Hunyuan | prefill only; decode pending |

Multi-GPU (pipeline + MoE-expert parallelism) and MTP speculative decode are wired but
still maturing. Tensor/FSDP parallelism is unusable here — the PCIe 1.0 x1 link is
~3,300× slower than HBM, so cross-device tokens go through `fni8.transport` compression
and stay on pipeline/expert boundaries only.

## A hardware note that bites people

Don't reflexively rewrite fp16 → bf16 for the rare fp-fallback matmuls. On this cuBLAS
(torch 2.10+cu129, real V100), bf16 GEMM is **2–3× slower** than fp16 at small batch.
Softmax, LSE, and `P·V` stay **fp32**; other fp fallbacks keep fp16 or step up to fp32 —
never blanket-convert to bf16. (See the #146 audit.)

## The fni8 family

- **[fni8](https://github.com/jajmangold/fni8)** — the dp4a kernels and `.fni8` format.
- **fni8-serve** (here) — this server + the HF→`.fni8` converter.
- **[ComfyUI-fni8](https://github.com/jajmangold/ComfyUI-fni8)** — the same kernels for
  diffusion DiTs in ComfyUI.

---

MIT (adapts the nano-vllm design, MIT; depends on `fni8`, BSD-3 — see `NOTICE`).
Contributors: read [AGENTS.md](AGENTS.md).
