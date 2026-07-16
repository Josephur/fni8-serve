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
| GLM-4.5/4.6, Hunyuan, LFM2 | ✅ int8 decode (GQA); LFM2 short-conv stays torch (cheap) |
| Qwen3-Next / 3.5 / 3.6 (Gated-DeltaNet hybrid) | ✅ int8 decode both halves (GQA + DeltaNet kernel); DeltaNet **prefill** still torch |
| DeepSeek-V3/V4 (MLA) | ✅ int8 **absorb** decode (default on); MLA **prefill** is fp32 einsum by design |
| MiniMax-Text (lightning) | softmax half decodes int8; lightning half is torch — int8 prefill unwired, no lightning decode kernel yet |

Multi-GPU (pipeline + MoE-expert parallelism) and MTP speculative decode are wired but
still maturing. The PCIe 1.0 x1 link makes frequent dense-model collectives expensive,
so pipeline/expert boundaries with `fni8.transport` are the default. Tensor parallelism
is still viable for selected large sparse MoEs such as 35B-A3B when active compute and
expert structure amortize communication; every TP configuration needs a topology-specific
throughput measurement. FSDP-style inference remains a poor default on this fleet.

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
