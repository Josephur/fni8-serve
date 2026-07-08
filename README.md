# fni8-serve

A minimal **W8A8 (INT8 dp4a) inference server** for the NVIDIA Volta / CMP 100‑210
fleet — a serving layer over the [`fni8`](../fused_ni8) FlashAttention‑2 dp4a kernels.

The scheduler / paged‑KV / continuous‑batching design follows
[nano‑vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT); the compute path is
swapped from fp16 flash‑attention to `fni8`'s int8 dp4a kernels, because on this
hardware the fp16 tensor cores are firmware‑gimped and **dp4a is the fast path**.

## Pre-quantized models on Hugging Face

Ready-to-serve `.fni8` weights (int8 **and** int4 in each repo) live under
**[huggingface.co/jajmangold](https://huggingface.co/jajmangold?search=fni8)** — each
is published as a *linked quantization*, so it also appears under its source model's
**Quantizations** tab. Produce more with `tools/forge.sh` (download → quantize →
`forge publish`).

## Why this exists (the hardware, honestly)

The deployment fleet is **CMP 100‑210** (GV100 silicon, 16 GB HBM2 @ 829 GB/s,
**PCIe 1.0 ×1 ≈ 250 MB/s**). Two facts drive every design choice:

1. **fp16/TF32 tensor cores are firmware‑gimped (~6.9 TFLOP/s).** INT8 `__dp4a` on the
   CUDA cores is healthy (~46 TOP/s). So we serve in **W8A8**, not fp16 — the opposite
   of a normal GPU's SOTA verdict. `fni8` is the kernel that does this.
2. **The interconnect is ~3,300× slower than HBM.** Tensor/FSDP parallelism is dead
   here (seconds per token); only **pipeline** and **MoE expert** parallelism survive,
   with the tokens on the wire compressed by `fni8.transport`. See
   `fused_ni8/utils/docs/transport-compression.md`.

## Architecture (what we keep vs swap vs replace)

| nano‑vllm piece | here |
| --- | --- |
| engine: scheduler, block_manager, sequence, runner | **keep** (port) — the serving loop |
| `layers/attention.py` (flash‑attn fp16) | **swap** → `fni8` int8 prefill + decode, int8 paged KV |
| `store_kvcache` Triton kernel | **adapt** → quantize‑on‑write into an int8 paged cache |
| `layers/linear.py` (fp16) | **swap** → int8 / **W4A8** dp4a GEMM (via `fni8`) |
| layernorm / rotary / activation / embed / sampler | **keep** (torch) |
| tensor parallelism (torch.distributed) | **replace** → PP + MoE‑EP + `fni8.transport` (multi‑GPU) |
| GGUF / safetensors loader | **replace** → `.fni8` zero‑transform mmap loader |

## Weights: the `.fni8` format (no GGUF)

Weights load via `fni8`'s `.fni8` container: the on‑disk bytes **are** the resident
dp4a layout, so loading is `mmap + cudaMemcpy` with **zero dequant/repack**. It carries
int8 (`per_row_i8`) and **4‑bit** (`per_group_i4`, int4/NF4) weights, fp32 scales, baked
Hadamard/smoothing flags, and a shard index for rank‑local partial loads. 4‑bit weights
halve the footprint (2× model capacity in 16 GB) and speed weight‑bandwidth‑bound decode;
they unpack to int8 in‑kernel for dp4a (sm_70 has no int4 matmul).

## Quickstart

```bash
# 1. Convert an HF checkpoint to .fni8 (int8 or 4-bit weights, resident dp4a layout)
python -m fni8serve.convert  /path/to/Qwen3-8B  qwen3-8b.fni8  --bits 8
```

```python
# 2. Serve it — arch is auto-detected from the checkpoint; continuous batching.
from fni8serve import ModelConfig, LLMEngine, SamplingParams, load_fni8_state_dict, checkpoint_info

cfg = ModelConfig.from_hf(checkpoint_info("qwen3-8b.fni8")["meta"]["config"])
engine = LLMEngine(cfg, load_fni8_state_dict("qwen3-8b.fni8"))
out = engine.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=32))
```

Model support is a **registry**: a family is a `ModelConfig` + a thin
`models/<family>.py` over shared layers, `@register_model`-ed — the engine never
changes. Registered today (all build + prefill through the dp4a kernels; see
[`fni8serve/models/COVERAGE.md`](fni8serve/models/COVERAGE.md)):

- **Qwen3**, **Qwen3-MoE**, **Gemma3** (full decode) — GQA + QK-norm / sliding-window.
- **DeepSeek** (MLA), **Qwen3-Next / 3.5 / 3.6** (hybrid Gated-DeltaNet + full attn),
  **LFM2** (short-conv + attn), **GLM-4.5/4.6**, **Hunyuan**, **MiniMax** (lightning) —
  the divergent-attention families, on ported fp16 backends (int8 accel is the
  DeltaNet/MLA kernel track in `fni8`).

## Status / roadmap

- [x] `.fni8` zero‑transform weight loader + HF→`.fni8` converter (`fni8serve.convert`)
- [x] **int8 / W4A8 dp4a GEMM** in `fni8`; the Linear seam runs it (fp16 fallback for NF4/CPU)
- [x] Continuous-batching **engine** (scheduler + slot KV + `LLMEngine.generate()`)
- [x] Modular **model registry** — 8+ families (Qwen3/-MoE, Gemma3, DeepSeek, Qwen3-Next,
      LFM2, GLM, Hunyuan, MiniMax); build + prefill validated
- [ ] Recurrent-state / latent **decode caching** for the linear/DeltaNet/MLA families
- [ ] **Paged‑KV** (block‑table) + int8 **quantize‑on‑write** in the `fni8` decode kernel
- [ ] int8 dp4a **DeltaNet + MLA** kernels in `fni8` (the divergent-attention accel)
- [ ] multi‑GPU **PP + MoE‑EP** with `fni8.transport`

## The fni8 family

Part of a three-repo stack on the same sm_70 fleet:
[`fni8`](https://github.com/jajmangold/fni8) (the dp4a kernels + `.fni8` format) ·
**`fni8-serve`** (this — LLM serving) ·
[`ComfyUI-fni8`](https://github.com/jajmangold/ComfyUI-fni8) (diffusion DiTs in ComfyUI).

## License

MIT (see `LICENSE`) — this project adapts the nano‑vllm design (MIT). It depends on
`fni8`, which is BSD‑3‑Clause. See `NOTICE`.
