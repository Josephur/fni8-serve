<!-- SPDX-License-Identifier: MIT -->
# Model coverage matrix

fni8-serve is **modular by construction**: a model family is a `ModelConfig` +
a thin `models/<family>.py` assembly over shared layers, registered with
`@register_model(...)`. The engine, runner, and kernels never change per family.
Support splits by **attention backend** — that's the axis that decides whether a
family runs on today's fni8 dp4a kernels or needs a new one.

Legend: ✅ concrete + GPU-tested · 🟡 framework/config ready, needs real weights ·
🟧 scaffold (registered, fp16-fallback) · ⛔ needs a new fni8 kernel.

| family | released | attention backend | MLP | status | notes |
|---|---|---|---|---|---|
| **Qwen3** dense | ✅ | GQA (full) | SwiGLU | ✅ | QK-norm pre-RoPE, explicit head_dim=128, θ=1e6 |
| **Qwen3-MoE** | ✅ | GQA (full) | top-k MoE (no shared) | ✅ | softmax→top-k→renorm; 128 experts / top-8 |
| **Gemma3** (text) | ✅ | GQA (full + **sliding**) | GeGLU | ✅ | 5-local:1-global, dual-θ RoPE, (1+w) norm, √d embed, scale=`qpas^-0.5` |
| **GLM-4.5/4.6** | ✅ | GQA + **partial RoPE** | MoE | 🟡 | partial-rotary in place; needs GLM weight-name map |
| **Hunyuan (HY3)** | partial | GQA + **cross-layer attn** | MoE + shared expert | 🟡 | shared-expert done; CLA = KV-share across layers (config flag) |
| **Qwen3-Next / Qwen3.5 / Qwen3.6** | ✅ | **hybrid: Gated DeltaNet (linear) + full** | ultra-sparse MoE + shared expert | ⛔ | full-attn + MoE + MTP covered; **DeltaNet linear-attn layers need a kernel** |
| **DiffusionGemma** | ✅ (post-cutoff) | **bidirectional** over canvas | GeGLU MoE | 🟡 | uses `attn_int8_fwd(causal=False)`; needs DiffusionDecodeStrategy (PR3) |
| **Gemma4 / gemma3n** | ✅ | GQA + AltUp/LAuReL/PLE/MatFormer | GeGLU | 🟧 | residual-mixing + per-layer-embeddings are new modules; settling upstream |
| **DeepSeek-V3/V4** | V3 ✅ | **MLA (latent KV)** | fine-grained MoE + shared | ⛔ | **MLA needs a kernel** (or decompress-to-MHA fallback); MoE+MTP covered |
| **MiniMax-Text** | ✅ | **lightning (linear) attn** hybrid | MoE | ⛔ | **linear-attn kernel** shared with Qwen3-Next DeltaNet |

## What's proven today (`tests/test_models.py`, GPU)

Registry → `build_model` → `ModelRunner` → prefill + greedy decode through the
fni8 dp4a GEMM + int8 attention kernels, for **Qwen3 dense, Qwen3 untied-head,
Qwen3-MoE, and Gemma3** (sliding-window + dual-θ RoPE + Gemma sandwich norms), plus
MoE routing math. Adding Qwen3/Gemma → **zero** engine changes: pure proof of the
modular seam.

## MTP (Qwen3-Next / DeepSeek-V3 style) — `models/mtp.py`

Shared embed + final-norm + LM head; per depth: two input RMSNorms + `fc(2H→H)` +
one decoder block. Draft `k` tokens, **verify in one causal forward** —
`fni8.attn_int8_verify` already ships that kernel (chain mask). Head + greedy draft
are built; the accept/verify loop lands with the engine (it needs the main model's
KV cache).

## The two kernel gaps (fni8 PRs, not serve code)

1. **Linear attention** (Gated DeltaNet / lightning): Qwen3-Next/3.5/3.6, MiniMax.
   A recurrent/chunked linear-attention kernel — fundamentally different from FA2.
   Until then those families fp16-fallback the linear layers (correct, slow).
2. **MLA — multi-head latent attention** (DeepSeek-V3/V4): compressed latent KV.
   Either a dedicated MLA kernel or decompress-latent→MHA and reuse the dp4a path
   (loses the KV-compression win). MoE + MTP for DeepSeek are already covered.

Diffusion decoding (DiffusionGemma / LLaDA) needs **no new kernel** — bidirectional
attention is `attn_int8_fwd(causal=False)` — only a `DiffusionDecodeStrategy`
(iterative masked denoising, prefix/block KV policy) in the engine.
