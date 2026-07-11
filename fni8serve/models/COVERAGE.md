<!-- SPDX-License-Identifier: MIT -->
# Model coverage matrix

fni8-serve is **modular by construction**: a model family is a `ModelConfig` +
a thin `models/<family>.py` assembly over shared layers, registered with
`@register_model(...)`. The engine, runner, and kernels never change per family.
Support splits by **attention backend** — that's the axis that decides whether a
family runs on today's fni8 dp4a kernels or needs a new one.

Legend (**status** column): ✅ concrete + GPU-tested (full decode) · 🟩 registered +
prefill-tested (decode needs recurrent-state/latent cache) · 🟡 config ready, needs real
weights · 🟧 scaffold · ⛔ int8 accel needs a new fni8 kernel (fp16 backend works).

**The `released` column ≠ "decodes".** ✅ under `released` only means a `.fni8` checkpoint
exists / is published for the family. Whether the family actually **decodes int8 end-to-end
is governed solely by the `status` column** — only a ✅ status means full int8 decode today
(🟩 = prefill only; ⛔ = decodes on the fp16 backend, not int8-accelerated). Do not read a
released-✅ as "supported for generation."

| family | released | attention backend | MLP | status | notes |
|---|---|---|---|---|---|
| **Qwen3** dense | ✅ | GQA (full) | SwiGLU | ✅ | QK-norm pre-RoPE, explicit head_dim=128, θ=1e6 |
| **Qwen3-MoE** | ✅ | GQA (full) | top-k MoE (no shared) | ✅ | softmax→top-k→renorm; 128 experts / top-8 |
| **Gemma3** (text) | ✅ | GQA (full + **sliding**) | GeGLU | ✅ | 5-local:1-global, dual-θ RoPE, (1+w) norm, √d embed, scale=`qpas^-0.5` |
| **LFM2 / LFM2-MoE** | ✅ | hybrid **short-conv** + GQA | SwiGLU (w1/w3/w2) | 🟩 | `full_attn_idxs`; double-gated conv(k=3); QK-norm; sigmoid MoE router |
| **GLM-4.5/4.6** | ✅ | GQA + **partial RoPE 0.5** + QKV-bias | sigmoid MoE + shared | 🟩 | e_score_correction_bias select, routed_scaling 2.5, first-k-dense |
| **Hunyuan** (A13B) | ✅ | GQA + QK-norm | softmax MoE + shared | 🟩 | `mlp.gate.wg`, `shared_mlp`; CLA (Large) not modeled |
| **Qwen3-Next / Qwen3.5 / Qwen3.6** | ✅ | **hybrid: Gated DeltaNet (linear) + full** | ultra-sparse MoE + shared | 🟩⛔ | fp16 DeltaNet backend runs; int8 chunked kernel = Track 2 |
| **DeepSeek-V3/V4** | V3 ✅ | **MLA (latent KV)** | fine MoE + shared | ✅⛔ | fp16 decompress MLA runs prefill + decode (`MLALatentCache`); absorb int8 kernel = Track 2 |
| **MiniMax-Text** | ✅ | **lightning (linear)** + softmax hybrid | softmax MoE | 🟩⛔ | fp16 lightning backend runs; postnorm α/β scaling; int8 = Track 2 |
| **DiffusionGemma** | ✅ (post-cutoff) | **bidirectional** over canvas | GeGLU MoE | 🟡 | `attn_int8_fwd(causal=False)`; needs DiffusionDecodeStrategy |
| **Gemma4 / gemma3n** | ✅ | GQA + AltUp/LAuReL/PLE/MatFormer | GeGLU | 🟧 | residual-mixing + per-layer-embeddings are new modules |

**Registered + full autoregressive decode** (`tests/test_more_models.py`,
`test_engine.py`, `test_deepseek.py`): LFM2, GLM, Hunyuan, MiniMax, Qwen3-Next all
build through the registry, prefill on the fni8 dp4a kernels, AND decode token-by-token
through the `RecurrentStateCache` (Gated-DeltaNet / lightning recurrent state + the
short-conv trailing window carried across steps). Decode is validated two ways: the
stepwise output matches a single teacher-forced forward over prompt+generated
(`test_*_decode_matches_teacher_forced`), and the same tokens come out through the
continuous-batching engine (`test_engine_qwen3_next_hybrid_matches_runner`). The
recurrent state is keyed **per slot**, so several divergent-family sequences decode
**concurrently** through one engine without corrupting each other's scan
(`test_engine_qwen3_next_concurrent_recurrent_decode`) — that's what makes them serve,
not just single-request generate. DeepSeek decodes through `MLALatentCache`, and its
decode step runs the Track-2 int8 **absorb** kernel (`fni8.mla_decode_absorb_int8`),
which folds `W_UK`/`W_UV` so the O(N) per-step up-projection GEMM disappears.

Real divergent-family **checkpoints** now load end-to-end: the converter persists the
`extra` dict (each family's hybrid layer map — LFM2 `layer_types`, MiniMax
`attn_type_list`, Qwen3-Next linear head dims, DeepSeek `kv_lora_rank`) into the
`.fni8` meta, which the old dump silently dropped. LFM2 is verified from a real HF
checkpoint (convert → int8 `.fni8` → tokenize → 32-token decode). Qwen3-Next / MiniMax
real-checkpoint loading additionally needs `from_hf` to derive the
`linear_attention` / `full_attention_interval` scalar flags from the HF config (the
synthetic-config tests set them by hand) — a small `from_hf` follow-up, tracked
separately; the decode path and cache are complete.
Generative multimodal (Z-Image, Qwen-Image, LTX, Wan, Qwen3-TTS) → see the
diffusion-pipeline plan; the DiT backbone reuses these kernels, the pipeline is new.

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
