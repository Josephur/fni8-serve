# fni8-serve

A minimal **W8A8 (INT8 dp4a) inference server** for the NVIDIA Volta / CMP 100‑210
fleet — a serving layer over the [`fni8`](../fused_ni8) FlashAttention‑2 dp4a kernels.

The scheduler / paged‑KV / continuous‑batching design follows
[nano‑vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT); the compute path is
swapped from fp16 flash‑attention to `fni8`'s int8 dp4a kernels, because on this
hardware the fp16 tensor cores are firmware‑gimped and **dp4a is the fast path**.

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

## Status / roadmap

- [x] Scaffold + integration seams to `fni8` (this repo)
- [x] `.fni8` zero‑transform weight loader (`fni8serve/loader.py`)
- [x] Attention seam → `fni8` prefill/decode; Linear seam (int8/W4A8, fp16‑correct fallback)
- [ ] **int8 / W4A8 dp4a GEMM** kernel in `fni8` (the #1 dependency; Linear flips to it)
- [ ] **Paged‑KV** support in the `fni8` decode kernel (block‑table indirection)
- [ ] int8 **quantize‑on‑write** KV store kernel
- [ ] Port nano‑vllm engine (scheduler / block_manager / continuous batching)
- [ ] v0: **single‑GPU W8A8 serving** end‑to‑end (Qwen3)
- [ ] v1: multi‑GPU **PP + MoE‑EP** with `fni8.transport`

`v0` is single‑GPU (no interconnect problem) and is the near‑term milestone.

## License

MIT (see `LICENSE`) — this project adapts the nano‑vllm design (MIT). It depends on
`fni8`, which is BSD‑3‑Clause. See `NOTICE`.
