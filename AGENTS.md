<!-- SPDX-License-Identifier: MIT -->
# AGENTS.md — `fni8-serve`

`fni8-serve` is the Python serving runtime over the sibling `fni8` CUDA backend. It
owns request scheduling, model assembly, caches, distribution, conversion, and the
OpenAI-compatible API. It does not own CUDA kernels.

Read this file before changing code. Claims are capabilities only after they are on
`main` and their required correctness and performance gates pass.

## Hard rules

- The production target is the CMP 100-210 fleet: GV100/sm_70, 16 GB HBM2, PCIe
  1.0 x1, firmware-limited fp16 tensor cores, healthy integer `__dp4a`.
- Keep Python 3.12, CUDA 12.9, and `torch==2.10.0+cu129`. CUDA 13 drops sm_70.
- Do not add CUDA code here. Kernel work belongs in `fni8`, with its TDD/performance
  contract, then gets wired here in a separate PR.
- Softmax, LSE, sampling logits, and other numerically load-bearing reductions stay
  fp32. An accuracy-gated fp fallback is a correct outcome.
- PCIe 1.0 x1 makes frequent collectives expensive, but does not categorically rule out
  tensor parallelism. TP can work for large sparse MoEs such as 35B-A3B when active
  compute and expert structure amortize communication. Dense-model TP and FSDP remain
  suspect; require a topology-specific benchmark instead of enabling them by default.
  Prefer pipeline/expert boundaries and `fni8.transport` when they measure better.
- GPU measurements and model outputs must identify the exact checkpoint, precision,
  graph mode, batch/concurrency, device, and fleet caveat.
- Never use the pinned live-server GPU for tests, profiling, or destructive operations.

## Loading and format truth

- The merged loading path on `main` is authoritative. Do not call a branch-only native
  GGUF loader "current" in user documentation.
- GGUF-native loading is the target migration. `.fni8` remains supported until the
  prove-before-delete P5 gate is explicitly completed.
- Preserve format/ABI/layout negotiation and converter provenance. Never silently
  reinterpret a resident-weight layout.

## Architecture

```text
API -> LLMEngine -> Scheduler/Sequence -> ModelRunner -> model registry/layers -> fni8
                              |
                    paged KV / recurrent / MLA caches
```

- `fni8serve/engine/`: request lifecycle, scheduling, caches, CUDA graphs, drafters.
- `fni8serve/models/`: family-specific assembly over shared layers. Keep the engine
  model-agnostic; update `models/COVERAGE.md` honestly.
- `fni8serve/layers/`: the only normal seam to backend operators.
- `fni8serve/api/`: untrusted external input. Preserve authentication, bounds, SSRF,
  cancellation, and streaming cleanup.
- `convert.py` / loaders: round-trip the full model config and fail loudly on unknown
  tensor layouts or missing weights.

## Tests and workflow

Run the light lane in the prebuilt sm_70 image; this repo must not trigger a CUDA
recompile for ordinary Python changes:

```bash
docker run --rm --gpus all -e PYTHONPATH="$PWD" -v "$PWD:$PWD" -w "$PWD" \
  fni8-built:sm70 bash -lc \
  'pip install -e ".[dev,convert,serve,structured]" --no-build-isolation -q && pytest -q'
```

- Write the failing test first for behavior changes.
- Model changes need teacher-forced parity and stepwise decode/cache tests.
- Scheduler changes need concurrent/adversarial request tests, not one happy path.
- Quantized comparisons use cosine/SQNR/relative-L1 against an fp32 oracle; do not
  weaken a tolerance to pass.
- Performance PRs require a quiet-GPU artifact and must distinguish kernel time,
  end-to-end tok/s, TTFT, and graph/eager mode.
- Run `ruff` and Trailmark on the changed Python graph.

Keep one logical change per PR. Do not mix model support, engine refactors, converter
changes, and benchmark-baseline updates.
