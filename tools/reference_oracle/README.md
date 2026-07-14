<!-- Copyright (c) 2026, fni8 authors; SPDX-License-Identifier: BSD-3-Clause
     (dump_reference.cpp is adapted from llama.cpp's examples/eval-callback, MIT) -->
# llama.cpp / GGUF reference-oracle harness

A **reusable, arch-agnostic** trusted-reference harness for validating any
fni8 / fni8-serve **text-model** bringup against a from-scratch fp32 oracle.
"We couldn't trust the fp reference" should stop being a per-model fight.

It is the generalized successor to flint8's Qwen3.5-only reference dump (flint8
#105), which (a) validated two int8 kernels to cos 1.000000 on the real 9B and,
more importantly, (b) **caught a real bug** a synthetic self-test could not — a
gated-DeltaNet GQA head mapping that was interleave where ggml uses tile. The
synthetic reference shared the same wrong mapping, so it passed. **A real
from-scratch oracle beats a self-consistency test.** That is the whole point of
this tool: score your kernel/model against llama.cpp's own fp32 forward, not
against your own assumptions.

## What's here

| file | role |
|---|---|
| `dump_reference.cpp` | thin eval-callback over **prebuilt** llama.cpp libs; dumps per-layer fp32 activations for **any** GGUF-supported arch. No llama.cpp source patch. |
| `build_dump.sh` | compile it against a built llama.cpp checkout (records the commit) |
| `dump.sh` | run it on CPU (`-ngl 0`, fp32) for a fixed prompt → `manifest.tsv` + `<name>.f32` |
| `reconcile.py` | score a candidate's per-layer activations vs the oracle: **cos + relL1 + first-divergence layer** |

## End-to-end: any HF model → GGUF → dump → reconcile

```bash
# 0. one-time: build the dumper against an on-box llama.cpp (any recent commit)
export LLAMA_CPP_DIR=/home/josh/archives/llama-cpp-turboquant   # has build/bin/*.so
bash build_dump.sh

# 1. get a GGUF. Either download one from HF, or convert an HF checkpoint:
python3 $LLAMA_CPP_DIR/convert_hf_to_gguf.py <hf_model_dir> --outfile model.gguf --outtype f16
#   (the arch must be in llama.cpp's registry — see the audit for which fni8-serve
#    families qualify: docs/reference-oracle-audit.md)

# 2. dump the fp32 oracle (CPU, fixed prompt). PATTERNS is optional; unset = dump ALL.
MODEL=model.gguf OUT=./ref_data \
  PATTERNS="Qcur-,Kcur-,Vcur-,__fattn__-,attn_output-,ffn_out-,l_out-" \
  bash dump.sh

# 3. capture your fni8-serve model's per-layer activations on the SAME prompt +
#    tokenization (meta.txt records the exact token ids), save as an .npz
#    {tensor_name: fp32 array} or a dump dir in the same format, then:
python3 reconcile.py compare ref_data candidate.npz --map name_map.json
```

`reconcile.py compare` prints one row per matched tensor (sorted by layer) and,
if any tensor falls below the gate (`cos >= 0.999`, `relL1 <= 0.02` by default),
the **first-divergence layer** — the single number that says where your forward
stops matching the oracle. Also: `reconcile.py ls ref_data` lists dumped
tensors; `reconcile.py selfcheck ref_data` runs a best-effort internal
residual-algebra sanity (opt-in; naming is arch-specific, so `compare` is the
real check).

## Tensor-contract convention

- **On disk:** each tensor is `<sanitized_name>.f32`, raw little-endian fp32,
  **`ne0` fastest**, logical shape `[ne3, ne2, ne1, ne0]` (ggml row-major).
  `manifest.tsv` has one row per name (`name ne0 ne1 ne2 ne3 op file`),
  **last-wins** (a name that recurs in the graph settles to its final value —
  e.g. `Qcur-<il>` is post-RoPE). `meta.txt` records model, prompt, `add_bos`,
  the exact **token ids**, `n_tokens`, and the llama.cpp commit.
- **Candidate side:** a dump dir in the same format, or an `.npz` of
  `{name: fp32 array}`. Comparison **flattens** both sides, so only element
  ORDER must agree (both are ggml/row-major → a straight `ravel` lines up).
  fni8-serve uses its own tensor names, so pass a JSON name-map
  (`{candidate_name: ref_name}`); unmapped names are tried verbatim.
- **Tokenization must match.** Feed your model the token ids from `meta.txt`,
  not a re-tokenization — a tokenizer mismatch reads as a layer-0 divergence.

## Why it's trustworthy

- **CPU `-ngl 0` ⇒ every op is fp32**, including `FLASH_ATTN_EXT`,
  `GATED_DELTA_NET`, conv, MoE routing — higher precision than any fp16/int8 path.
- **No llama.cpp source patch:** we link its prebuilt `libllama`/`libggml` and
  install our own `cb_eval`. The graph is stock llama.cpp for whatever arch the
  GGUF declares; the pinned commit is stamped into `meta.txt`.
- **Byte-reproducible:** fixed prompt + `--temp 0` + `-ngl 0`. Re-dumping and
  reconciling gives cos 1.000000 / relL1 0.0 across every tensor (verified).

## Worked example (verified on-box)

Qwen3.5-9B (`qwen35`), prompt `"The capital of France is Paris."` (7 tokens),
CPU fp32, 217 activation tensors dumped (`Qcur/Kcur/Vcur/__fattn__/attn_output/
ffn_out/l_out/linear_attn_out/result_output`):

- **Determinism:** dump Q6_K twice → `compare` → **cos 1.000000, relL1 0.000000
  on all 217 tensors** ("no divergence").
- **Divergence localization** (proves the scorer catches a bug, #105's lesson):
  inject 5%·σ noise into `__fattn__-15` of a candidate → `compare` reports
  `__fattn__-15  cos 0.998766  relL1 0.111` and **`FIRST DIVERGENCE: layer 15`**,
  while layers 3/7/11/19/23/27/31 stay at cos 1.000000.

`tests/test_reference_oracle.py` unit-tests the scorer's metrics and
first-divergence logic (no GGUF/torch needed).

## Notes / limits

- **Text models only** — this oracle is llama.cpp's decoder graph. Image/video
  DiTs, VLMs, audio, and diffusion decoders need a different oracle (see the
  audit's bucket C).
- If a GGUF built for an older llama.cpp fails to load on a newer commit (tensor
  layout drift), re-convert from the HF checkpoint at the pinned commit. (Seen
  with an old Qwen3.5 Q4_K_XL missing `blk.32.ssm_conv1d.weight`.)
- `selfcheck` residual identities depend on arch-specific graph names (they hold
  for `qwen35`; LFM2 folds the residual under different names) — it is a bonus,
  never a substitute for `compare`.
