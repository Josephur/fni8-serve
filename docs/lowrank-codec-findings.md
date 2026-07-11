# Low-rank ("VAE-ish") wire codec — findings (issue #184)

**Verdict: rejected as a PP-boundary wire codec — but for an end-to-end reason,
not a per-boundary one.** On the *per-boundary* iso-quality axis (max compression
at a fidelity floor — the correct axis for a bandwidth-bound link) low-rank looks
like a large win. But that axis is **invalid for LM PP boundaries**: per-boundary
SQNR/cosine (0.99+) does not predict end-to-end token fidelity. A controlled E2E
test shows the residual stream's next-token-discriminative signal concentrates in
the *lowest*-variance directions — exactly what a low-rank projection discards
first — so even a single low-rank boundary collapses generation, while int8/int4
(bounded elementwise quant noise) keep it coherent.

> **Correction to the earlier draft.** The first pass dismissed low-rank as
> "int8 strictly dominates on fidelity at iso-ratio." That was the wrong axis:
> on the PCIe-1.0-x1 fleet (~250 MB/s, HBM:wire ≈ 3316:1) **wire bytes are the
> bottleneck**, so the metric is *max compression at an acceptable fidelity
> floor*, not *fidelity at a fixed ratio*. On the corrected axis low-rank wins
> per-boundary (see §1). It is rejected on §2 (E2E), which the earlier framing
> never measured.

All numbers below are on **Qwen3-1.7B-Base** (`d = 2048`, 28 layers), boundary at
`hidden_states[14]` (a representative 2-way PP split), held-out test tokens,
calibration-fit SVD basis. Verified in `fni8-serve-test` docker (GPU device 3).

## §1 — Iso-quality: max compression at a fidelity floor (per-boundary)

Shipped codecs on real boundary activations:

| codec | ratio | SQNR | cos |
| ----- | ----- | ---- | --- |
| int8 | 2.00× | 24.5 dB | 0.9982 |
| int4 | 3.76× | 12.4 dB | 0.9714 |
| int4-had | 3.76× | 20.3 dB | 0.9953 |
| nf4 | 3.76× | 15.1 dB | 0.9847 |

Low-rank sweep (int8 latent, 0.5 % raw channels), ratio = `2d / (r + n_raw·2)`
(basis is amortized calibration, shipped once):

| rank r | ratio | SQNR | cos |
| ------ | ----- | ---- | --- |
| 128 | 27.7× | 13.9 dB | 0.9796 |
| 160 | 22.8× | 15.0 dB | 0.9841 |
| 224 | 16.8× | 16.8 dB | 0.9895 |
| 288 | 13.3× | 18.4 dB | 0.9927 |
| 352 | 11.0× | 19.8 dB | 0.9947 |
| 400 | 9.75× | 20.4 dB | 0.9954 |

**Max compression at a quality floor** (this is the honest bandwidth-bound metric):

| fidelity floor | best shipped codec | best low-rank | low-rank advantage |
| -------------- | ------------------ | ------------- | ------------------ |
| SQNR ≥ 14 dB, cos ≥ 0.98 | int4-had 3.76× | **r=160, 22.8×** | **6.0× more** |
| SQNR ≥ 18 dB, cos ≥ 0.98 | int4-had 3.76× | **r=288, 13.3×** | **3.5× more** |
| SQNR ≥ 20 dB, cos ≥ 0.98 | int4-had 3.76× | **r=400, 9.75×** | **2.6× more** |
| SQNR ≥ 24 dB, cos ≥ 0.98 | int8 2.00× | none (caps ~22 dB) | int8 wins |

By this table alone low-rank is a big win at any floor below ~22 dB. That is what
the corrected axis shows, and it is why the codec deserved a proper E2E test
rather than an iso-ratio dismissal. (Exact figures vary a few tenths of a dB with
the calibration sample; reproduce with `bench/lowrank_activation_probe.py`.)

## §2 — E2E validation: the per-boundary axis is misleading for LMs

Method: install the codec as a real PP boundary via a forward hook on the
boundary decoder layer (round-trip its output), then greedy-generate and compare
to the un-hooked single-GPU reference — logit cosine, top-1 next-token agreement,
and exact-completion count. `xN` = codec installed at N boundaries (a 4-way PP has
3). Harness soundness is proven by the identity control.

| config | per-boundary fidelity | logit_cos | top-1 agree | exact |
| ------ | --------------------- | --------- | ----------- | ----- |
| **identity passthrough (control)** | lossless | 1.0000 | **1.000** | 4/4 |
| int8 ×1 | 24.5 dB | 0.9911 | **0.800** | 4/6 |
| int8 ×3 (4-way PP) | 24.5 dB | 0.9810 | 0.622 | 2/6 |
| int4-had ×1 | 20.3 dB | 0.9840 | 0.763 | 2/4 |
| **low-rank r=400 ×1** (int8 latent) | **20.4 dB / cos 0.9954** | 0.93 | **0.017** | **0/6** |
| low-rank r=256–768, fp16 latent (pure truncation) | cos ≥ 0.996 | ~0.96 | 0.075–0.094 | 0/4 |
| low-rank r=256–512, int8 per-column latent | — | ~0.96 | 0.075–0.088 | 0/4 |

Low-rank r=400 (10×, a per-boundary SQNR of **20.4 dB** and cos **0.9954** —
comfortably above the int4 gate) produces **degenerate, looping text** at a single
boundary ("The answer is. The answer is." / "the sun is a source of light. The sun
is a source of light."). Top-1 next-token agreement is **1.7 %**. int8 and int4-had,
at *lower* per-boundary cos, stay coherent (76–80 % top-1). Quantization is not the
cause — the fp16-latent (no quant) and per-column-int8 variants fail identically
(`bench/lowrank_e2e_probe.py` runs the pure-projection variant and reproduces this).

## §3 — Mechanism: the signal is in the low-variance tail

The failure is not gradual truncation; it is **structural**. Sweeping the
truncation rank (pure projection, no quantization) with a full-rank calibration
basis:

| projection rank r | % of d | top-1 agree |
| ----------------- | ------ | ----------- |
| 512 | 25 % | 0.075 |
| 1024 | 50 % | 0.325 |
| 1280–2040 | 62–99.6 % | **0.575 (plateau)** |
| 2047 (drops 1 direction) | 99.95 % | **0.625** |
| 2048 = identity | 100 % | 1.000 |

Dropping **a single bottom singular direction** of the calibration covariance
already costs 37 % of top-1 agreement, and the damage plateaus: keeping 99.6 % of
directions still only recovers 57.5 %. The residual stream's next-token signal
lives disproportionately in the **lowest-variance** directions — precisely the
ones a variance-ordered SVD basis truncates first. int8/int4 spread a bounded,
roughly-random error across *all* directions and so preserve those low-variance
readout directions; low-rank deletes a whole subspace and takes the readout with
it. This is why per-boundary SQNR/cos (which are dominated by the high-variance
massive-activation direction) completely fail to predict E2E behavior.

## §4 — Compute cost (moot, noted for completeness)

The intended tradeoff was projection matmul (down `d→r`, up `r→d`, i.e. `2·d·r`
MACs/token per boundary) against wire bytes saved on the 250 MB/s link — which on
a bandwidth-bound fleet would favor low-rank, since compute idles waiting on the
wire. This analysis is **moot**: the codec is rejected on correctness (§2), which
dominates any bandwidth/compute tradeoff. No point buying compression that breaks
the model.

## Disposition

- **Rejected** as a PP-boundary wire codec. `fni8serve/dist/lowrank.py` is retained
  as a mechanically-correct reference (deterministic, orthonormal, exact raw-channel
  passthrough) and is **not wired into the transport path**; the boundary codec
  stays int8-by-default with int4/int4-had/nf4 as the denser options.
- The per-boundary iso-quality win in §1 is real but **illusory for LMs** — the
  lesson is that any candidate boundary codec must be validated **end-to-end**
  (top-1 / generation coherence), not on per-boundary SQNR/cos alone.
- A future low-rank variant might survive if its basis were fit to preserve the
  LM-head-relevant (low-variance) directions rather than top variance — e.g. a
  task/gradient-weighted or Fisher-weighted projection. That is a research
  direction, out of scope here.

## Reproduce

- `bench/lowrank_activation_probe.py` — per-boundary iso-quality table (§1).
- `bench/lowrank_e2e_probe.py` — E2E generation with a low-rank PP boundary,
  the truncation-rank sweep, and the identity/near-identity controls (§2, §3).

Both load Qwen3-1.7B-Base and run under `fni8-serve-test` on one GPU (device 3).
