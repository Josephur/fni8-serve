# SPDX-License-Identifier: BSD-3-Clause
"""GGUF-native Qwen3.5-9B decode benchmark — measures tok/s with CUDA graphs.

Bare-docker run (NEVER docker compose, NEVER GPU 4):
    docker run --rm --gpus '"device=11"' --entrypoint bash -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -v /srv/nvme-data/containers/projects/fused_ni8:/fused_ni8 \
        -v /srv/nvme-data/containers/projects/flint8_work:/flint8_work \
        -w /work -e PYTHONPATH=/work:/fused_ni8 \
        fni8-serve-test:latest bench/gguf_9b_bench.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# ── Step 0: install gguf if missing ─────────────────────────────────────────
try:
    import gguf  # noqa: F401
except ImportError:
    print("[setup] installing gguf-py ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf", "-q"])
    print("[setup] gguf installed.")

import torch

print(
    f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}, "
    f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
)

# ── Step 1: verify fni8.linear_q4k imports ──────────────────────────────────
import fni8

assert hasattr(fni8, "linear_q4k"), "fni8.linear_q4k NOT available — fused kernel missing"
print(f"[check] fni8.linear_q4k available (fused Q4_K dp4a kernel)")

from fni8serve.layers.linear import _FNI8_HAS_Q4K

print(f"[check] _FNI8_HAS_Q4K = {_FNI8_HAS_Q4K}")

# ── Step 2: load GGUF native ────────────────────────────────────────────────
GGUF_PATH = "/flint8_work/Qwen3.5-9B-UD-Q4_K_XL.gguf"
assert os.path.exists(GGUF_PATH), f"GGUF not found: {GGUF_PATH}"

from fni8serve.gguf_native import gguf_config, load_gguf_engine

t0 = time.perf_counter()
cfg = gguf_config(GGUF_PATH)
t_cfg = time.perf_counter() - t0
print(f"[load] gguf_config: {t_cfg:.2f}s")
print(f"[load] arch={cfg.arch}, layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}")
print(
    f"[load] heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}, "
    f"head_dim={cfg.resolved_head_dim()}"
)
print(
    f"[load] linear_attention={cfg.linear_attention}, "
    f"full_attn_interval={cfg.full_attention_interval}"
)
print(f"[load] num_experts={cfg.num_experts}, vocab={cfg.vocab_size}")

# Check CUDA graph support
print(f"[load] is_moe={cfg.is_moe()}, latent_attention={cfg.latent_attention}")
for i in range(min(5, cfg.num_hidden_layers)):
    print(f"[load] layer {i}: {cfg.attention_kind(i)}")

PYTORCH_ALLOC_CONF = "expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PYTORCH_ALLOC_CONF)

t0 = time.perf_counter()
engine = load_gguf_engine(
    GGUF_PATH,
    device="cuda",
    max_num_seqs=1,
    max_len=2048,
    spec_decode=False,
)
t_load = time.perf_counter() - t0
print(f"[load] engine built in {t_load:.2f}s")

# Report native vs fallback path
print(f"[load] native_types={'Q4_K' if _FNI8_HAS_Q4K else '(none — int8 fallback)'}")

# CUDA graph status
runner = engine.runner
gd = getattr(runner, "graphed", None)
if gd is not None:
    print(f"[graph] GraphedDecode supported={gd.supported}, reason={gd._unsupported_reason}")
else:
    print(f"[graph] no GraphedDecode on runner")

# ── Step 3: benchmark decode ────────────────────────────────────────────────
PROMPT_LEN = 256
NEW_TOKENS = 128

# Generate a synthetic prompt (repeat a common token to fill PROMPT_LEN tokens)
# Use token 1 (typically a valid token in most tokenizers)
prompt_ids = [1] * PROMPT_LEN

from fni8serve.engine.sequence import SamplingParams

print(f"\n[bench] prompt_len={PROMPT_LEN}, new_tokens={NEW_TOKENS}")

# Warmup: run a short generation to warm up CUDA graphs / allocations
print("[bench] warming up ...")
warmup_ids = prompt_ids[:32]
wid = engine.add_request(warmup_ids, SamplingParams(max_tokens=8, temperature=0.0))
while engine.scheduler.has_work():
    engine.step()
engine.forget(wid)
torch.cuda.synchronize()
print("[bench] warmup done")

# Actual benchmark — use generate() which handles prefill + decode correctly
torch.cuda.reset_peak_memory_stats()
t0_total = time.perf_counter()

outputs = engine.generate([prompt_ids], SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0))
output_ids = outputs[0] if outputs else []

torch.cuda.synchronize()
t_total = time.perf_counter() - t0_total
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

# ── Step 4: report ──────────────────────────────────────────────────────────
print(f"\n[debug] output_ids ({len(output_ids)} tokens): {output_ids[:30]}")

# Steady-state estimate: total_time / num_tokens
tok_per_sec = len(output_ids) / t_total if t_total > 0 and output_ids else 0

print(f"\n{'=' * 60}")
print(f"RESULTS: Qwen3.5-9B GGUF-native decode")
print(f"{'=' * 60}")
print(f"  Decode tok/s (end-to-end):    {tok_per_sec:.1f}")
print(f"  Target:                        69")
print(f"  vs target:                     {tok_per_sec / 69 * 100:.1f}%")
print(f"  Output tokens:                 {len(output_ids)}")
print(f"  Total time:                    {t_total:.2f}s")
print(f"  Peak VRAM:                     {peak_vram:.2f} GB")
print(f"  VRAM < 16GB:                   {'YES' if peak_vram < 16.0 else 'NO — OVER BUDGET'}")
print(f"  Native Q4_K path:              {'YES' if _FNI8_HAS_Q4K else 'NO (int8 fallback)'}")
print(f"  CUDA graphs:                   {'ON' if gd and gd.supported else 'OFF'}")

# Coherence check: decode output should be non-empty, non-repetitive
if output_ids:
    unique = len(set(output_ids))
    print(f"  Unique tokens:                 {unique}/{len(output_ids)}")
    coherent = unique > len(output_ids) * 0.1
    print(f"  Coherent:                      {'YES' if coherent else 'NO — likely garbage'}")
else:
    print(f"  Coherent:                      NO — empty output!")

# Verbatim output if short enough
if len(output_ids) <= 200:
    print(f"  Raw output ids: {output_ids}")
