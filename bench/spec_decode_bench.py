# SPDX-License-Identifier: MIT
"""Spec-decode net-speedup benchmark: off / MTP-only / n-gram-only / cascade, on a
STRUCTURED prompt (code/JSON — where n-gram shines) and PROSE (where MTP carries).

Reports per drafter: decode tok/s, acceptance-length (AL = mean tokens committed per
spec step), draft accept-rate, and greedy bit-identity vs plain decode.

Run in the fni8-serve test image on ONE free gpu (never 4/5/7/9/11/14 while profiling):
    docker run --rm --gpus '"device=7"' --entrypoint python3 -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -v /mnt/24tb:/mnt/24tb -w /work -e PYTHONPATH=/work \
        fni8-serve-test:latest bench/spec_decode_bench.py
"""
from __future__ import annotations

import os
import time

import torch

from fni8serve.engine.drafters import NgramDrafter
from fni8serve.engine.llm_engine import LLMEngine
from fni8serve.engine.sequence import SamplingParams
from fni8serve.loader import checkpoint_info, load_fni8_state_dict
from fni8serve.models.config import ModelConfig

FNI8 = os.environ.get("QWEN35_9B_FNI8", "/mnt/24tb/fni8-forge/weights/Qwen__Qwen3.5-9B.b4.fni8")
TOK = os.environ.get("QWEN35_9B_TOK", "/mnt/24tb/qwen35-08b/tok")
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "128"))
SPEC_K = int(os.environ.get("BENCH_SPEC_K", "6"))
MAXLEN = int(os.environ.get("BENCH_MAXLEN", "1024"))

STRUCTURED = (
    "Repeat this JSON exactly three times as a list:\n"
    '{"name": "widget", "price": 9.99, "tags": ["a", "b", "c"], "in_stock": true}'
)
PROSE = "Explain, in a short paragraph, why the sky appears blue during the day."


def encode(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ids[0] if ids and isinstance(ids[0], list) else ids


def configure(runner, mode, k):
    runner._drafter_mode = mode
    runner._spec_k = k
    runner._ngram = NgramDrafter(min_n=2, max_n=4, max_k=k) if mode in ("cascade", "ngram") else None
    # ngram-only: null out the MTP fallback so a miss proposes nothing (k effectively 0
    # that step); cascade keeps the MTP fallback. Toggle via a flag the runner reads.
    runner._ngram_only = mode == "ngram"


def run(eng, prompt, mode, k):
    r = eng.runner
    r._spec_enabled = mode != "off"
    if mode != "off":
        configure(r, mode, k)
    r.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
    params = SamplingParams(temperature=0.0, max_tokens=MAXTOK)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng.generate([list(prompt)], params)[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    st = dict(r.spec_stats)
    al = (len(out) / st["steps"]) if st["steps"] else 1.0
    acc = (st["accepts"] / st["drafts"]) if st["drafts"] else 0.0
    return out, len(out) / dt, al, acc, st


def main():
    from transformers import AutoTokenizer

    meta_cfg = dict(checkpoint_info(FNI8)["meta"]["config"])
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5")
    weights = load_fni8_state_dict(FNI8, device="cuda")
    eng = LLMEngine(cfg, weights, device="cuda", max_num_seqs=1, max_len=MAXLEN, enable_cuda_graph=False)
    tok = AutoTokenizer.from_pretrained(TOK)

    for label, text in (("STRUCTURED (JSON)", STRUCTURED), ("PROSE", PROSE)):
        prompt = encode(tok, text)
        ref, tps_off, _, _, _ = run(eng, prompt, "off", SPEC_K)
        print(f"\n=== {label} — prompt {len(prompt)} tok, max_new {MAXTOK} ===")
        print(f"  {'drafter':<12} {'tok/s':>8} {'speedup':>8} {'AL':>6} {'accept':>7}  identical")
        print(f"  {'off':<12} {tps_off:>8.2f} {'1.00x':>8} {'1.00':>6} {'-':>7}  -")
        for mode in ("mtp", "ngram", "cascade"):
            out, tps, al, acc, st = run(eng, prompt, mode, SPEC_K)
            ident = out == ref
            print(
                f"  {mode:<12} {tps:>8.2f} {tps / tps_off:>7.2f}x {al:>6.2f} {acc:>7.3f}  {ident}"
                f"   (steps={st['steps']} drafts={st['drafts']} acc={st['accepts']})"
            )


if __name__ == "__main__":
    main()
