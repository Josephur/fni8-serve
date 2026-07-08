# SPDX-License-Identifier: MIT
"""bench_llm — measure prefill/decode tok/s, time-to-first-token, and peak VRAM for
a `.fni8` LLM through `LLMEngine` (int8 dp4a), plus per-tensor weight SQNR against
the original HF checkpoint where it's available locally.

Run inside the fni8 container, pinned to a free profiling-capable GPU (host idx
4/7/9/11/14 — never the counter-locked CMP cards). See tools/bench.sh for the
containerized wrapper that pins the device and reports fleet-load state.

    python3 tools/bench_llm.py /mnt/24tb/fni8-forge/weights/Qwen__Qwen3-8B.b8.fni8 \\
        --out-dir bench

Caveats (read before trusting a number):
  * There is currently no fp16 execution path in fni8-serve itself — every
    quantizable linear is routed through `to_qtensor` (see
    fni8serve/models/weights.py), so this harness cannot produce an in-repo
    fp16-vs-int8 *latency* delta. `--hf-dir` only buys weight-quantization SQNR
    (a static, load-time signal), not a decode-time fp16 baseline. A true
    fp16 latency/perplexity baseline needs a second reference implementation
    (e.g. a `transformers` fp16 forward) — tracked as follow-up work, not done here.
  * Numbers are only meaningful when the box is otherwise idle on the GPU you pin
    to; the fleet also runs quant/forge jobs, so always state the concurrent-load
    caveat next to any timing claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make fni8serve importable

import torch

from fni8serve import ModelConfig, checkpoint_info, is_supported, load_fni8_state_dict
from fni8serve.engine import LLMEngine, SamplingParams


def _name(path: str) -> str:
    return Path(path).stem


def _load_config(path: str) -> ModelConfig:
    meta = checkpoint_info(path)["meta"]
    cfg_dict = dict(meta["config"])
    cfg_dict["arch"] = meta.get("arch", cfg_dict.get("arch"))
    return ModelConfig(**cfg_dict)


def _weight_sqnr(fni8_path: str, hf_dir: str) -> dict[str, float] | None:
    """Per-tensor SQNR (dB) of the dequantized `.fni8` int8 weights against the
    original HF fp16/bf16 safetensors, when the raw checkpoint is available
    locally (e.g. still sitting in the forge staging dir). Returns None if no
    matching safetensors are found."""
    from safetensors import safe_open

    hf_files = sorted(Path(hf_dir).glob("*.safetensors"))
    if not hf_files:
        return None

    quantized = load_fni8_state_dict(fni8_path, device="cpu")
    out: dict[str, float] = {}
    for f in hf_files:
        with safe_open(str(f), framework="pt") as sf:
            for name in sf.keys():
                qt = quantized.get(name)
                if qt is None or getattr(qt, "scheme", None) != "per_row_i8":
                    continue  # raw (norm/embedding) tensor, int4 (differently packed), or missing
                orig = sf.get_tensor(name).float()
                deq = qt.data.float() * qt.scale.unsqueeze(-1)
                err = (orig - deq).pow(2).mean()
                sig = orig.pow(2).mean()
                if err > 0:
                    out[name] = float(10 * torch.log10(sig / err))
    return out or None


def bench_one(path: str, *, num_prompts: int, prompt_len: int, max_new_tokens: int,
              device: str = "cuda", hf_dir: str | None = None) -> dict:
    cfg = _load_config(path)
    if not is_supported(cfg.arch):
        return {"model": _name(path), "arch": cfg.arch, "skipped": "arch not registered"}

    torch.cuda.reset_peak_memory_stats(device)
    weights = load_fni8_state_dict(path, device=device)

    t_load0 = time.perf_counter()
    engine = LLMEngine(cfg, weights, device=device, max_num_seqs=max(num_prompts, 1),
                       max_len=prompt_len + max_new_tokens + 8)
    torch.cuda.synchronize(device)
    load_s = time.perf_counter() - t_load0

    # Synthetic prompts (real accuracy runs should tokenize real text instead —
    # this harness only measures throughput/latency/VRAM, not output quality).
    prompts = [[(i + j) % cfg.vocab_size for j in range(prompt_len)] for i in range(num_prompts)]
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
    ids = [engine.add_request(p, params) for p in prompts]

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    engine.step()                      # first scheduled step is prefill for a cold batch
    torch.cuda.synchronize(device)
    ttft_s = time.perf_counter() - t0
    prefill_toks = sum(len(p) for p in prompts)

    t1 = time.perf_counter()
    while engine.scheduler.has_work():
        engine.step()
    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - t1
    decode_toks = sum(len(engine._out[i].output_ids) - 1 for i in ids)  # -1: TTFT token

    result = {
        "model": _name(path),
        "arch": cfg.arch,
        "weight_bits": cfg.weight_bits,
        "num_prompts": num_prompts,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "load_s": load_s,
        "ttft_s": ttft_s,
        "prefill_tok_s": prefill_toks / ttft_s if ttft_s > 0 else None,
        "decode_tok_s": decode_toks / decode_s if decode_s > 0 else None,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }
    if hf_dir:
        sqnr = _weight_sqnr(path, hf_dir)
        if sqnr:
            result["weight_sqnr_db_mean"] = sum(sqnr.values()) / len(sqnr)
            result["weight_sqnr_db_min"] = min(sqnr.values())
    return result


def main():
    ap = argparse.ArgumentParser(description="Benchmark .fni8 LLMs through LLMEngine")
    ap.add_argument("checkpoints", nargs="+", help="one or more .fni8 file paths")
    ap.add_argument("--out-dir", default="bench")
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-dir", default=None,
                    help="dir of original HF safetensors for this checkpoint, for weight SQNR "
                         "(only meaningful for a single-checkpoint run)")
    ap.add_argument("--gpu-load-caveat", default=None,
                    help="free-text note on concurrent fleet load, stored alongside the result "
                         "(e.g. 'pinned to idx 11; idx 4/7/9 busy with forge quant jobs')")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ckpt in a.checkpoints:
        print(f"[bench] {ckpt}")
        try:
            r = bench_one(ckpt, num_prompts=a.num_prompts, prompt_len=a.prompt_len,
                          max_new_tokens=a.max_new_tokens, device=a.device,
                          hf_dir=a.hf_dir if len(a.checkpoints) == 1 else None)
        except Exception as e:  # noqa: BLE001 - keep going, record the failure
            r = {"model": _name(ckpt), "error": str(e)}
        if a.gpu_load_caveat:
            r["gpu_load_caveat"] = a.gpu_load_caveat
        results.append(r)
        (out_dir / f"{r['model']}.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps(r, indent=2))

    _write_table(results, out_dir / "results.md")


def _write_table(results: list[dict], out_path: Path) -> None:
    cols = ["model", "arch", "weight_bits", "prefill_tok_s", "decode_tok_s", "ttft_s",
            "peak_vram_gb", "weight_sqnr_db_mean"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        row = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = f"{v:.2f}"
            row.append(str(v) if v is not None else ("skipped" if "skipped" in r else "-"))
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
