# SPDX-License-Identifier: MIT
"""forge — download HF models, quantize them to the resident `.fni8` format on the
archive disk, and purge the transient download so the drive never fills.

Per model: check free space -> snapshot_download to staging/ -> convert to
weights/<name>.fni8 -> delete staging/ -> record in MANIFEST.json. A crash still
purges its staging dir (atexit + per-model try/finally). LLMs go through
`fni8serve.convert`; diffusion DiTs quantize the transformer weights inline with the
same skip-list `ComfyUI-fni8` uses (norms/modulation/embedders stay fp16).

Runs inside the fni8 container (needs the built `fni8` + torch + safetensors +
huggingface_hub). See tools/forge.sh for the container wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make fni8serve importable

BASE = Path(os.environ.get("FORGE_BASE", "/mnt/24tb/fni8-forge"))
STAGING, WEIGHTS, LOGS = BASE / "staging", BASE / "weights", BASE / "logs"
MANIFEST = BASE / "MANIFEST.json"
MIN_FREE_GB = float(os.environ.get("FORGE_MIN_FREE_GB", "300"))
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def free_gb(p: Path) -> float:
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9


def _name(repo: str) -> str:
    return repo.replace("/", "__")


def _purge_stale_staging():
    """Remove any leftover staging/<name> dirs from a prior crash. An OOM SIGKILL
    (e.g. GLM-4.5-Air) kills the process outright, skipping forge_one's per-model
    `finally` cleanup, so a stale staging dir can otherwise sit on the archive disk
    (up to a full model's worth of shards) until someone notices."""
    if not STAGING.exists():
        return
    for d in STAGING.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print(f"[purge] removed stale staging dir {d.name}")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"models": {}}


def _save(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


def _set(repo: str, **kw):
    m = _manifest()
    m["models"].setdefault(repo, {})
    m["models"][repo].update(kw)
    _save(m)


def _repo_size_gb(repo: str) -> float:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, token=HF_TOKEN, files_metadata=True)
    total = sum((s.size or 0) for s in (info.siblings or [])
                if s.rfilename.endswith((".safetensors", ".bin")))
    return total / 1e9


def _download(repo: str, dest: Path, allow: list[str], subfolder: str | None = None):
    from huggingface_hub import snapshot_download
    patterns = allow if not subfolder else [f"{subfolder}/{p}" for p in allow] + ["*.json"]
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=patterns, token=HF_TOKEN)


def quant_llm(repo: str, bits: int, group: int) -> Path:
    from fni8serve.convert import convert_hf_to_fni8
    name = _name(repo)
    stage, out = STAGING / name, WEIGHTS / f"{name}.b{bits}.fni8"
    _download(repo, stage, ["*.safetensors", "config.json", "*.model", "tokenizer*", "*.txt"])
    convert_hf_to_fni8(str(stage), str(out), weight_bits=bits, group_size=group)
    return out


# ---- DiT quant (inline; mirrors ComfyUI-fni8's skip-list) ----
_DIT_SKIP = ("norm", "modulation", "adaln", "ada_ln", "pos_embed", "patch_embed",
             "_embed", "embedder", "time_in", "txt_in", "img_in", "vector_in",
             "guidance_in", "context_embedder", "final_layer", "proj_out")


def quant_dit(repo: str, subfolder: str, bits: int) -> Path:
    import torch
    from safetensors.torch import load_file

    from fni8 import QTensor, save_fni8
    from fni8.quant.core import quantize_int8_rowwise
    from fni8.quant.lowbit import quantize_lowbit

    name = _name(repo)
    stage, out = STAGING / name, WEIGHTS / f"{name}.dit.b{bits}.fni8"
    _download(repo, stage, ["*.safetensors", "config.json"], subfolder=subfolder)
    tdir = stage / subfolder if (stage / subfolder).exists() else stage
    sd: dict = {}
    for f in sorted(tdir.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    src_bf16 = any(w.dtype == torch.bfloat16 for w in sd.values())
    qsd: dict = {}
    for k, w in sd.items():
        w = w.detach().cpu()                        # keep NATIVE dtype (don't truncate bf16!)
        kl = k.lower()
        if w.dim() == 2 and w.shape[-1] % 4 == 0 and not any(s in kl for s in _DIT_SKIP):
            wf = w.float()                          # quantize from full precision
            if bits == 4 and w.shape[-1] % 128 == 0:
                codes, sc = quantize_lowbit(wf, 4, dim=-1, group_size=128)
                c = codes.to(torch.int64)
                packed = ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8)
                qsd[k] = QTensor(packed.contiguous(), sc.float().contiguous(),
                                 scheme="per_group_i4", group_size=128, codebook="int4")
            else:
                q, sc = quantize_int8_rowwise(wf)
                qsd[k] = QTensor(q.contiguous(), sc.squeeze(-1).float().contiguous(),
                                 scheme="per_row_i8")
        else:
            # Raw passthrough (norms/modulation/embedders): fp16 unless it overflows
            # (bf16 DiTs can have out-of-fp16-range tensors -> inf -> black images).
            w16 = w.to(torch.float16)
            raw = w.float() if (torch.isinf(w16).any() and not torch.isinf(w.float()).any()) else w16
            qsd[k] = QTensor(raw, None, scheme="raw")
    save_fni8(str(out), qsd, meta={"kind": "dit", "repo": repo, "bits": bits,
                                   "native_dtype": "bfloat16" if src_bf16 else "float16"})
    return out


def forge_one(repo: str, *, kind: str, bits: int, group: int, subfolder: str) -> bool:
    m = _manifest()
    if m["models"].get(repo, {}).get("status") == "done":
        print(f"[skip] {repo} already done"); return True
    stage = STAGING / _name(repo)
    log = (LOGS / f"{_name(repo)}.log").open("a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {repo}: {msg}"
        print(line); log.write(line + "\n"); log.flush()

    try:
        avail = free_gb(BASE)
        if avail < MIN_FREE_GB:
            say(f"SKIP — archive low ({avail:.0f}GB < {MIN_FREE_GB:.0f}GB floor)")
            _set(repo, status="skipped_disk"); return False
        try:
            need = _repo_size_gb(repo)
            say(f"~{need:.1f}GB download; {avail:.0f}GB free")
            if avail - need < MIN_FREE_GB:
                say("SKIP — would drop below floor"); _set(repo, status="skipped_disk"); return False
        except Exception as e:
            say(f"size probe failed ({e}); proceeding")
        _set(repo, status="downloading", started=time.time())
        say(f"quant {kind} bits={bits} ...")
        out = quant_dit(repo, subfolder, bits) if kind == "dit" else quant_llm(repo, bits, group)
        size = out.stat().st_size / 1e9
        _set(repo, status="done", kind=kind, bits=bits, out=str(out),
             out_gb=round(size, 3), finished=time.time())
        say(f"DONE -> {out.name} ({size:.2f}GB)")
        return True
    except Exception as e:
        say(f"FAILED: {e}\n{traceback.format_exc()}")
        _set(repo, status="failed", error=str(e)); return False
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True); say("purged staging")
        log.close()


def main():
    ap = argparse.ArgumentParser(description="forge: HF -> .fni8 with disk cleanup")
    ap.add_argument("action", choices=["one", "batch", "status"])
    ap.add_argument("target", nargs="?", help="repo id (one) or models.txt path (batch)")
    ap.add_argument("--kind", choices=["llm", "dit"], default="llm")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--subfolder", default="transformer")
    a = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    for d in (STAGING, WEIGHTS, LOGS):
        d.mkdir(exist_ok=True)

    if a.action == "status":
        m = _manifest()
        for repo, v in sorted(m["models"].items()):
            print(f"  {v.get('status','?'):14} {repo}  {v.get('out_gb','')}")
        print(f"archive free: {free_gb(BASE):.0f} GB")
        return

    if a.action == "one":
        ok = forge_one(a.target, kind=a.kind, bits=a.bits, group=a.group, subfolder=a.subfolder)
        sys.exit(0 if ok else 1)

    # batch: lines "repo[,kind[,bits]]", '#' comments
    _purge_stale_staging()
    for raw in Path(a.target).read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        repo = parts[0]
        kind = parts[1] if len(parts) > 1 else "llm"
        bits = int(parts[2]) if len(parts) > 2 else a.bits
        if free_gb(BASE) < MIN_FREE_GB:
            print(f"[halt] archive below floor; stopping batch before {repo}"); break
        forge_one(repo, kind=kind, bits=bits, group=a.group, subfolder=a.subfolder)


if __name__ == "__main__":
    main()
