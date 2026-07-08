# SPDX-License-Identifier: MIT
"""Publish `.fni8` quants to the Hugging Face Hub as **linked** quantizations.

The point of the linkage: a repo whose model-card frontmatter declares
``base_model: <parent>`` + ``base_model_relation: quantized`` shows up under the
parent model's **"Quantizations"** section on HF (the other relations are
``finetune``/``adapter``/``merge``). So every upload carries a generated model card,
not just the raw ``.fni8`` bytes.

License hygiene: we do NOT guess licenses — `publish_one` reads the *parent's* own
license + gated flag from the Hub and (a) refuses to publicly re-host anything
non-commercial or gated, flagging it for a human decision, and (b) inherits the
parent's license tag onto our card otherwise.

`model_card` / `is_restricted` / `repo_name` are pure and unit-tested; `publish_one`
wraps them with the Hub IO.
"""
from __future__ import annotations

from pathlib import Path

# License tags / substrings that forbid (or gate) public re-hosting of a derivative.
# Matched case-insensitively as substrings of the parent's `license` tag.
NONCOMMERCIAL_MARKERS = (
    "-nc", "-nc-", "noncommercial", "non-commercial", "cc-by-nc", "cc-nc",
    "flux-1-dev-non-commercial", "research", "research-only", "rail",  # *RAIL carry use-restrictions
)


def is_restricted(license_tag: str | None, gated) -> bool:
    """True if the parent forbids/gates public redistribution, so we must NOT create
    a public quant repo for it (hold for a human decision).

    `gated` is HF's flag: False / None (open) vs 'auto' / 'manual' / True (gated).
    A missing license is treated as restricted — never publish something whose terms
    we can't read."""
    if gated not in (False, None, "", "false"):
        return True
    if not license_tag:
        return True
    lt = license_tag.strip().lower()
    if lt in ("unknown", "other"):          # `other` = a custom license we can't vet
        return True
    return any(m in lt for m in NONCOMMERCIAL_MARKERS)


def repo_name(parent_repo: str, hf_user: str) -> str:
    """Our repo id for a parent's quant: ``<hf_user>/<parent-basename>-fni8``.
    e.g. ``black-forest-labs/FLUX.1-dev`` -> ``jajmangold/FLUX.1-dev-fni8``.
    Bit-width is NOT in the repo name — both int8 and int4 files live in one repo."""
    base = parent_repo.split("/")[-1]
    return f"{hf_user}/{base}-fni8"


def parse_fni8_name(filename: str) -> dict | None:
    """Recover (parent_repo, kind, bits) from a forge output filename. This is the
    inverse of forge's naming (`repo.replace('/','__')` + `.dit`? + `.b{bits}.fni8`),
    so publishing can group every bit-width of a model into its one repo by scanning
    the weights dir — no dependence on the single-status manifest.

    ``Qwen__Qwen3-0.6B.b8.fni8``            -> Qwen/Qwen3-0.6B      llm  8
    ``Qwen__Qwen-Image.dit.b8.fni8``        -> Qwen/Qwen-Image      dit  8
    Returns None if the name doesn't match."""
    if not filename.endswith(".fni8"):
        return None
    stem = filename[: -len(".fni8")]                     # drop .fni8
    parts = stem.split(".")
    if not parts[-1].startswith("b") or not parts[-1][1:].isdigit():
        return None
    bits = int(parts[-1][1:])
    rest = parts[:-1]
    kind = "llm"
    if rest and rest[-1] == "dit":
        kind = "dit"
        rest = rest[:-1]
    name = ".".join(rest)                                # base names may contain dots
    parent_repo = name.replace("__", "/")
    if "/" not in parent_repo:
        return None
    return {"parent_repo": parent_repo, "kind": kind, "bits": bits}


def _fni8_tags(kind: str) -> list[str]:
    tags = ["fni8", "int8", "w8a8", "dp4a", "volta", "sm_70", "quantized"]
    if kind == "dit":
        tags += ["diffusion", "text-to-image", "comfyui"]
    return tags


def _scheme(bits: int) -> str:
    return "int4 per-group W4A8 (int8 activations)" if bits == 4 else "int8 per-row W8A8"


def model_card(
    parent_repo: str,
    kind: str,
    quants: list[dict],
    *,
    license_tag: str | None,
    pipeline_tag: str | None = None,
    native_dtype: str | None = None,
) -> str:
    """Render README.md for a repo that may hold SEVERAL quant files (e.g. int8 +
    int4). `quants` is ``[{"bits": int, "filename": str, "gb": float}, ...]``.

    The frontmatter's ``base_model`` + ``base_model_relation: quantized`` file the
    repo under the parent's Quantizations on the Hub. A **Files** table + per-bits
    tags make the precision of each file unmistakable (you can tell 4- vs 8-bit at a
    glance)."""
    bits_present = sorted({q["bits"] for q in quants})
    fm: list[str] = ["---"]
    fm.append(f"base_model: {parent_repo}")
    fm.append("base_model_relation: quantized")
    if license_tag:
        fm.append(f"license: {license_tag}")
    if pipeline_tag:
        fm.append(f"pipeline_tag: {pipeline_tag}")
    fm.append("tags:")
    tags = list(_fni8_tags(kind)) + [f"{b}-bit" for b in bits_present]
    for t in tags:
        fm.append(f"  - {t}")
    fm.append("---")

    dt = (f"\n- **Source dtype:** `{native_dtype}` (raw tensors kept fp16, upcast to "
          f"fp32 only on overflow).") if native_dtype else ""
    label = " + ".join(f"int{b}" for b in bits_present)
    rows = "\n".join(
        f"| `{q['filename']}` | **int{q['bits']}** | {_scheme(q['bits'])} | "
        f"{q['gb']:.2f} GB |"
        for q in sorted(quants, key=lambda q: q["bits"])
    )
    body = f"""
# {parent_repo.split('/')[-1]} — fni8 ({label} dp4a)

Quantization of [`{parent_repo}`](https://huggingface.co/{parent_repo}) to the
**`.fni8`** resident format for the [**fni8**](https://github.com/jajmangold/fni8)
W8A8 DP4A kernels on **NVIDIA Volta (sm_70)** — Tesla V100 / CMP 100-210.

## Files (precision per file)

| file | precision | scheme | size |
|------|-----------|--------|------|
{rows}

Pick the file for the precision you want — **`.b8.` = int8 (W8A8)**, **`.b4.` = int4
(W4A8)**. Weights are fp32-scaled and stored in the resident dp4a VRAM layout (loads
with no dequant/repack).{dt}

- **Why dp4a:** sm_70 has no int8 tensor cores; the contraction runs on the
  `__dp4a` CUDA-core intrinsic. On the CMP-100-210 fleet (firmware-gimped fp16
  tensor cores) dp4a is the fast path, not a compromise.

## Use it

- **Kernels:** <https://github.com/jajmangold/fni8>
- **LLM serving:** <https://github.com/jajmangold/fni8-serve>
- **ComfyUI (diffusion DiTs):** <https://github.com/jajmangold/ComfyUI-fni8>

This is a derivative quantization; its license follows the parent model above.
"""
    return "\n".join(fm) + "\n" + body


def parent_meta(parent_repo: str, token: str | None) -> dict:
    """Fetch the parent's license tag, gated flag, and pipeline_tag from the Hub."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(parent_repo, token=token)
    cd = info.card_data or {}
    lic = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
    return {
        "license": lic,
        "gated": getattr(info, "gated", False),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
    }


def publish_repo(
    parent_repo: str,
    quant_files: list,
    *,
    hf_user: str,
    token: str | None,
    kind: str = "llm",
    native_dtype: str | None = None,
    private: bool = False,
    skip_restricted: bool = False,
) -> dict:
    """Publish ALL quant files of one parent into its single `<name>-fni8` repo, then
    write ONE model card that enumerates them (so 4- vs 8-bit is unmistakable).

    `quant_files` is ``[(bits, path), ...]``. Creates the repo, uploads every file,
    and writes the card last. Publishes every model by default; the parent's license
    is inherited. `skip_restricted=True` holds non-commercial/gated parents."""
    from huggingface_hub import HfApi

    meta = parent_meta(parent_repo, token)
    if skip_restricted and is_restricted(meta["license"], meta["gated"]):
        return {"status": "held_restricted", "parent": parent_repo,
                "license": meta["license"], "gated": meta["gated"]}

    rid = repo_name(parent_repo, hf_user)
    api = HfApi()
    api.create_repo(rid, token=token, private=private, exist_ok=True, repo_type="model")

    quants = []
    for bits, path in sorted(quant_files):
        fname = Path(path).name
        api.upload_file(path_or_fileobj=str(path), path_in_repo=fname, repo_id=rid,
                        token=token, commit_message=f"upload {fname}")
        quants.append({"bits": bits, "filename": fname, "gb": Path(path).stat().st_size / 1e9})

    card = model_card(parent_repo, kind, quants, license_tag=meta["license"],
                      pipeline_tag=meta["pipeline_tag"], native_dtype=native_dtype)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=rid, token=token, commit_message="model card (linked quant)")
    return {"status": "published", "repo": rid, "license": meta["license"],
            "bits": [q["bits"] for q in quants], "url": f"https://huggingface.co/{rid}"}
