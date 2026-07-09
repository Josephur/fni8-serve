# SPDX-License-Identifier: MIT
"""Zero-transform weight loader — reads the `.fni8` container from `fni8`.

The on-disk bytes ARE the resident dp4a layout, so loading is mmap + copy with no
dequant / repack / transpose. Supports rank-local PARTIAL loads via the shard index
(a PP stage or MoE expert loads only its tensors — the mmap faults in only those
pages), which is what makes weight loading tractable on the 250 MB/s fleet.
"""

from __future__ import annotations

from typing import Any

import torch

from fni8 import FQReader, QTensor, save_fni8  # the format lives in fni8


def load_fni8_checkpoint(
    path: str,
    *,
    device: str = "cuda",
    names: list[str] | None = None,
    shard: str | None = None,
) -> dict[str, QTensor]:
    """Load (part of) a `.fni8` checkpoint as {name: QTensor} on `device`.

    names : explicit tensor list, or None for all.
    shard : a key into the checkpoint's shard index, e.g. a PP-stage or expert id;
            loads only that shard's tensors. Mutually exclusive with `names`.
    """
    with FQReader(path) as r:
        if shard is not None:
            idx = r.shards
            if shard in idx.get("experts", {}):
                names = idx["experts"][shard]
            elif shard.startswith("pp:"):
                names = idx["pp_stages"][int(shard[3:])]
            else:
                raise KeyError(f"unknown shard {shard!r}; have {list(idx)}")
        names = names if names is not None else r.names
        return r.load_many(names, device)


def load_fni8_state_dict(
    path: str, *, device: str = "cuda", names: list[str] | None = None, shard: str | None = None
) -> dict:
    """Build-ready state dict: quantized tensors stay QTensor, `raw` tensors (norms,
    embeddings, router gate) are unwrapped to plain fp16 Tensors — exactly what the
    model builders expect (LinearW8A8 takes a QTensor; RMSNorm/Embedding take a
    Tensor). Feed straight into `build_model(cfg, state_dict)`."""
    loaded = load_fni8_checkpoint(path, device=device, names=names, shard=shard)
    out: dict = {}
    for name, qt in loaded.items():
        out[name] = qt.data if getattr(qt, "scheme", None) == "raw" else qt
    return out


def checkpoint_info(path: str) -> dict:
    """Header summary (arch, quant, tensor count, shard index) without loading data."""
    with FQReader(path) as r:
        return {
            "arch": r.header["arch"],
            "quant": r.header["quant"],
            "version": r.header["version"],
            "num_tensors": len(r.names),
            "shards": {k: len(v) for k, v in r.shards.items()},
            "meta": r.header.get("__meta__", {}),
        }


# ── training checkpoint save / load (issue #133) ──────────────────────────


def save_training_checkpoint(
    path: str,
    model_state_dict: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save model weights + optimizer state to a ``.fni8`` checkpoint.

    Each model tensor is stored as a ``raw`` QTensor (fp16).  Optimizer
    buffers are stored under flat names prefixed ``optimizer.state.*``.
    Training metadata (optimizer type, param groups) is embedded in the
    file header's ``__meta__`` dict.
    """
    from fni8serve.training.optimizer import serialize_optimizer_state

    opt_tensors, opt_meta = serialize_optimizer_state(optimizer)

    all_tensors: dict[str, QTensor] = {}
    for name, t in model_state_dict.items():
        all_tensors[name] = QTensor(t.contiguous().cpu(), None, scheme="raw")
    for name, t in opt_tensors.items():
        all_tensors[name] = QTensor(t.contiguous().cpu(), None, scheme="raw")

    full_meta: dict[str, Any] = dict(meta or {})
    full_meta["training"] = True
    full_meta.update(opt_meta)

    save_fni8(path, all_tensors, meta=full_meta)


def load_training_checkpoint(
    path: str,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Load model weights from a training ``.fni8`` checkpoint.

    Restores the model state dict (unwrapped to plain Tensors, matching the
    format of ``nn.Module.state_dict()``) and updates *optimizer* in place.
    Returns the model state dict.
    """
    from fni8serve.training.optimizer import deserialize_optimizer_state

    loaded = load_fni8_checkpoint(path, device=device)
    info = checkpoint_info(path)
    meta = info.get("meta", {})

    model_sd: dict[str, Any] = {}
    opt_flat: dict[str, torch.Tensor] = {}
    for name, qt in loaded.items():
        if name.startswith("optimizer.state."):
            opt_flat[name] = qt.data
        else:
            model_sd[name] = qt.data

    opt_state_dict = deserialize_optimizer_state(opt_flat, meta)
    optimizer.load_state_dict(opt_state_dict)

    return model_sd
