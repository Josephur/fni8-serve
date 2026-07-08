# SPDX-License-Identifier: MIT
"""Zero-transform weight loader — reads the `.fni8` container from `fni8`.

The on-disk bytes ARE the resident dp4a layout, so loading is mmap + copy with no
dequant / repack / transpose. Supports rank-local PARTIAL loads via the shard index
(a PP stage or MoE expert loads only its tensors — the mmap faults in only those
pages), which is what makes weight loading tractable on the 250 MB/s fleet.
"""
from __future__ import annotations

from fni8 import FQReader, QTensor  # the format lives in fni8


def load_fni8_checkpoint(
    path: str, *, device: str = "cuda", names: list[str] | None = None,
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


def checkpoint_info(path: str) -> dict:
    """Header summary (arch, quant, tensor count, shard index) without loading data."""
    with FQReader(path) as r:
        return {
            "arch": r.header["arch"], "quant": r.header["quant"],
            "version": r.header["version"], "num_tensors": len(r.names),
            "shards": {k: len(v) for k, v in r.shards.items()},
            "meta": r.header.get("__meta__", {}),
        }
