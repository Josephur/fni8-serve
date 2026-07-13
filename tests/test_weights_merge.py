# SPDX-License-Identifier: MIT
"""Merge-helper state-dict contract (`fni8serve.models.weights`).

`qkv_weight`/`gate_up_weight` fuse projections on the output axis. They may optionally
POP their source rows to bound the merge transient on a card-filling load — but that is
opt-in (`consume_on_merge()`), because the DEFAULT must leave the caller's state dict
intact: many callers build a second model from the same dict (an engine + a reference
runner, an fp16 + an int8 model), and a destructive default deletes rows out from under
them (the `KeyError: '...q_proj.weight'` regression this guards against).
"""
from __future__ import annotations

import torch

from fni8serve.models.weights import consume_on_merge, gate_up_weight, qkv_weight

_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _attn_mlp_sd(prefix: str = "model.layers.0") -> dict:
    r = lambda a, b: torch.randn(a, b, device=_DEV, dtype=torch.float16) * 0.02  # noqa: E731
    H, hd, nh, nkv, inter = 64, 16, 4, 2, 128
    return {
        f"{prefix}.self_attn.q_proj.weight": r(nh * hd, H),
        f"{prefix}.self_attn.k_proj.weight": r(nkv * hd, H),
        f"{prefix}.self_attn.v_proj.weight": r(nkv * hd, H),
        f"{prefix}.mlp.gate_proj.weight": r(inter, H),
        f"{prefix}.mlp.up_proj.weight": r(inter, H),
    }


def test_merge_is_non_destructive_by_default():
    """Default merge INDEXES its sources — the state dict is untouched, so a second
    consumer can still read q/k/v/gate/up."""
    p = "model.layers.0"
    sd = _attn_mlp_sd(p)
    before = set(sd)

    qkv = qkv_weight(sd, f"{p}.self_attn")
    gate_up = gate_up_weight(sd, f"{p}.mlp")

    assert set(sd) == before  # nothing popped
    # Merged rows == sum of source rows on the output axis.
    assert qkv.data.shape[0] == sum(
        sd[f"{p}.self_attn.{n}_proj.weight"].shape[0] for n in ("q", "k", "v")
    )
    assert gate_up.data.shape[0] == (
        sd[f"{p}.mlp.gate_proj.weight"].shape[0] + sd[f"{p}.mlp.up_proj.weight"].shape[0]
    )

    # The exact failing scenario: build (merge) a SECOND time from the same dict.
    qkv2 = qkv_weight(sd, f"{p}.self_attn")
    assert qkv2.data.shape == qkv.data.shape


def test_consume_on_merge_pops_sources():
    """Under `consume_on_merge()` the sources are popped (freed as consumed) so the
    single-card load path bounds its transient; the merged weight is identical."""
    p = "model.layers.0"
    sd = _attn_mlp_sd(p)

    with consume_on_merge():
        qkv = qkv_weight(sd, f"{p}.self_attn")
        gate_up = gate_up_weight(sd, f"{p}.mlp")

    for n in ("q", "k", "v"):
        assert f"{p}.self_attn.{n}_proj.weight" not in sd
    for n in ("gate", "up"):
        assert f"{p}.mlp.{n}_proj.weight" not in sd
    assert qkv.data.shape[0] == (4 + 2 + 2) * 16
    assert gate_up.data.shape[0] == 2 * 128


def test_consume_flag_restored_after_context():
    """The flag is a scoped toggle — after the context, merges are non-destructive again
    (and it restores correctly even on an exception)."""
    p = "model.layers.0"
    try:
        with consume_on_merge():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    sd = _attn_mlp_sd(p)
    qkv_weight(sd, f"{p}.self_attn")
    assert f"{p}.self_attn.q_proj.weight" in sd  # not popped -> flag was restored
