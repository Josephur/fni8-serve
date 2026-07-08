# SPDX-License-Identifier: MIT
"""Scaffold smoke tests: the integration seams to fni8 (loader + Linear) work, and
the config enforces the fleet's parallelism rules. Needs `fni8` installed."""
import pytest
import torch

pytest.importorskip("fni8")
from fni8 import QTensor, save_fni8  # noqa: E402

from fni8serve import ServeConfig, load_fni8_checkpoint  # noqa: E402
from fni8serve.layers import LinearW8A8  # noqa: E402


def _i8(out, in_):
    w = torch.randn(out, in_)
    s = w.abs().amax(-1, keepdim=True) / 127
    q = torch.round(w / s).clamp_(-127, 127).to(torch.int8)
    return QTensor(q, s.squeeze(-1).float(), scheme="per_row_i8")


def _i4(out, in_, g):
    w = torch.randn(out, in_)
    wg = w.reshape(out, in_ // g, g)
    s = wg.abs().amax(-1, keepdim=True) / 7
    c = torch.round(wg / s).clamp_(-7, 7).reshape(out, in_).to(torch.int64)
    u = (c & 0xF).to(torch.int32)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)
    return QTensor(packed, s.squeeze(-1).float(), scheme="per_group_i4",
                   group_size=g, codebook="int4")


def test_config_forbids_tp_on_this_fleet():
    ServeConfig(model="x.fni8")                       # tp=1 ok
    with pytest.raises(ValueError, match="TP is not viable"):
        ServeConfig(model="x.fni8", tensor_parallel_size=2)


def test_roundtrip_load_and_linear_w8(tmp_path):
    path = str(tmp_path / "m.fni8")
    save_fni8(path, {"mlp.up.weight": _i8(128, 256)})
    w = load_fni8_checkpoint(path, device="cpu")
    assert "mlp.up.weight" in w
    lin = LinearW8A8(w["mlp.up.weight"])
    y = lin(torch.randn(4, 256))
    assert y.shape == (4, 128) and torch.isfinite(y).all()


def test_linear_w4_seam(tmp_path):
    path = str(tmp_path / "m.fni8")
    save_fni8(path, {"mlp.down.weight": _i4(64, 128, 32)})
    w = load_fni8_checkpoint(path, device="cpu")
    lin = LinearW8A8(w["mlp.down.weight"])
    assert lin.in_features == 128 and lin.out_features == 64
    y = lin(torch.randn(2, 128))
    assert y.shape == (2, 64) and torch.isfinite(y).all()


def test_shard_partial_load(tmp_path):
    path = str(tmp_path / "m.fni8")
    ts = {f"model.layers.{i}.w": _i8(32, 64) for i in range(4)}
    save_fni8(path, ts, shards={"pp_stages": [["model.layers.0.w", "model.layers.1.w"],
                                              ["model.layers.2.w", "model.layers.3.w"]]})
    w = load_fni8_checkpoint(path, device="cpu", shard="pp:1")
    assert set(w) == {"model.layers.2.w", "model.layers.3.w"}
