#!/usr/bin/env python3
# Copyright (c) 2026, fni8 authors
# SPDX-License-Identifier: BSD-3-Clause
#
# Arch-agnostic reference-oracle reconciliation scorer.
#
# Given the trusted llama.cpp fp32 reference dump (dump_reference.cpp output) and
# an fni8 / fni8-serve model's per-layer activations captured on the SAME prompt
# + tokenization, emit per-tensor cosine-sim + relative-L1 and the
# FIRST-DIVERGENCE layer — the single number that tells a kernel/model port
# where its forward pass stops matching a from-scratch oracle.
#
# This is the generalized successor to flint8's verify_reference.py (#105). That
# script only did an internal residual-algebra self-consistency check (no
# external oracle, because the boxes had no torch). The DURABLE lesson from #105
# was the opposite: a real from-scratch oracle beats a self-consistency test —
# it caught a GQA head-mapping bug (interleave vs. tile) that the synthetic
# self-test structurally could not, because the synthetic reference shared the
# same wrong mapping. So the primary mode here is CROSS-ORACLE comparison
# (`compare`); the self-consistency residual check is kept as an opt-in bonus
# (`selfcheck`).
#
# ---------------------------------------------------------------------------
# Tensor-contract convention (how a candidate names its activations)
# ---------------------------------------------------------------------------
# A "candidate" is either
#   (a) another dump directory in the SAME on-disk format (manifest.tsv + <name>.f32
#       little-endian fp32, ne0 fastest, logical shape [ne3,ne2,ne1,ne0]), or
#   (b) an .npz whose keys are tensor names and values are fp32 arrays.
# Reconciliation matches candidate names to reference names. Because fni8-serve
# uses its own tensor names, pass a JSON name-map (candidate_name -> ref_name);
# any unmapped name is tried verbatim against the reference. Comparison is
# shape-agnostic: tensors are flattened, so only the element ORDER must agree
# (both are row-major / ggml ne0-fastest, so a straight ravel lines up).
#
# Usage:
#   python3 reconcile.py compare   REF_DIR CAND (--map map.json) (--cos 0.999)
#   python3 reconcile.py selfcheck REF_DIR (--layers N) (--tol 2e-4)
#   python3 reconcile.py ls        REF_DIR         # list dumped tensors
#
# CAND is a dump dir or an .npz. Exit code is nonzero if the first-divergence
# gate trips (compare) or an identity fails (selfcheck).
import argparse
import json
import os
import re
import sys

import numpy as np


# ------------------------------- loading -----------------------------------
def _load_dump(d):
    """Load a dump dir (manifest.tsv + .f32) -> {name: ndarray[ne3,ne2,ne1,ne0]}."""
    files, shapes = {}, {}
    man = os.path.join(d, "manifest.tsv")
    with open(man) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            name, ne0, ne1, ne2, ne3, _op, fn = p
            ne = (int(ne3), int(ne2), int(ne1), int(ne0))
            arr = np.fromfile(os.path.join(d, fn), dtype=np.float32)
            files[name] = arr.reshape(ne)
            shapes[name] = ne
    return files, shapes


def _load_candidate(path):
    """A candidate is a dump dir OR an .npz -> {name: ndarray}."""
    if os.path.isdir(path):
        return _load_dump(path)[0]
    if path.endswith(".npz"):
        z = np.load(path)
        return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    raise SystemExit(f"candidate must be a dump dir or .npz, got {path}")


# ------------------------------- metrics -----------------------------------
def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def rel_l1(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.abs(a - b).sum() / (np.abs(b).sum() + 1e-12))


_LAYER_RE = re.compile(r"-(\d+)\b")


def _layer_of(name):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


# ------------------------------- compare -----------------------------------
def cmd_compare(args):
    ref, _ = _load_dump(args.ref_dir)
    cand = _load_candidate(args.candidate)
    name_map = {}
    if args.map:
        with open(args.map) as f:
            name_map = json.load(f)

    rows = []
    for cname, arr in cand.items():
        rname = name_map.get(cname, cname)
        if rname not in ref:
            continue
        r = ref[rname]
        if r.size != arr.size:
            rows.append((cname, rname, _layer_of(rname), float("nan"),
                         float("nan"), f"SIZE {arr.size} vs {r.size}"))
            continue
        rows.append((cname, rname, _layer_of(rname), cos(arr, r),
                     rel_l1(arr, r), ""))

    if not rows:
        print("no candidate tensors matched the reference (check --map / names)")
        return 2

    rows.sort(key=lambda t: (t[2], t[1]))
    print(f"{'cand':32s} {'ref':28s} {'L':>3s} {'cos':>10s} {'relL1':>10s}  note")
    first_div = None
    for cname, rname, il, c, l1, note in rows:
        flag = ""
        bad = note or (not np.isnan(c) and (c < args.cos or l1 > args.rell1))
        if bad and first_div is None and il >= 0:
            first_div = (il, rname)
        if bad:
            flag = "  <-- DIVERGENCE"
        print(f"{cname[:32]:32s} {rname[:28]:28s} {il:3d} "
              f"{c:10.6f} {l1:10.6f}  {note}{flag}")

    print(f"\nmatched {len(rows)} tensors; "
          f"gate: cos>={args.cos}, relL1<={args.rell1}")
    if first_div is not None:
        print(f"FIRST DIVERGENCE: layer {first_div[0]}  ({first_div[1]})")
        return 1
    print("no divergence: candidate matches the oracle within gate")
    return 0


# ------------------------------ selfcheck ----------------------------------
def cmd_selfcheck(args):
    """Generic residual-stream self-consistency (best-effort, opt-in bonus).

    llama.cpp graphs expose per-layer `ffn_out-<il>` and `l_out-<il>`. Where the
    layer-input chain and a mixer-output tensor are also present under the usual
    names, we check `l_out == ffn_out + <attn_residual>` and the mixer residual.
    This proves a dump is a COHERENT forward (not stale/garbled) without any
    external oracle — but per #105 it is NOT a substitute for `compare`.
    """
    ref, shp = _load_dump(args.ref_dir)

    def hs(name):
        a = ref[name]
        return a.reshape(-1, a.shape[-1])

    def relerr(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    # candidate names for the layer-input, mixer-output, and post-mixer residual
    RESID = ["attn_residual", "ffn_inp", "attn_out_resid"]
    MIXER = ["attn_output", "linear_attn_out", "kqv_out", "attn_out"]
    fails = checked = 0
    for il in range(args.layers):
        lin = "model.input_embed" if il == 0 else f"l_out-{il-1}"
        lo_name = f"l_out-{il}"
        ffo_name = f"ffn_out-{il}"
        if lin not in ref or lo_name not in ref or ffo_name not in ref:
            continue
        resid_name = next((f"{r}-{il}" for r in RESID if f"{r}-{il}" in ref), None)
        if resid_name is None:
            continue
        lo, ffo, ar = hs(lo_name), hs(ffo_name), hs(resid_name)
        e2 = relerr(lo, ffo + ar)
        ok2 = e2 < args.tol
        fails += not ok2
        checked += 1
        print(f"L{il:02d} mlp-residual   relerr={e2:.2e}  {'ok' if ok2 else 'FAIL'}")

        mname = next((f"{m}-{il}" for m in MIXER if f"{m}-{il}" in ref), None)
        if mname is not None:
            mo, xin = hs(mname), hs(lin)
            e = relerr(ar, mo + xin)
            ok = e < args.tol
            fails += not ok
            checked += 1
            print(f"L{il:02d} mix-residual   relerr={e:.2e}  {'ok' if ok else 'FAIL'}")

    if "result_output" in ref:
        lg = ref["result_output"].ravel()
        print(f"\nresult_output: shape={shp['result_output']} "
              f"finite={bool(np.isfinite(lg).all())} argmax={int(lg.argmax())}")
    if checked == 0:
        print("\nno residual identities checkable (names differ for this arch); "
              "use `compare` against a candidate dump instead.")
        return 0
    print(f"\n{checked - fails}/{checked} residual identities hold (tol={args.tol})")
    return 1 if fails else 0


def cmd_ls(args):
    _, shp = _load_dump(args.ref_dir)
    for name in sorted(shp, key=lambda n: (_layer_of(n), n)):
        print(f"{name:40s} {shp[name]}")
    print(f"\n{len(shp)} tensors")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="cross-oracle cos/relL1 + first divergence")
    c.add_argument("ref_dir")
    c.add_argument("candidate", help="dump dir or .npz of candidate activations")
    c.add_argument("--map", help="JSON name-map: candidate_name -> ref_name")
    c.add_argument("--cos", type=float, default=0.999, help="cos gate (default 0.999)")
    c.add_argument("--rell1", type=float, default=0.02, help="relL1 gate (default 0.02)")
    c.set_defaults(fn=cmd_compare)

    s = sub.add_parser("selfcheck", help="internal residual-algebra sanity (opt-in)")
    s.add_argument("ref_dir")
    s.add_argument("--layers", type=int, default=64)
    s.add_argument("--tol", type=float, default=2e-4)
    s.set_defaults(fn=cmd_selfcheck)

    l = sub.add_parser("ls", help="list dumped tensors + shapes")
    l.add_argument("ref_dir")
    l.set_defaults(fn=cmd_ls)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
