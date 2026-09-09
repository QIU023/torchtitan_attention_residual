#!/usr/bin/env python3
"""Emit the PR-body tables from the tpfull matrix: per stream, one row per cell,
loss and grad norm at steps 1/3/10 with percent against the stream's parent tp=1 cell."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TAG", "tpfull")
import run_full as rf

LABEL = {"parent_tp1": "tp=1, parent", "tp1": "tp=1, this branch", "tp2_sp": "tp=2 SP on", "tp2_nosp": "tp=2 SP off", "tp4_sp": "tp=4 SP on", "tp4_nosp": "tp=4 SP off"}
STREAM = {"dp1": ("dp1 stream, 256 tokens per step", "dp1"), "dp2": ("dp2 stream, 512 tokens per step (a second dp rank reads other samples; compare within the stream)", "dp2"), "dp2ep2": ("dp2 x ep2 stream, 512 tokens per step", "dp2 x ep2")}

def cell_label(name, stream):
    core = name[len(stream) + 1:]
    be = "spmd_types" if core.endswith("_st") else "partial_dtensor"
    core = core[:-3]
    return f"{LABEL[core]}, {be}"

def f(v, ref, kind):
    if v is None: return "n/a"
    s = f"`{v:.6f}`" if kind == "loss" else f"`{v:g}`"
    if ref is None: return s
    if v == ref: return s + " (bitwise)"
    return s + f" (`{100*abs(v-ref)/ref:.3g}%`)"

names = [c[0] for c in rf.cells()]
R = {n: rf.parse(n) or {} for n in names}
for stream, (caption, _) in STREAM.items():
    ref = R[f"{stream}_parent_tp1_pd"]
    print(f"{caption}:\n")
    print("| cell | loss 1 | grad norm 1 | loss 3 | grad norm 3 | loss 10 | grad norm 10 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in names:
        if not n.startswith(stream + "_"): continue
        r = R[n]; cells = []
        for s in (1, 3, 10):
            v = r.get(s); rv = ref.get(s)
            cells += [f(v[0] if v else None, rv[0] if rv else None, "loss"), f(v[1] if v else None, rv[1] if rv else None, "gn")]
        print(f"| {cell_label(n, stream)} | " + " | ".join(cells) + " |")
    print()
