"""Rank 0's traced step in the DEP runs: per-action GPU compute of the tower stage and the text
stage, compute outside any PP action (encodes run ahead), and the idle gaps between actions.

Usage: python ana_dep_trace.py <trace_dir containing rank0_trace.json[.gz]> [tower_stage]
"""
import glob
import gzip
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))

d = sys.argv[1]
tower = int(sys.argv[2]) if len(sys.argv) > 2 else 0
path = (glob.glob(os.path.join(d, "**", "rank0_trace.json*"), recursive=True) or [None])[0]
if path is None:
    sys.exit(f"no rank0 trace under {d}")
tr = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
ev = tr.get("traceEvents", tr)
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def inter_len(a, s, e):
    return sum(max(0, min(y, e) - max(x, s)) for x, y in a)


comp = union([(k["ts"], k["ts"] + k["dur"]) for k in kern if "nccl" not in k["name"].lower()])
busy = union([(k["ts"], k["ts"] + k["dur"]) for k in kern])
act = re.compile(r"^PP:(\d+)(F|B|I|W)(\d+)$")
acts = sorted(
    [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
     if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and act.match(e.get("name", ""))]
)
if not acts:
    sys.exit("no PP annotations")
lo, hi = acts[0][0], max(a[1] for a in acts)
rows = {}
for s, e, n in acts:
    m = act.match(n)
    key = ("tower" if int(m.group(1)) == tower else "text") + " " + m.group(2)
    rows.setdefault(key, []).append(inter_len(comp, s, e) / 1e3)
print(f"rank 0, schedule span {(hi - lo) / 1e3:.1f} ms, {len(acts)} PP actions")
for k in sorted(rows):
    v = rows[k]
    print(f"  {k}: n={len(v)} compute mean {statistics.mean(v):.2f} ms, min {min(v):.2f}, max {max(v):.2f}")
inside = union([(s, e) for s, e, _ in acts])
outside = [[s, e] for s, e in comp]
tot_comp = inter_len(comp, lo, hi)
in_acts = sum(inter_len(comp, s, e) for s, e in inside)
print(f"  compute in the span {tot_comp / 1e3:.1f} ms, inside PP actions {in_acts / 1e3:.1f} ms, outside {(tot_comp - in_acts) / 1e3:.1f} ms")
# idle gaps inside the schedule span
gaps = []
prev_end = lo
for s, e in busy:
    if e <= lo or s >= hi:
        continue
    s, e = max(s, lo), min(e, hi)
    if s > prev_end:
        gaps.append((prev_end, s))
    prev_end = max(prev_end, e)
if hi > prev_end:
    gaps.append((prev_end, hi))
total_idle = sum(e - s for s, e in gaps)
print(f"  idle inside the span {total_idle / 1e3:.1f} ms in {len(gaps)} gaps; gaps over 1 ms:")
for s, e in gaps:
    if e - s < 1000:
        continue
    before = [a for a in acts if a[1] <= s + 1]
    after = [a for a in acts if a[0] >= e - 1]
    print(f"    {(e - s) / 1e3:7.2f} ms after {before[-1][2] if before else 'start'} before {after[0][2] if after else 'end'}")
