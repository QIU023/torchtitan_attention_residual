"""DEP on 4 x H100: the numerics comparison, then for the widened model the step time over steps 12 to 30
and, per rank of each traced step, where the compute that runs outside every PP action falls (DEP's
tower work has no annotation of its own): before the first action, in the gaps between actions, or
after the last one.

Usage: python ana_dep4.py <results/dep>
"""
import glob
import gzip
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from steps import step_seconds, steps  # noqa: E402

R = sys.argv[1]
CFGS = ("w_dep_off", "w_dep_k25", "w_dep_bubble")
ACT = re.compile(r"^PP:(\d+)(F|B|I|W)(\d+)$")


def rc(name):
    p = os.path.join(R, name, "rc")
    return open(p).read().strip() if os.path.exists(p) else "missing"


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def clip(iv, lo, hi):
    return [[max(s, lo), min(e, hi)] for s, e in iv if min(e, hi) > max(s, lo)]


def subtract(a, b):
    out = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append([cur, bs])
            cur = max(cur, be)
        if cur < e:
            out.append([cur, e])
    return out


def placement(trace_path):
    tr = json.load(gzip.open(trace_path, "rt") if trace_path.endswith(".gz") else open(trace_path))
    ev = tr.get("traceEvents", tr)
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
    comp = union([(k["ts"], k["ts"] + k["dur"]) for k in kern if "nccl" not in k["name"].lower()])
    acts = union([(e["ts"], e["ts"] + e["dur"]) for e in ev
                  if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and ACT.match(e.get("name", ""))])
    if not acts or not comp:
        return None
    lo, hi = acts[0][0], acts[-1][1]
    outside = subtract(comp, acts)
    return {
        "window": (max(e for _, e in comp) - min(s for s, _ in comp)) / 1e3,
        "in_actions": length(clip(comp, lo, hi)) / 1e3 - length(clip(outside, lo, hi)) / 1e3,
        "before": length(clip(outside, float("-inf"), lo)) / 1e3,
        "between": length(clip(outside, lo, hi)) / 1e3,
        "after": length(clip(outside, hi, float("inf"))) / 1e3,
    }


print("## numerics (pp2 x vpp4 x tp2 x ep2, every micro-batch with images, 3 steps)\n")
for n in ("off_a", "off_b", "k25", "bubble"):
    r = steps(os.path.join(R, n, "run.log"))
    print(f"- {n}: rc {rc(n)}; loss " + " / ".join(r[s]["loss"] if s in r else "-" for s in (1, 2, 3))
          + "; grad norm " + " / ".join(r[s]["gn"] if s in r else "-" for s in (1, 2, 3)))
cmp = os.path.join(R, "cmp.txt")
if os.path.exists(cmp):
    print("\n```\n" + open(cmp).read().rstrip() + "\n```")
print("\n## widened model, step time over steps 12 to 30\n")
print("| config | rc | s / step | vs DEP off | peak GiB per rank (max reserved logged) |")
print("|---|---|---:|---:|---|")
base = None
for c in CFGS:
    t = steps(os.path.join(R, f"time_{c}", "run.log"))
    s = step_seconds(t, 12, 30) if t else None
    base = s if c == "w_dep_off" else base
    mem = []
    for f in sorted(glob.glob(os.path.join(R, f"time_{c}", "mem", "rank*.json"))):
        recs = json.load(open(f))["records"]
        mem.append(max(x["max_reserved_gib"] for x in recs))
    rel = f"{(s / base - 1) * 100:+.1f}%" if s and base else ""
    print(f"| {c} | {rc('time_' + c)} | {s:.4f} | {rel} | {' / '.join(f'{m:.1f}' for m in mem)} |" if s
          else f"| {c} | {rc('time_' + c)} | | | |")
print("\n## traced step 15: compute outside PP actions per rank (ms)\n")
print("| config | rank | window | compute in actions | outside: before first | between actions | after last |")
print("|---|---:|---:|---:|---:|---:|---:|")
for c in CFGS:
    for path in sorted(glob.glob(os.path.join(R, f"trace_{c}", "out", "**", "rank*_trace.json*"), recursive=True)):
        rank = int(re.search(r"rank(\d+)_trace", path).group(1))
        p = placement(path)
        if p:
            print(f"| {c} | {rank} | {p['window']:.1f} | {p['in_actions']:.1f} | {p['before']:.1f} | {p['between']:.1f} | {p['after']:.1f} |")
