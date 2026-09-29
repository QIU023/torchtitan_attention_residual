"""Per-rank compute, communication and their overlap in one traced step (titan's profiler).

Usage: python analyze_trace.py <trace_dir> [--json out.json]

Reads every rank*_trace.json(.gz) under <trace_dir>. Per rank, on its GPU timeline:
  window     first to last kernel of the traced step
  compute    union of non-NCCL kernel intervals
  comm       union of NCCL kernel intervals (a P2P kernel is resident until its peer matches it)
  overlap    compute and comm at the same time
  exposed    comm with no compute running
  idle       neither compute nor comm
  fwd / bwd  compute time inside the PP:<stage>F<mb> / PP:<stage>B<mb> action annotations
All times in milliseconds.
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def _union(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def _length(intervals):
    return sum(end - start for start, end in intervals)


def _intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            out.append([start, end])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


_ACTION = re.compile(r"^PP:(\d+)(F|B|I|W)(\d+)$")


def analyze(trace):
    events = trace.get("traceEvents", trace)
    kernels = [
        e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e
    ]
    if not kernels:
        return None
    comm = _union(
        [(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" in e["name"].lower()]
    )
    compute = _union(
        [
            (e["ts"], e["ts"] + e["dur"])
            for e in kernels
            if "nccl" not in e["name"].lower()
        ]
    )
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    busy = _union(compute + comm)
    overlap = _intersect(compute, comm)
    fwd = bwd = 0.0
    actions = {"F": 0, "B": 0}
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
            continue
        m = _ACTION.match(e.get("name", ""))
        if not m:
            continue
        span = [[e["ts"], e["ts"] + e["dur"]]]
        inside = _length(_intersect(compute, span))
        if m.group(2) == "F":
            fwd += inside
            actions["F"] += 1
        else:
            bwd += inside
            actions["B"] += 1
    ms = 1e-3
    return {
        "window": (end - start) * ms,
        "compute": _length(compute) * ms,
        "comm": _length(comm) * ms,
        "overlap": _length(overlap) * ms,
        "exposed": (_length(comm) - _length(overlap)) * ms,
        "idle": ((end - start) - _length(busy)) * ms,
        "fwd": fwd * ms,
        "bwd": bwd * ms,
        "actions": actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--json")
    args = ap.parse_args()
    paths = sorted(
        glob.glob(os.path.join(args.trace_dir, "**", "rank*_trace.json*"), recursive=True)
    )
    if not paths:
        sys.exit(f"no rank*_trace.json under {args.trace_dir}")
    rows = {}
    for path in paths:
        rank = int(re.search(r"rank(\d+)", os.path.basename(path)).group(1))
        result = analyze(_load(path))
        if result is not None:
            rows[rank] = result
    cols = ["window", "compute", "fwd", "bwd", "comm", "overlap", "exposed", "idle"]
    print("| rank | " + " | ".join(cols) + " | F / B actions |")
    print("|---:|" + "---:|" * len(cols) + "---|")
    for rank in sorted(rows):
        r = rows[rank]
        print(
            f"| {rank} | "
            + " | ".join(f"{r[c]:.1f}" for c in cols)
            + f" | {r['actions']['F']} / {r['actions']['B']} |"
        )
    n = len(rows)
    for label, agg in (("max", max), ("mean", lambda xs: sum(xs) / n)):
        print(
            f"| {label} | "
            + " | ".join(f"{agg([rows[k][c] for k in rows]):.1f}" for c in cols)
            + " | |"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
