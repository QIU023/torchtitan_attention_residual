"""Measured bubble fill of DEP: one traced step of DEP off, K2.5 and bubble at the same level and micro-batch count.

Usage: python ana_fill.py <off trace dir> <k25 trace dir> <bubble trace dir>

Per rank, on its GPU timeline (compute = every non-NCCL kernel; times in ms):
  span      first to last kernel of the traced step (optimizer included)
  window    first to last PP action annotation (PP:<stage>F|B|I|W<mb>): the schedule
  busy      compute inside the window; idle = window - busy (with tp1 and ep1 the only collectives are the pipeline's
            P2P, so idle is the bubble)
  gaps      window time outside every F/B/I/W action; in gaps = compute there (the vision work DEP ran after a send
            or between actions)
  before / after   compute before and after the window (upfront encodes; trailing tower backwards and the optimizer)
Then bubble against K2.5, whose vision work all sits outside the window:
  moved in  busy(bubble) - busy(k25): vision work the planner put inside the schedule (text work is the same in both)
  fill      moved in / idle(k25): the share of K2.5's bubble that became vision work
  stretch   window(bubble) - window(k25); saved = span(k25) - span(bubble)
"""

import glob
import gzip
import json
import os
import re
import sys

_ACTION = re.compile(r"^PP:(\d+)(F|B|I|W)(\d+)$")


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


def _complement(intervals, lo, hi):
    out, cur = [], lo
    for start, end in intervals:
        if start > cur:
            out.append([cur, min(start, hi)])
        cur = max(cur, end)
    if cur < hi:
        out.append([cur, hi])
    return out


def rank_stats(trace):
    events = trace.get("traceEvents", trace)
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
    compute = _union([(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" not in e["name"].lower()])
    actions = _union(
        [
            (e["ts"], e["ts"] + e["dur"])
            for e in events
            if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and _ACTION.match(e.get("name", ""))
        ]
    )
    if not compute or not actions:
        return None
    start, end = compute[0][0], compute[-1][1]
    lo, hi = actions[0][0], actions[-1][1]
    busy = _length(_intersect(compute, [[lo, hi]]))
    gaps = _complement(actions, lo, hi)
    ms = 1e-3
    return {
        "span": (end - start) * ms,
        "window": (hi - lo) * ms,
        "busy": busy * ms,
        "idle": (hi - lo - busy) * ms,
        "gaps": _length(gaps) * ms,
        "in gaps": _length(_intersect(compute, gaps)) * ms,
        "before": _length(_intersect(compute, [[start, lo]])) * ms,
        "after": _length(_intersect(compute, [[hi, end]])) * ms,
    }


def mode_stats(trace_dir):
    rows = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "**", "rank*_trace.json*"), recursive=True)):
        rank = int(re.search(r"rank(\d+)", os.path.basename(path)).group(1))
        stats = rank_stats(_load(path))
        if stats is not None:
            rows[rank] = stats
    return rows


def main():
    modes = dict(zip(("off", "k25", "bubble"), (mode_stats(d) for d in sys.argv[1:4])))
    cols = ["span", "window", "busy", "idle", "gaps", "in gaps", "before", "after"]
    for mode, rows in modes.items():
        print(f"{mode}: {sys.argv[1 + list(modes).index(mode)]}")
        print("| rank | " + " | ".join(cols) + " |")
        print("|---:|" + "---:|" * len(cols))
        for rank in sorted(rows):
            print(f"| {rank} | " + " | ".join(f"{rows[rank][c]:.1f}" for c in cols) + " |")
        print()
    k25, bub, off = modes["k25"], modes["bubble"], modes["off"]
    print("| rank | idle k25 | moved in | fill | in gaps (bubble) | stretch | span off | span k25 | span bubble | saved vs k25 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot_idle = tot_moved = 0.0
    for rank in sorted(set(k25) & set(bub)):
        moved = bub[rank]["busy"] - k25[rank]["busy"]
        idle = k25[rank]["idle"]
        tot_idle += idle
        tot_moved += moved
        span_off = off[rank]["span"] if rank in off else float("nan")
        print(
            f"| {rank} | {idle:.1f} | {moved:.1f} | {moved / idle:.0%} | {bub[rank]['in gaps']:.1f} | "
            f"{bub[rank]['window'] - k25[rank]['window']:+.1f} | {span_off:.1f} | {k25[rank]['span']:.1f} | "
            f"{bub[rank]['span']:.1f} | {k25[rank]['span'] - bub[rank]['span']:+.1f} |"
        )
    if tot_idle:
        print(f"all ranks: K2.5 idle in the schedule {tot_idle:.1f} ms, moved in by bubble {tot_moved:.1f} ms, "
              f"fill {tot_moved / tot_idle:.0%}")


if __name__ == "__main__":
    main()
