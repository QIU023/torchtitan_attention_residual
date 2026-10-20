"""Where DEP's vision work ran in one traced step, from the DEP:E<mb> / DEP:B<mb> annotations of the probe patch
(probe_annotate_vision_dep_pkg.patch; kit only). Measured, GPU kernel time.

Usage: python ana_dep_place.py <trace dir>

The schedule span is global: from the first PP action annotation on any rank to the last one on any rank (the ranks
share the host clock). Each encode (E) and tower backward (B) is split by where its kernels ran:
  before   ahead of the span (upfront encodes)
  lead     inside the span, before the rank's own first action (its warmup bubble)
  inner    inside the rank's own window, between its actions
  trail    inside the span, after the rank's own last action (its cooldown bubble)
  after    behind the span (the epilogue)
lead + inner + trail is the vision work that ran in pipeline bubbles: the hiding rate is its share of all vision work.
A lead or inner item whose last kernel ends less than 20 us before the rank's next action is counted as adjacent: the
action may have waited for it. The cost ratio is the mean encode over the mean forward of a middle stage (neither the
first nor the last), both in kernel time, the unit of vision_dep.bubble_cost_ratio.
"""

import glob
import os
import re
import statistics
import sys

from ana_fill import _ACTION, _intersect, _length, _load, _union

_DEP = re.compile(r"^DEP:(E|B)(\d+)$")
_WHERE = ("before", "lead", "inner", "trail", "after")
_ADJACENT_US = 20.0


def _rank_events(path):
    events = _load(path).get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
    compute = _union([(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" not in e["name"].lower()])
    actions, deps = [], []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
            continue
        name = e.get("name", "")
        m = _ACTION.match(name)
        if m:
            actions.append((int(m.group(1)), m.group(2), e["ts"], e["ts"] + e["dur"]))
        m = _DEP.match(name)
        if m:
            deps.append((m.group(1), int(m.group(2)), e["ts"], e["ts"] + e["dur"]))
    return compute, sorted(actions, key=lambda a: a[2]), deps


def main():
    ranks = {}
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "**", "rank*_trace.json*"), recursive=True)):
        rank = int(re.search(r"rank(\d+)", os.path.basename(path)).group(1))
        ranks[rank] = _rank_events(path)
    lo = min(a[0][2] for _, a, _ in ranks.values() if a)
    hi = max(max(x[3] for x in a) for _, a, _ in ranks.values() if a)
    stages = {s for _, a, _ in ranks.values() for s, _, _, _ in a}
    first, last = min(stages), max(stages)
    totals = {(k, w): 0.0 for k in "EB" for w in _WHERE}
    counts = {k: 0 for k in "EB"}
    durations = {k: [] for k in "EB"}
    adjacent = 0.0
    idle_total = 0.0
    forwards = []
    print(f"schedule span {(hi - lo) * 1e-3:.1f} ms")
    print("| rank | idle in span | " + " | ".join(f"{k} {w}" for k in "EB" for w in _WHERE) + " | adjacent |")
    print("|---:|---:|" + "---:|" * (2 * len(_WHERE) + 1))
    for rank in sorted(ranks):
        compute, actions, deps = ranks[rank]
        idle = (hi - lo - _length(_intersect(compute, [[lo, hi]]))) * 1e-3
        idle_total += idle
        forwards += [_length(_intersect(compute, [[a, b]])) * 1e-3 for s, kind, a, b in actions
                     if kind == "F" and first < s < last]
        rlo, rhi = actions[0][2], max(a[3] for a in actions)
        regions = {"before": [[float("-inf"), lo]], "lead": [[lo, rlo]], "inner": [[rlo, rhi]],
                   "trail": [[rhi, hi]], "after": [[hi, float("inf")]]}
        cells = {(k, w): 0.0 for k in "EB" for w in _WHERE}
        rank_adjacent = 0.0
        for kind, _, start, end in deps:
            kernels = _intersect(compute, [[start, end]])
            counts[kind] += 1
            durations[kind].append(_length(kernels) * 1e-3)
            for where, region in regions.items():
                ms = _length(_intersect(kernels, region)) * 1e-3
                cells[(kind, where)] += ms
                totals[(kind, where)] += ms
            done = kernels[-1][1] if kernels else None
            if done is not None and lo <= done <= rhi:
                nxt = [a for _, _, a, _ in actions if a >= done - 1]
                if nxt and min(nxt) - done < _ADJACENT_US:
                    rank_adjacent += _length(kernels) * 1e-3
        adjacent += rank_adjacent
        print(f"| {rank} | {idle:.1f} | " + " | ".join(f"{cells[(k, w)]:.1f}" for k in "EB" for w in _WHERE)
              + f" | {rank_adjacent:.1f} |")
    print("| all | " + f"{idle_total:.1f} | " + " | ".join(f"{totals[(k, w)]:.1f}" for k in "EB" for w in _WHERE)
          + f" | {adjacent:.1f} |")
    vision = sum(totals.values())
    hidden = sum(totals[(k, w)] for k in "EB" for w in ("lead", "inner", "trail"))
    encode = statistics.mean(durations["E"]) if durations["E"] else float("nan")
    backward = statistics.mean(durations["B"]) if durations["B"] else float("nan")
    stage_f = statistics.mean(forwards)
    print(f"encodes {counts['E']} (mean {encode:.3f} ms), backwards {counts['B']} (mean {backward:.3f} ms, "
          f"{backward / encode:.2f}x an encode), middle stage forward {stage_f:.3f} ms over {len(forwards)} actions")
    print(f"cost ratio (annotated kernel time): {encode / stage_f:.3f}")
    share = f"{hidden / vision:.0%}" if vision else "n/a"
    print(f"hiding rate: {hidden:.1f} of {vision:.1f} ms of vision work in pipeline bubbles ({share}); "
          f"{adjacent:.1f} ms of it ends right before the rank's next action; idle in the span {idle_total:.1f} ms")


if __name__ == "__main__":
    main()
