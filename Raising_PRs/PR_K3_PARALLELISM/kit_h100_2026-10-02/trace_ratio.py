"""The planner's cost ratio as the step actually ran it, from one traced K2.5 step (measured, GPU kernel time).

Usage: python trace_ratio.py <k25 trace dir> <number of encodes before the schedule (the plan log line)>

encode = compute before the schedule window, summed over the ranks, divided by the number of upfront encodes (K2.5 runs
every encode there; the window starts at the first PP action annotation);
stage forward = compute inside each PP:<stage>F<mb> annotation of a middle stage (neither the first nor the last),
averaged over all of them;
cost ratio = encode / stage forward, the unit of vision_dep.bubble_cost_ratio.
"""

import glob
import os
import re
import sys

from ana_fill import _ACTION, _intersect, _length, _load, _union


def main():
    trace_dir, n_encodes = sys.argv[1], int(sys.argv[2])
    before_total, forwards, stages = 0.0, [], set()
    per_rank = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "**", "rank*_trace.json*"), recursive=True)):
        rank = int(re.search(r"rank(\d+)", os.path.basename(path)).group(1))
        events = _load(path).get("traceEvents", [])
        kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
        compute = _union([(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" not in e["name"].lower()])
        actions = []
        for e in events:
            if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
                continue
            m = _ACTION.match(e.get("name", ""))
            if m:
                actions.append((int(m.group(1)), m.group(2), e["ts"], e["ts"] + e["dur"]))
                stages.add(int(m.group(1)))
        if not compute or not actions:
            continue
        lo = min(a[2] for a in actions)
        before = _length(_intersect(compute, [[compute[0][0], lo]])) * 1e-3
        before_total += before
        per_rank[rank] = before
        rank_forwards = [(s, _length(_intersect(compute, [[a, b]])) * 1e-3) for s, kind, a, b in actions if kind == "F"]
        forwards.extend(rank_forwards)
    first, last = min(stages), max(stages)
    middle = [ms for s, ms in forwards if first < s < last]
    stage_f = sum(middle) / len(middle)
    encode = before_total / n_encodes
    print("compute before the schedule per rank (ms): " + ", ".join(f"{r}: {v:.2f}" for r, v in sorted(per_rank.items())))
    print(f"encodes before the schedule: {n_encodes}; per encode {encode:.3f} ms")
    print(f"middle stage forwards: {len(middle)} actions, mean {stage_f:.3f} ms")
    print(f"cost ratio (trace, kernel time): {encode / stage_f:.3f}")


if __name__ == "__main__":
    main()
