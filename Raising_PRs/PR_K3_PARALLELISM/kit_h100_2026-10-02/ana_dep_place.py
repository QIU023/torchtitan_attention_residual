"""Where DEP's vision work ran in one traced step, from the DEP:E<mb> / DEP:B<mb> annotations of the probe patch
(probe_annotate_vision_dep.patch; kit only). Measured, GPU kernel time.

Usage: python ana_dep_place.py <trace dir>

The schedule span is global: from the first PP action annotation on any rank to the last one on any rank (the ranks
share the host clock). Each encode (E) and tower backward (B) is placed by where its kernels ran:
  before   ahead of the span (K2.5's upfront encodes, bubble's first pipeline-degree ones)
  lead     inside the span, before the rank's own first action (its warmup bubble)
  inner    inside the rank's own window, between its actions
  trail    inside the span, after the rank's own last action (its cooldown bubble)
  after    behind the span (the epilogue)
lead + inner + trail is the vision work that ran in pipeline bubbles. Per rank the table also gives the rank's idle time
inside the span (span minus its compute there).
"""

import glob
import os
import re
import sys

from ana_fill import _ACTION, _intersect, _length, _load, _union

_DEP = re.compile(r"^DEP:(E|B)(\d+)$")
_WHERE = ("before", "lead", "inner", "trail", "after")


def _rank_events(path):
    events = _load(path).get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
    compute = _union([(e["ts"], e["ts"] + e["dur"]) for e in kernels if "nccl" not in e["name"].lower()])
    actions, deps = [], []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
            continue
        name = e.get("name", "")
        if _ACTION.match(name):
            actions.append((e["ts"], e["ts"] + e["dur"]))
        m = _DEP.match(name)
        if m:
            deps.append((m.group(1), int(m.group(2)), e["ts"], e["ts"] + e["dur"]))
    return compute, _union(actions), deps


def main():
    ranks = {}
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "**", "rank*_trace.json*"), recursive=True)):
        rank = int(re.search(r"rank(\d+)", os.path.basename(path)).group(1))
        ranks[rank] = _rank_events(path)
    lo = min(actions[0][0] for _, actions, _ in ranks.values() if actions)
    hi = max(actions[-1][1] for _, actions, _ in ranks.values() if actions)
    print(f"schedule span {(hi - lo) * 1e-3:.1f} ms")
    print("| rank | idle in span | " + " | ".join(f"E {w}" for w in _WHERE) + " | " + " | ".join(f"B {w}" for w in _WHERE) + " |")
    print("|---:|---:|" + "---:|" * (2 * len(_WHERE)))
    totals = {(k, w): [0, 0.0] for k in "EB" for w in _WHERE}
    idle_total = 0.0
    for rank in sorted(ranks):
        compute, actions, deps = ranks[rank]
        idle = (hi - lo) * 1e-3 - _length(_intersect(compute, [[lo, hi]])) * 1e-3
        idle_total += idle
        rlo, rhi = actions[0][0], actions[-1][1]
        cells = {(k, w): [0, 0.0] for k in "EB" for w in _WHERE}
        for kind, _, start, end in deps:
            ms = _length(_intersect(compute, [[start, end]])) * 1e-3
            mid = (start + end) / 2
            where = ("before" if mid < lo else "after" if mid > hi else "lead" if mid < rlo
                     else "trail" if mid > rhi else "inner")
            cells[(kind, where)][0] += 1
            cells[(kind, where)][1] += ms
            totals[(kind, where)][0] += 1
            totals[(kind, where)][1] += ms
        print(f"| {rank} | {idle:.1f} | " + " | ".join(f"{cells[(k, w)][0]} / {cells[(k, w)][1]:.1f}"
              for k in "EB" for w in _WHERE) + " |")
    print("| all | " + f"{idle_total:.1f} | " + " | ".join(f"{totals[(k, w)][0]} / {totals[(k, w)][1]:.1f}"
          for k in "EB" for w in _WHERE) + " |")
    in_bubbles = sum(totals[(k, w)][1] for k in "EB" for w in ("lead", "inner", "trail"))
    vision = sum(totals[(k, w)][1] for k in "EB" for w in _WHERE)
    share = f"{in_bubbles / vision:.0%}" if vision else "n/a"
    print(f"vision work {vision:.1f} ms, in pipeline bubbles {in_bubbles:.1f} ms ({share} of it), "
          f"idle in the span {idle_total:.1f} ms (cells: count / ms)")


if __name__ == "__main__":
    main()
