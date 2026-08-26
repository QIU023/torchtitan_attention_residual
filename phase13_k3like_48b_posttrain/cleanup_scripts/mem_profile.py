"""Per-rank peak memory for one cell, from its run log.

The adapter's whole point is that a rank reuses the block stack its own virtual
stages already hold and ships only the delta, so the stack does not accumulate
towards the last rank. That shows up in memory before it shows up in loss.
"""
import re, sys, os

def peaks(path):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="ignore").read())
    # one "memory: X GiB(Y%)" per rank per logged step; keep each rank's max by
    # position within a step group
    per_step = {}
    for m in re.finditer(r"step: +(\d+) .*?memory: +([0-9.]+)GiB\(([0-9.]+)%\)", t):
        per_step.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    if not per_step:
        return []
    n = max(len(v) for v in per_step.values())
    out = [0.0] * n
    for vals in per_step.values():
        for i, v in enumerate(vals):
            if i < n:
                out[i] = max(out[i], v)
    return out

for label, path in (a.split("=", 1) for a in sys.argv[1:]):
    p = peaks(path)
    if not p:
        print(f"{label:<8} no memory lines"); continue
    print(f"{label:<8} n={len(p)}  peaks={[round(x,2) for x in p]}  "
          f"max={max(p):.2f}  spread={max(p)-min(p):.2f}")
