"""Per cell: loss / grad norm at steps 1, 5, 10, the max relative loss gap to a reference cell, and peak memory."""

import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"step:\s*(\d+)\s+loss:\s*([0-9.]+)\s+grad_norm:\s*([0-9.]+)\s+memory:\s*([0-9.]+)GiB")


def read(path):
    steps = {}
    for raw in open(path, errors="replace"):
        m = LINE.search(ANSI.sub("", raw))
        if m and raw.startswith("[rank0]"):
            steps[int(m.group(1))] = (m.group(2), m.group(3), float(m.group(4)))
    return steps


root, ref_name, *names = sys.argv[1:]
ref = read(f"{root}/{ref_name}/run.log")
print(f"{'cell':28s} {'step 1':>18s} {'step 5':>18s} {'step 10':>18s} {'max rel gap':>12s} {'peak GiB':>8s}")
for name in [ref_name, *names]:
    s = read(f"{root}/{name}/run.log")
    cols = [f"{s[k][0]} / {s[k][1]}" if k in s else "-" for k in (1, 5, 10)]
    gaps = [abs(float(s[k][0]) - float(ref[k][0])) / float(ref[k][0]) for k in s if k in ref]
    gap = f"{max(gaps):.2e}" if gaps and name != ref_name else ""
    peak = max(v[2] for v in s.values()) if s else 0.0
    print(f"{name:28s} {cols[0]:>18s} {cols[1]:>18s} {cols[2]:>18s} {gap:>12s} {peak:8.2f}")
