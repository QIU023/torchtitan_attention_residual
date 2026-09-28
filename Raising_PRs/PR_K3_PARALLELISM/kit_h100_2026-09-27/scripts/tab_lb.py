"""Tables for the PP lower-bound campaign: every rank's peak memory and the loss trajectories.

Usage: python tab_lb.py <LB_ROOT> <step> <cell> [<cell> ...]

Memory: per rank, the peak allocated and the rank store's high-water mark in the given step
(records are taken just before titan resets the peaks; titan also resets once before step 1), with max and mean over ranks.
Losses: loss and grad norm of every step from each cell's rank-7 log, and whether every cell
matches the first one bitwise.
"""

import json
import os
import re
import sys


def _memory(root, cell, step):
    out = {}
    folder = os.path.join(root, cell, "mem")
    for name in sorted(os.listdir(folder)):
        if not re.fullmatch(r"rank\d+\.json", name):
            continue
        with open(os.path.join(folder, name)) as f:
            data = json.load(f)
        # records[0] is titan's reset before the first step, so step k is records[k]
        out[data["rank"]] = data["records"][step]
    return out


def _losses(root, cell):
    pattern = re.compile(r"step:\s+(\d+)\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)")
    rows = {}
    with open(os.path.join(root, cell, "train.log")) as f:
        for line in f:
            m = pattern.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if m:
                rows[int(m.group(1))] = (m.group(2), m.group(3))
    return rows


def main():
    root, step, cells = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    mem = {c: _memory(root, c, step) for c in cells}
    ranks = sorted(next(iter(mem.values())))
    for key, label in (("max_allocated_gib", "peak allocated"), ("store_peak_gib", "rank store")):
        print(f"\n{label}, GiB, step {step}")
        print("| rank | " + " | ".join(cells) + " |")
        print("|---:|" + "---:|" * len(cells))
        for r in ranks:
            print(f"| {r} | " + " | ".join(f"{mem[c][r].get(key, 0):.2f}" for c in cells) + " |")
        for agg, fn in (("max", max), ("mean", lambda xs: sum(xs) / len(xs))):
            print(
                f"| {agg} | "
                + " | ".join(f"{fn([mem[c][r].get(key, 0) for r in ranks]):.2f}" for c in cells)
                + " |"
            )
    losses = {c: _losses(root, c) for c in cells}
    steps = sorted(losses[cells[0]])
    print("\nloss / grad norm per step")
    print("| step | " + " | ".join(cells) + " |")
    print("|---:|" + "---|" * len(cells))
    for s in steps:
        print(f"| {s} | " + " | ".join("/".join(losses[c].get(s, ("-", "-"))) for c in cells) + " |")
    for c in cells[1:]:
        same = all(losses[c].get(s) == losses[cells[0]][s] for s in steps)
        print(f"{c} vs {cells[0]}: {'identical on all ' + str(len(steps)) + ' steps' if same else 'DIFFERS'}")


if __name__ == "__main__":
    main()
