"""Every PP rank's peak for the balance cells, with the spread across ranks and what each rank
moved out, parked for others and was planned to reach.

Usage: python balance_table.py <LB_ROOT> <step> <cell> [<cell> ...]
"""

import glob
import json
import os
import re
import sys

GIB = 2**30


def _load(root, cell, step):
    peaks, moved = {}, {}
    for f in glob.glob(os.path.join(root, cell, "mem", "rank*.json")):
        data = json.load(open(f))
        recs = data["records"]
        rank = data["rank"]
        peaks[rank] = recs[step]["max_allocated_gib"]
        a, b = recs[step - 1].get("storage", {}), recs[step].get("storage", {})
        moved[rank] = {
            k[: -len("_bytes")]: (b.get(k, 0) - a.get(k, 0)) / GIB
            for k in ("host_bytes", "remote_bytes")
            if b.get(k, 0) != a.get(k, 0)
        }
    plan = None
    files = sorted(glob.glob(os.path.join(root, cell, "mem", "plan*.json")))
    if files:
        plan = json.load(open(files[0]))
    return peaks, moved, plan


def _planned(plan):
    out = {}
    for rank, text in plan["summary"].items():
        m = re.search(r"peak ([\d.]+) -> ([\d.]+) GiB", text)
        out[int(rank)] = (float(m.group(1)), float(m.group(2)))
    return out


def main():
    root, step, cells = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    data = {c: _load(root, c, step) for c in cells}
    ranks = sorted(data[cells[0]][0])
    print(f"peak allocated per PP rank, GiB, step {step}")
    print("| rank | " + " | ".join(cells) + " |")
    print("|---:|" + "---:|" * len(cells))
    for r in ranks:
        print(f"| {r} | " + " | ".join(f"{data[c][0][r]:.2f}" for c in cells) + " |")
    for name, fn in (
        ("max", max),
        ("min", min),
        ("mean", lambda v: sum(v) / len(v)),
        ("spread", lambda v: max(v) - min(v)),
    ):
        print(
            f"| {name} | "
            + " | ".join(f"{fn(list(data[c][0].values())):.2f}" for c in cells)
            + " |"
        )
    for c in cells:
        peaks, moved, plan = data[c]
        if plan is None:
            continue
        print(f"\n{c}: plan (profiled -> planned peak), moved out this step, pool held for others")
        planned = _planned(plan)
        pools = {}
        for src, dst in plan["dests"].items():
            pools[int(dst)] = pools.get(int(dst), 0.0) + plan["spans_gib"][src]
        print("| rank | profiled | planned | measured | out to host | out to a peer | pool held |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for r in ranks:
            p0, p1 = planned[r]
            m = moved.get(r, {})
            print(
                f"| {r} | {p0:.2f} | {p1:.2f} | {peaks[r]:.2f} | {m.get('host', 0):.2f} | "
                f"{m.get('remote', 0):.2f} | {pools.get(r, 0):.2f} |"
            )
        print(f"dests {plan['dests']}")


if __name__ == "__main__":
    main()
