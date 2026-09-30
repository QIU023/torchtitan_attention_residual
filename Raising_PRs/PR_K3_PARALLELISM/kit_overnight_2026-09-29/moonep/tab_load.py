"""Tables for run_load.sh: python tab_load.py <out>

Per cell: rc, loss at steps 1 / 10 / 20, the per-rank load with static expert placement (max over mean, the mean
over micro-batches and the worst micro-batch, from the router counts), and on MoonEP whether every dispatch put
exactly S x K real rows on the rank and how many slots held a copied expert. Step time over steps 11 to 30 for
the timing cells.
"""

import datetime
import glob
import json
import os
import re
import sys

STEP = re.compile(r"\[titan\] (\S+ \S+) .*step:\s+(\d+)\s+loss:\s+(-?[\d.]+)\s+grad_norm:\s+([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def steps(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in ANSI.sub("", open(path, errors="replace").read()).splitlines():
        m = STEP.search(line)
        if m and float(m.group(3)) > 0:
            ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
            out.setdefault(int(m.group(2)), (m.group(3), m.group(4), ts))
    return out


def load(cell_dir):
    ranks = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(cell_dir, "load", "rank*.json")))]
    if not ranks:
        return None
    first = ranks[0]["steps"]
    measured = first[1:] if len(first) > 1 else first
    mean = sum(s["mean_max_over_mean"] for s in measured) / max(len(measured), 1)
    worst = max((s["worst_max_over_mean"] for s in measured), default=float("nan"))
    rows = [m for r in ranks for s in r["steps"] for m in s["moonep"]]
    exact = sum(1 for m in rows if m["real_rows"] == m["expected_rows"])
    nonzero = [m["nonzero_rows"] for m in rows if m.get("nonzero_rows", -1) >= 0]
    if nonzero:
        exact_nz = sum(1 for m in rows if m.get("nonzero_rows") == m["expected_rows"])
        print(f"   {os.path.basename(cell_dir)}: nonzero rows per dispatch min {min(nonzero)} mean {sum(nonzero) / len(nonzero):.1f} "
              f"max {max(nonzero)} (S x K = {rows[0]['expected_rows']}), {exact_nz} / {len(nonzero)} exactly S x K; "
              f"padded minus the zero-fill counts: min {min(m['real_rows'] for m in rows)} max {max(m['real_rows'] for m in rows)}")
    slots = [m["slots_used"] for m in rows if m["slots_used"] >= 0]
    return mean, worst, len(rows), exact, (sum(slots) / len(slots) if slots else float("nan"))


def main(out):
    print("| cell | rc | loss 1 / 10 / 20 | static placement max / mean (mean, worst) | MoonEP dispatches with S x K rows | mean slots used |")
    print("|---|---|---|---|---:|---:|")
    for cell in sorted(d for d in os.listdir(out) if d.startswith("num_")):
        d = os.path.join(out, cell)
        rc = open(os.path.join(d, "rc")).read().strip() if os.path.exists(os.path.join(d, "rc")) else "-"
        s = steps(os.path.join(d, "run.log"))
        loss = " / ".join(s[k][0] if k in s else "-" for k in (1, 10, 20))
        stats = load(d)
        if stats is None:
            print(f"| {cell} | {rc} | {loss} | - | - | - |")
            continue
        mean, worst, n, exact, slots = stats
        moon = f"{exact} / {n}" if n else "-"
        print(f"| {cell} | {rc} | {loss} | {mean:.2f}, {worst:.2f} | {moon} | {slots:.1f} |" if n else
              f"| {cell} | {rc} | {loss} | {mean:.2f}, {worst:.2f} | - | - |")
    timing = sorted(d for d in os.listdir(out) if d.startswith("time_"))
    if timing:
        print("\n| timing cell | rc | s / step, steps 11 to 30 |")
        print("|---|---|---:|")
        for cell in timing:
            d = os.path.join(out, cell)
            rc = open(os.path.join(d, "rc")).read().strip() if os.path.exists(os.path.join(d, "rc")) else "-"
            s = steps(os.path.join(d, "run.log"))
            ks = [k for k in range(10, 31) if k in s]
            per = (s[ks[-1]][2] - s[ks[0]][2]).total_seconds() / (ks[-1] - ks[0]) if len(ks) > 1 else float("nan")
            print(f"| {cell} | {rc} | {per:.4f} |")


if __name__ == "__main__":
    main(sys.argv[1])
