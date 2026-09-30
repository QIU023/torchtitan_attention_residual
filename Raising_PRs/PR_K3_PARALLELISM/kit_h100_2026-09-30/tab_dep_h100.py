"""Tables of the H100 DEP campaign (run_dep_h100.sh): step time, peaks, plans and measured fill; the numerics cells.

Usage: python tab_dep_h100.py <results dir, e.g. results/dep_h100_pp4vpp4>
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kit_h100_2026-09-29", "dep"))
from steps import mem_by_rank, step_seconds, steps  # noqa: E402

MODES = (("w_dep_off", "DEP off"), ("w_dep_k25", "K2.5"), ("w_dep_bubble", "bubble"))


def _plan(log):
    with open(log, errors="replace") as f:
        m = re.search(r"vision_dep: (.*)", re.sub(r"\x1b\[[0-9;]*m", "", f.read()))
    return m.group(1).strip() if m else ""


def timing(root):
    cells = sorted({re.match(r"time_(.+)_w_dep_", os.path.basename(d)).group(1)
                    for d in glob.glob(os.path.join(root, "time_*_w_dep_off"))})
    print("| level | M | mode | s/step (steps 5 to 20) | vs DEP off | vs K2.5 | peak GiB per rank (step 20) | plan |")
    print("|---|---:|---|---:|---:|---:|---|---|")
    for cell in cells:
        level, m = cell.rsplit("_m", 1)
        secs = {}
        for key, label in MODES:
            log = os.path.join(root, f"time_{cell}_{key}", "run.log")
            if not os.path.exists(log):
                continue
            rec = steps(log)
            secs[key] = step_seconds(rec, 5, 20) if rec else None
            mem = mem_by_rank(log, 20)
            t = secs[key]
            vs_off = f"{t / secs['w_dep_off'] - 1:+.1%}" if t and secs.get("w_dep_off") and key != "w_dep_off" else ""
            vs_k25 = f"{t / secs['w_dep_k25'] - 1:+.1%}" if t and secs.get("w_dep_k25") and key == "w_dep_bubble" else ""
            peaks = " / ".join(f"{mem[r]:.1f}" for r in sorted(mem))
            print(f"| {level} | {m} | {label} | {t:.3f} | {vs_off} | {vs_k25} | {peaks} | {_plan(log)} |"
                  if t else f"| {level} | {m} | {label} | failed | | | | |")
        fill = os.path.join(root, f"fill_{cell}.txt")
        if os.path.exists(fill):
            with open(fill) as f:
                line = [x for x in f if x.startswith("all ranks")]
            if line:
                print(f"| {level} | {m} | measured fill (step 10 trace) | {line[0].strip()[len('all ranks: '):]} | | | | |")


def numerics(root):
    cells = [("num_off_a", "DEP off"), ("num_off_b", "DEP off, again"), ("num_k25", "K2.5"), ("num_bubble", "bubble")]
    recs = {name: steps(os.path.join(root, name, "run.log")) for name, _ in cells
            if os.path.exists(os.path.join(root, name, "run.log"))}
    if "num_off_a" not in recs:
        return
    ref = recs["num_off_a"]
    cols = [s for s in (1, 5, 10, 20) if s in ref]
    print()
    print("| cell | " + " | ".join(f"loss step {s}" for s in cols) + " | " + " | ".join(f"grad norm step {s}" for s in cols) + " |")
    print("|---|" + "---:|" * (2 * len(cols)))
    for name, label in cells:
        rec = recs.get(name)
        if not rec:
            continue
        row = []
        for field in ("loss", "gn"):
            for s in cols:
                if s not in rec:
                    row.append("")
                    continue
                v, r = rec[s][field], ref[s][field]
                diff = "identical" if v == r else f"{float(v) / float(r) - 1:+.2%}"
                row.append(f"`{v}`" if name == "num_off_a" else f"`{v}`<br>{diff}")
        same = sum(1 for s in rec if s in ref and rec[s]["loss"] == ref[s]["loss"])
        print(f"| {label} ({same}/{len(ref)} steps loss identical) | " + " | ".join(row) + " |")


def main():
    root = sys.argv[1]
    timing(root)
    numerics(root)


if __name__ == "__main__":
    main()
