"""Summarise run_attnres_h100.sh logs: loss at steps 1/10/last, peak memory, mean tps (steps 5 on)."""
import glob
import os
import re
import sys

out = sys.argv[1]
row = re.compile(r"step:\s+(\d+)\s+loss:\s+([-\d.]+).*?memory:\s+([\d.]+)GiB.*?tps:\s+([\d,]+)")
print("| cell | loss@1 | loss@10 | loss@last | peak GiB | tps (steps 5+) | rc |")
print("|---|---|---|---|---|---|---|")
for log in sorted(glob.glob(os.path.join(out, "*.log"))):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(log).read())
    steps = {int(m[1]): (float(m[2]), float(m[3]), int(m[4].replace(",", ""))) for m in row.finditer(text)}
    rc = (re.findall(r"^rc=(\d+)", text, re.M) or ["?"])[-1]
    if not steps:
        print(f"| {os.path.basename(log)[:-4]} | | | | | | {rc} |"); continue
    last = max(steps)
    tps = [v[2] for s, v in steps.items() if s >= 5]
    print(f"| {os.path.basename(log)[:-4]} | {steps[1][0]:.5f} | {steps.get(10, (float('nan'),))[0]:.5f} | {steps[last][0]:.5f} (step {last}) | {max(v[1] for v in steps.values()):.2f} | {sum(tps) / max(1, len(tps)):.0f} | {rc} |")
