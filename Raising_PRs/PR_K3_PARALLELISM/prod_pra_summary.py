"""Collect the min / mean / max rows of every prod_pra_*.out.txt into one summary."""

import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fullac_h100a", "fullac_h100b", "fullac_gb300a", "fullac_gb300b", "fullac_h100a_host25", "fullac_h100b_host25",
         "fullac_h100a_unfused", "report_h100a", "report_h100b", "report_gb300a", "report_gb300b", "report_h100a_host25",
         "report_h100b_host25", "report_h100a_unfused"]
for key in ORDER:
    path = os.path.join(HERE, f"prod_pra_{key}.out.txt")
    if not os.path.exists(path):
        continue
    lines = open(path).read().splitlines()
    header = next((l[4:] for l in lines if l.startswith("=== ")), "?")
    print(f"#### prod_pra_{key} | {header}")
    for l in lines:
        if not l.startswith("| ") or l.startswith("| scheme"):
            continue
        cells = [c.strip() for c in l.strip("|").split("|")]
        name = cells[0]
        n = len(cells)
        mn, mean, mx, step, stall, nic, host = cells[n - 8 : n - 1]
        print(f"  {name:<36} min {float(mn):6.1f} mean {float(mean):6.1f} max {float(mx):6.1f} | step {float(step):6.2f} "
              f"stall {float(stall):6.3f} | host GB {float(host):6.0f}")
    for l in lines:
        if l.startswith("PR A block + hidden"):
            print("  " + l)
