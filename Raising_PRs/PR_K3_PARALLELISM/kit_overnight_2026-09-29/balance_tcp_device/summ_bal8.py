"""Per cell: rank 7's loss and grad norm per step (identity against the first cell), step times from the
log timestamps (mean of steps 3 to 6), nvidia-smi's highest memory.used per GPU during the cell, and any
process holding memory on a GPU other than its own (a stray CUDA context).

Usage: python summ_bal8.py <root> <cell> [<cell> ...]
"""

import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

LINE = re.compile(
    r"\[rank7\]:\[titan\] (\S+ \S+) .*step:\s+(\d+)\s+loss:\s+([\d.]+)\s+grad_norm:\s+([\d.]+)"
)


def steps(root, cell):
    out = {}
    for raw in open(os.path.join(root, cell, "train.log"), errors="replace"):
        m = LINE.search(re.sub(r"\x1b\[[0-9;]*m", "", raw))
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
            out[int(m.group(2))] = (m.group(3), m.group(4), t)
    return out


def bus_to_index():
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return {b.strip().lower(): int(i) for i, b in (r.split(",") for r in rows)}


def smi(root, cell, bus):
    gpu_max = defaultdict(int)
    f = os.path.join(root, cell, "smi_gpu.txt")
    if os.path.exists(f):
        for raw in open(f):
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                gpu_max[int(parts[0])] = max(gpu_max[int(parts[0])], int(parts[1]))
    pid_gpus = defaultdict(lambda: defaultdict(int))
    f = os.path.join(root, cell, "smi_apps.txt")
    if os.path.exists(f):
        for raw in open(f):
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                g = bus.get(parts[1].lower(), parts[1])
                pid_gpus[int(parts[0])][g] = max(pid_gpus[int(parts[0])][g], int(parts[2] or 0))
    stray = {pid: dict(g) for pid, g in pid_gpus.items() if len(g) > 1}
    return dict(gpu_max), stray


def main():
    root, cells = sys.argv[1], sys.argv[2:]
    bus = bus_to_index()
    data = {c: steps(root, c) for c in cells}
    ref = cells[0]
    print("loss / grad norm per step (rank 7)")
    print("| step | " + " | ".join(cells) + " |")
    print("|---:|" + "---|" * len(cells))
    for s in sorted(data[ref]):
        print(f"| {s} | " + " | ".join(
            f"{data[c][s][0]}/{data[c][s][1]}" if s in data[c] else "-" for c in cells) + " |")
    for c in cells[1:]:
        same = [s for s in data[ref] if s in data[c] and data[c][s][:2] == data[ref][s][:2]]
        print(f"{c} vs {ref}: identical on {len(same)}/{len(data[ref])} steps")
    print("\nstep time from rank 7's log timestamps, s")
    print("| cell | " + " | ".join(f"step {s}" for s in range(2, 7)) + " | mean 3-6 |")
    print("|---|" + "---:|" * 6)
    for c in cells:
        d = data[c]
        dt = {s: d[s][2] - d[s - 1][2] for s in range(2, 7) if s in d and s - 1 in d}
        mean = [dt[s] for s in range(3, 7) if s in dt]
        tail = f" | {sum(mean) / len(mean):.2f} |" if mean else " | - |"
        print(f"| {c} | " + " | ".join(f"{dt[s]:.2f}" if s in dt else "-" for s in range(2, 7)) + tail)
    print("\nnvidia-smi highest memory.used per GPU during the cell, GiB (contexts and cache included)")
    rows = {c: smi(root, c, bus) for c in cells}
    print("| GPU | " + " | ".join(cells) + " |")
    print("|---:|" + "---:|" * len(cells))
    for g in range(8):
        print(f"| {g} | " + " | ".join(
            f"{rows[c][0].get(g, 0) / 1024:.2f}" if rows[c][0] else "-" for c in cells) + " |")
    for c in cells:
        if rows[c][1]:
            print(f"{c}: processes on more than one GPU (MiB per GPU): {rows[c][1]}")
        elif rows[c][0]:
            print(f"{c}: every process holds memory on one GPU only")


if __name__ == "__main__":
    main()
