"""Where each rank's peak falls, from the per-action traces of two cells (PPMEM_ACTION_TRACE).

Usage: python diag_actions.py <LB_ROOT> <cell_a> <cell_b>

Per rank: the action with the highest in-action peak in each cell, that peak, the rank store's
bytes after that action, and the store's own high-water mark over the step.
"""

import json
import os
import sys


def _load(root, cell):
    out = {}
    folder = os.path.join(root, cell, "mem")
    for name in os.listdir(folder):
        if name.startswith("actions"):
            with open(os.path.join(folder, name)) as f:
                data = json.load(f)
            out[data["rank"]] = data["actions"]
    return out


def main():
    root, a, b = sys.argv[1:4]
    cells = {a: _load(root, a), b: _load(root, b)}
    print(
        "| rank | "
        + " | ".join(f"{c}: peak action, peak, store then, store max" for c in cells)
        + " | peak change |"
    )
    print("|---:|" + "---|" * len(cells) + "---:|")
    for rank in sorted(cells[a]):
        row, peaks = [], []
        for cell, traces in cells.items():
            actions = traces[rank]
            top = max(actions, key=lambda x: x["peak_gib"])
            store_max = max(x["store_gib"] for x in actions)
            row.append(
                f"{top['action']}, {top['peak_gib']:.2f}, {top['store_gib']:.2f}, {store_max:.2f}"
            )
            peaks.append(top["peak_gib"])
        print(f"| {rank} | " + " | ".join(row) + f" | {peaks[1] - peaks[0]:+.2f} |")


if __name__ == "__main__":
    main()
