"""Whether a rank's step peak falls inside its pipeline actions, from a traced cell.

Usage: python peak_where.py <LB_ROOT> <cell> <step>

Per rank: the step's peak (the probe's record), the highest in-action peak of the traced step
and its action, and the memory left allocated after the step's last action.
"""

import json
import os
import sys


def main():
    root, cell, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
    folder = os.path.join(root, cell, "mem")
    print("| rank | step peak | highest action peak | at | after last action |")
    print("|---:|---:|---:|---|---:|")
    for rank in range(8):
        with open(os.path.join(folder, f"rank{rank}.json")) as f:
            records = json.load(f)["records"]
        path = os.path.join(folder, f"actions{rank}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            actions = json.load(f)["actions"]
        top = max(actions, key=lambda x: x["peak_gib"])
        # the traced step resets the peak per action, so the step before it gives the step peak
        print(
            f"| {rank} | {records[step - 1]['max_allocated_gib']:.2f} | {top['peak_gib']:.2f} | "
            f"{top['action']} | {actions[-1]['after_gib']:.2f} |"
        )


if __name__ == "__main__":
    main()
