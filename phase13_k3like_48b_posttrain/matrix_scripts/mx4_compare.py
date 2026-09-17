"""Compare two mx4 matrix runs cell by cell, step 1 first.

    python mx4_compare.py <baseline dir> <new dir>

Step 1 is the correctness bar: two trees that differ only in code that should not
move numerics must agree there bitwise, cell for cell. Steps 3 and 10 are shown
because the body shows them, never argued from. A cell missing on either side, or
one whose seed assertion failed, is called out rather than silently skipped.
"""

from __future__ import annotations

import pathlib
import re
import sys

CELL = re.compile(r"^(\S+)\s+(seed-ok|ASSERT-SEED-FAIL|ABORT\S*)\s+(?:rc=(\S+))?\s*(.*)$")
STEP = re.compile(r"s(\d+)=(\S+)")


def read(path: pathlib.Path) -> dict[str, dict]:
    cells: dict[str, dict] = {}
    if not path.is_file():
        return cells
    for line in path.read_text().splitlines():
        m = CELL.match(line)
        if not m:
            continue
        name, seed, rc, rest = m.groups()
        steps = {int(k): v for k, v in STEP.findall(rest)}
        cells[name] = {"seed": seed, "rc": rc, "steps": steps}
    return cells


def main() -> None:
    base_dir, new_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    base, new = read(base_dir / "results.txt"), read(new_dir / "results.txt")
    names = sorted(set(base) | set(new))
    print(f"baseline {base_dir.name}: {len(base)} cells; new {new_dir.name}: {len(new)} cells")
    print("| cell | step 1 baseline | step 1 new | step 1 equal | step 3 | step 10 |")
    print("| --- | --- | --- | :---: | --- | --- |")
    moved, missing = [], []
    for name in names:
        b, n = base.get(name), new.get(name)
        if b is None or n is None:
            missing.append(f"{name} ({'new only' if b is None else 'baseline only'})")
            continue
        b1, n1 = b["steps"].get(1, "n/a"), n["steps"].get(1, "n/a")
        same = b1 == n1 and b1 != "n/a"
        if not same:
            moved.append(name)
        flags = []
        if b["seed"] != "seed-ok" or n["seed"] != "seed-ok":
            flags.append(f"seed {b['seed']}/{n['seed']}")
        if (b["rc"] or "0") != "0" or (n["rc"] or "0") != "0":
            flags.append(f"rc {b['rc']}/{n['rc']}")
        suffix = (" " + "; ".join(flags)) if flags else ""
        print(
            f"| {name}{suffix} | {b1} | {n1} | {'yes' if same else 'NO'} | "
            f"{b['steps'].get(3, 'n/a')} vs {n['steps'].get(3, 'n/a')} | "
            f"{b['steps'].get(10, 'n/a')} vs {n['steps'].get(10, 'n/a')} |"
        )
    print()
    print(f"cells compared: {len(names) - len(missing)}; step 1 identical: {len(names) - len(missing) - len(moved)}")
    if moved:
        print("step 1 moved, locate before writing any of it up: " + ", ".join(moved))
    if missing:
        print("not on both sides: " + ", ".join(missing))


if __name__ == "__main__":
    main()
