#!/usr/bin/env python3
"""Judge a matrix run against the frozen pre-merge (dev-branch) baseline.

The acceptance criterion for the merge + declarative refactor is "the three matrices
agree with the un-merged branch". This does the comparison mechanically, per cell, so
the verdict is not a human reading two tables side by side -- which is how a cell that
had moved in the 5th digit got called byte-identical once already.

Three verdicts per cell, and the distinction matters:

* ``SAME``      -- every printed loss identical. What the refactor should produce.
* ``DRIFT``     -- same length, all differences within --tol. Real but bounded; the
                   upstream merge itself moves some cells here (a reduction moved onto
                   the device, changing summation order), so this is not automatically
                   a failure -- it is a number that has to be explained.
* ``BROKE``     -- was passing in the baseline, is FAIL now. Never acceptable.
* ``FIXED``     -- was FAIL in the baseline, passes now.
* ``MISSING``   -- cell absent from one side, which is itself a finding.

Exit status is 1 if anything BROKE, so a driver script can gate on it.

Usage:
    compare_to_dev_baseline.py BASELINE.txt CANDIDATE.txt [--tol 1e-4] [--label full]
"""

from __future__ import annotations

import argparse
import sys


def read_table(path: str) -> dict[str, list[str] | None]:
    """``{cell: [loss strings]}``, or None for a failed cell."""
    out: dict[str, list[str] | None] = {}
    for line in open(path):
        if not line.strip():
            continue
        fields = line.split()
        name = fields[0]
        out[name] = None if "FAIL" in line else fields[1:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base, cand = read_table(args.baseline), read_table(args.candidate)
    verdicts: dict[str, list[str]] = {}
    worst_drift = 0.0

    for name in sorted(set(base) | set(cand)):
        if name not in base or name not in cand:
            verdicts.setdefault("MISSING", []).append(name)
            continue
        b, c = base[name], cand[name]
        if b is None and c is None:
            verdicts.setdefault("BOTH_FAIL", []).append(name)
        elif b is None:
            verdicts.setdefault("FIXED", []).append(name)
        elif c is None:
            verdicts.setdefault("BROKE", []).append(name)
        elif b == c:
            verdicts.setdefault("SAME", []).append(name)
        elif len(b) != len(c):
            verdicts.setdefault("MISSING", []).append(f"{name}(step count differs)")
        else:
            d = max(abs(float(x) - float(y)) for x, y in zip(b, c))
            worst_drift = max(worst_drift, d)
            key = "DRIFT" if d <= args.tol else "BROKE"
            verdicts.setdefault(key, []).append(f"{name}(max {d:.1e})")

    tag = f"[{args.label}] " if args.label else ""
    for key in ("SAME", "DRIFT", "FIXED", "BOTH_FAIL", "MISSING", "BROKE"):
        if key in verdicts:
            print(f"{tag}{key:9} {len(verdicts[key]):2}  {', '.join(verdicts[key])}")
    if worst_drift:
        print(f"{tag}worst drift {worst_drift:.2e} (tolerance {args.tol:.0e})")

    broke = verdicts.get("BROKE", [])
    print(f"{tag}VERDICT   {'FAIL' if broke else 'PASS'}")
    return 1 if broke else 0


if __name__ == "__main__":
    sys.exit(main())
