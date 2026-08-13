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


def _is_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def read_table(path: str) -> dict[str, list[str] | None]:
    """``{cell: [loss strings]}``, or None for a failed cell.

    Keeps only NUMERIC tokens: the maxdeg runner truncates each row with "..." and also
    emits skip notes ("tp8 / cp8 : only 4 attention heads"), and both were being read as
    cells -- the notes inflated the SAME count and the "..." crashed the float compare.
    A row whose second field is neither a number nor FAIL is not a cell.
    """
    out: dict[str, list[str] | None] = {}
    for line in open(path):
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0]
        if "FAIL" in line:
            out[name] = None
            continue
        losses = [t for t in fields[1:] if _is_number(t)]
        if losses:
            out[name] = losses
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
