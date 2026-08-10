"""Compare a fresh matrix table against a published one, parsing BOTH from files.

The baseline is read out of the reference markdown rather than retyped, because a
transcribed baseline cannot be distinguished from a transcription error, and this
project has already lost a matrix to a collector bug.

Status changes are reported before numbers: a cell that went pass -> FAIL or
FAIL -> pass is the finding, and a relative difference computed against a missing
row is not.

Usage:
    python compare_to_baseline.py <new_collected.txt> [--baseline MATRIX_18_SDPA_2026-08-09.md]

`new_collected.txt` is whatever `collect13.sh` printed. Rows are
`<cell> <loss1> <loss2> ...` or `<cell> FAIL (n/m) <error>`.
"""

from __future__ import annotations

import argparse
import pathlib
import re


_CELL = re.compile(r"^(\S+)\s+(.*)$")


def parse_rows(text: str) -> dict[str, list[float] | str]:
    """Rows keyed by cell name. A failed cell maps to its FAIL string."""
    rows: dict[str, list[float] | str] = {}
    for line in text.splitlines():
        line = line.rstrip()
        match = _CELL.match(line)
        if not match:
            continue
        name, rest = match.group(1), match.group(2).strip()
        if name.startswith("#") or name in {"step1", "###"}:
            continue
        if rest.startswith(("FAIL", "MISSING")):
            rows[name] = rest
            continue
        values = re.findall(r"-?\d+\.\d+", rest)
        # A real row is a run of losses; anything shorter is prose that happened
        # to start with a word and a number. The max-degree table is the one
        # exception -- it publishes two points, labelled, so accept those on the
        # label rather than on the count.
        if "step1" in rest and len(values) == 2:
            rows[name] = [float(values[0]), float(values[1])]
        elif len(values) >= 3:
            rows[name] = [float(v) for v in values]
    return rows


def parse_baseline(path: pathlib.Path) -> dict[str, list[float] | str]:
    """Baseline table out of the reference doc's fenced blocks."""
    blocks = re.findall(r"```(.*?)```", path.read_text(), flags=re.S)
    rows: dict[str, list[float] | str] = {}
    for block in blocks:
        rows.update(parse_rows(block))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new")
    parser.add_argument(
        "--baseline",
        default=str(
            pathlib.Path(__file__).resolve().parent.parent
            / "MATRIX_18_SDPA_2026-08-09.md"
        ),
    )
    args = parser.parse_args()

    new = parse_rows(pathlib.Path(args.new).read_text())
    base = parse_baseline(pathlib.Path(args.baseline))
    if not base:
        raise SystemExit(f"no baseline rows parsed from {args.baseline}")

    print(f"baseline rows: {len(base)}   new rows: {len(new)}\n")

    changed = []
    for cell in sorted(set(base) | set(new)):
        b, n = base.get(cell), new.get(cell)
        b_ok, n_ok = isinstance(b, list), isinstance(n, list)
        if b is None or n is None:
            changed.append((cell, f"only in {'new' if b is None else 'baseline'}"))
        elif b_ok != n_ok:
            changed.append((cell, "pass -> FAIL" if b_ok else "FAIL -> pass"))
    if changed:
        print("STATUS CHANGES")
        for cell, what in changed:
            print(f"  {cell:24} {what}")
        print()

    print(f"{'cell':24} {'step1 rel':>10} {'last rel':>10}  baseline -> new (step 1)")
    for cell in sorted(set(base) & set(new)):
        b, n = base[cell], new[cell]
        if not (isinstance(b, list) and isinstance(n, list)):
            continue
        rel1 = abs(n[0] - b[0]) / abs(b[0]) if b[0] else float("nan")
        k = min(len(b), len(n)) - 1
        rel_last = abs(n[k] - b[k]) / abs(b[k]) if b[k] else float("nan")
        print(
            f"{cell:24} {rel1:>10.2e} {rel_last:>10.2e}  "
            f"{b[0]:.5f} -> {n[0]:.5f}"
        )


if __name__ == "__main__":
    main()
