#!/usr/bin/env python3
"""Diff re-run CP cells against a baseline, per step.

    python3 cmp_cp_cells.py <out_dir> <baseline> [<baseline_out_dir>]

<baseline> is either a gate percell.txt or a directory of run logs.

The CP declarative rewrite touched no numerical path, so the bar here is
IDENTICAL, not close. Any difference means the rewrite reached something it was
not supposed to reach.

Two things this parser gets right that a naive one does not:

* Every rank logs the same step, so a raw match count is 2x (or 8x) the step
  count. Dedup is keyed on the STEP NUMBER, not on adjacent-duplicate collapsing
  the way `uniq` does it -- two consecutive steps whose loss and grad_norm both
  round to the same five digits are rare but real, and uniq would silently eat
  one and report a short run.
* Ranks are asserted to agree. They always have here, but a CP bug that
  desynchronised them would otherwise be invisible: the first rank's numbers
  would match the baseline and the run would look clean.

Caveat that limits what a match proves: stdout carries five significant digits,
so two runs can agree here and still differ below that. For a stronger claim the
comparison has to come from the TensorBoard record (upstream
scripts/loss_compare.py).
"""

import re
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"(?:step:\s*)?(\d+)\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)")


def parse_steps(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """-> ([(loss, grad_norm) by step], [disagreement notes])."""
    text = ANSI_RE.sub("", text)
    by_step: dict[int, tuple[str, str]] = {}
    notes: list[str] = []
    for step_s, loss, gnorm in STEP_RE.findall(text):
        step = int(step_s)
        if step in by_step:
            if by_step[step] != (loss, gnorm):
                notes.append(
                    f"step {step}: ranks disagree {by_step[step]} vs {(loss, gnorm)}"
                )
            continue
        by_step[step] = (loss, gnorm)
    return [by_step[k] for k in sorted(by_step)], notes


def parse_baseline(path: Path) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """percell.txt -> {(arm, cell): steps}, CP cells only."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        arm, _matrix, cell, rest = parts
        if "cp" not in cell:
            continue
        steps, _ = parse_steps(rest)
        if steps:
            out[(arm, cell)] = steps
    return out


ARMS = ("mm_full", "mm_lora", "text")


def load_dir(path: Path) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Load run logs from either layout, CP cells only.

    flat: ``<arm>_<cell>.log``            -- what run_cp_cells.sh writes
    gate: ``<arm>_{13,max}/<cell>.log``   -- what run_postmerge_gate.sh writes

    Both are read here rather than renaming files into one shape outside, because
    a rename step is where an arm and a cell get paired up wrongly and the
    comparison silently comes out clean.
    """
    out = {}
    for sub in sorted(path.iterdir()):
        if not sub.is_dir():
            continue
        arm = next((a for a in ARMS if sub.name == a + "_13" or sub.name == a + "_max"), None)
        if arm is None:
            continue
        for log in sorted(sub.glob("*.log")):
            if "cp" not in log.stem:
                continue
            steps, _ = parse_steps(log.read_text(errors="replace"))
            if steps:
                out[(arm, log.stem)] = steps
    if out:
        return out
    # Arm names contain underscores (mm_full), and so do cell names
    # (fsdp2_pp2_cp2), so the flat split has to know the arms rather than guess.
    for log in sorted(path.glob("*.log")):
        arm = next((a for a in ARMS if log.stem.startswith(a + "_")), None)
        if arm is None:
            continue
        cell = log.stem[len(arm) + 1 :]
        if "cp" not in cell:
            continue
        steps, _ = parse_steps(log.read_text(errors="replace"))
        if steps:
            out[(arm, cell)] = steps
    return out


parse_baseline_dir = load_dir


def main() -> int:
    out_dir, baseline_path = Path(sys.argv[1]), Path(sys.argv[2])
    baseline = (
        load_dir(baseline_path) if baseline_path.is_dir() else parse_baseline(baseline_path)
    )
    if not baseline:
        print(f"no CP cells parsed out of {baseline_path} -- wrong format?")
        return 2
    current = load_dir(out_dir)

    # Cells proven not to reproduce themselves cannot support a before/after claim in
    # either direction. Reported separately rather than dropped, so the exclusion stays
    # visible in the output instead of living only in someone's memory.
    UNRELIABLE = {("mm_full", "tp2_pp2_cp2")}

    identical = differing = missing = 0
    for (arm, cell), want in sorted(baseline.items()):
        if (arm, cell) in current:
            got = current[(arm, cell)]
        else:
            print(f"  MISSING  {arm:9s} {cell:20s} (no log -- did not run)")
            missing += 1
            continue
        if (arm, cell) in UNRELIABLE:
            verdict = "same" if got == want else "differs"
            print(
                f"  EXCLUDED {arm:9s} {cell:20s} ({verdict}; backward is "
                "nondeterministic -- proves nothing either way)"
            )
            continue
        if got == want:
            identical += 1
            continue
        differing += 1
        print(f"  DIFF     {arm:9s} {cell:20s}")
        for i, (w, g) in enumerate(zip(want, got), 1):
            if w != g:
                print(
                    f"      step {i:2d}  loss {w[0]} -> {g[0]}"
                    f"   grad_norm {w[1]} -> {g[1]}"
                )
        if len(got) != len(want):
            print(f"      step count {len(want)} -> {len(got)}")

    total = len(baseline) - len(UNRELIABLE & set(baseline))
    print(
        f"\n  identical: {identical}/{total}   differing: {differing}   missing: {missing}"
    )
    return 0 if differing == 0 and missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
