"""Pairs of run_e2e_sac_1001.sh: python cmp_sac_1001.py <results dir>

For each activation checkpointing mode (the recipe's SelectiveAC, FullAC), loss and grad norm per step of the
standard and MoonEP cells (one warm cache, deterministic), whether step 1 is equal, and the relative loss gap;
then the steps and exit code of the MoonEP cell with SPMD type checking on.
"""

import re
import sys
from pathlib import Path

STEP = re.compile(r"step:\s+(\d+)\s+loss:\s+(-?[\d.]+)\s+grad_norm:\s+([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read(path: Path) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    if path.exists():
        for line in ANSI.sub("", path.read_text(errors="replace")).splitlines():
            m = STEP.search(line)
            if m and float(m[2]) > 0:
                out.setdefault(int(m[1]), (m[2], m[3]))
    return out


def rc(root: Path, name: str) -> str:
    p = root / name / "rc"
    return p.read_text().strip() if p.exists() else "-"


def main() -> None:
    root = Path(sys.argv[1])
    for ac in ("sac", "full"):
        s, m = read(root / f"std_{ac}" / "run.log"), read(root / f"moon_{ac}" / "run.log")
        same = sum(1 for k in s if m.get(k) == s[k])
        print(f"## {ac}: standard {rc(root, f'std_{ac}')}, moonep {rc(root, f'moon_{ac}')}; "
              f"step 1 equal: {1 in s and s.get(1) == m.get(1)}; steps equal {same} / {len(s)}")
        print("| step | standard loss / grad norm | MoonEP loss / grad norm | relative loss gap |")
        print("|---:|---|---|---:|")
        for k in sorted(s):
            if k in m:
                gap = abs(float(m[k][0]) - float(s[k][0])) / float(s[k][0])
                print(f"| {k} | {s[k][0]} / {s[k][1]} | {m[k][0]} / {m[k][1]} | {gap:.2e} |")
        print()
    tc = read(root / "moon_typecheck" / "run.log")
    print(f"typecheck cell ({rc(root, 'moon_typecheck')}): steps {sorted(tc)}")


if __name__ == "__main__":
    main()
