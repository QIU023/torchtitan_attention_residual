"""Tables for the #4656 matrix: main's loss and grad norm at steps 1/10/20, and the number of the
100 steps whose logged loss and grad norm on #4656 equal main's."""

import re
import sys
from pathlib import Path

STEP = re.compile(r"\[rank0\].*step:\s+(\d+)\s+loss:\s+([\d.]+)\s+grad_norm:\s+([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def steps(path: Path) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    if path.exists():
        for line in ANSI.sub("", path.read_text(errors="replace")).splitlines():
            m = STEP.search(line)
            if m:
                out.setdefault(int(m[1]), (m[2], m[3]))
    return out


def main(root: Path) -> None:
    names = sorted({p.name[5:] for p in root.glob("main_*")})
    print("| cell | loss 1 / 10 / 20 (main) | grad norm 1 / 10 / 20 (main) | #4656 steps equal to main |")
    print("|---|---|---|---:|")
    for name in names:
        a, b = steps(root / f"main_{name}" / "run.log"), steps(root / f"pr_{name}" / "run.log")
        loss = " / ".join(a.get(s, ("-", "-"))[0] for s in (1, 10, 20))
        gn = " / ".join(a.get(s, ("-", "-"))[1] for s in (1, 10, 20))
        equal = sum(1 for s in a if b.get(s) == a[s])
        print(f"| {name} | {loss} | {gn} | {equal} / {len(a)} |")
    fresh = steps(root / "pr_t1_tp1_fresh" / "run.log")
    ref = steps(root / "main_t1_tp1" / "run.log")
    if fresh:
        print(f"\n#4656 tp1 on a fresh cache: {sum(1 for s in ref if fresh.get(s) == ref[s])} / {len(ref)} steps equal to main")
    for p in sorted(root.glob("*_tc_*")):
        print(f"{p.name}: {(p / 'rc').read_text().strip() if (p / 'rc').exists() else 'no rc'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
