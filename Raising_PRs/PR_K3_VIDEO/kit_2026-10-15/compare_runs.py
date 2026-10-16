"""Compare two runs of the same cells: loss and grad norm at every step (last pipeline stage), at log precision."""
import re
import sys
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"step:\s*(\d+)\s+loss:\s*(-?[0-9.]+)\s+grad_norm:\s*([0-9.]+)")


def series(log: Path) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for line in ansi.sub("", log.read_text(errors="replace")).splitlines():
        m = pat.search(line)
        if m:
            step, loss, gn = int(m.group(1)), m.group(2), m.group(3)
            if step not in out or float(loss) > float(out[step][0]):
                out[step] = (loss, gn)
    return out


old, new = Path(sys.argv[1]), Path(sys.argv[2])
same = 0
for cell in sorted(p.name for p in new.iterdir() if (p / "run.log").exists() and p.name != "seed"):
    if not (old / cell / "run.log").exists():
        continue
    a, b = series(old / cell / "run.log"), series(new / cell / "run.log")
    steps = sorted(set(a) & set(b))
    diff = [s for s in steps if a[s] != b[s]]
    same += not diff and bool(steps)
    first = f"first differing step {diff[0]}: {a[diff[0]]} vs {b[diff[0]]}" if diff else "identical"
    print(f"{cell:22s} steps {len(steps):2d} step1 {b[steps[0]][0] if steps else '-'}  {first}")
print(f"{same} cells identical on every step")
