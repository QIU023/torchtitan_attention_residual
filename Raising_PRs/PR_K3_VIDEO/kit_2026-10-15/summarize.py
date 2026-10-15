"""Per cell: rc line from progress.txt, step 1 / 5 / 10 loss from the last pipeline stage (the max over ranks)."""
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
progress = {m.group(1): m.group(2) for m in re.finditer(r"(\S+) rc=(\d+)", (out / "progress.txt").read_text())}
ansi = re.compile(r"\x1b\[[0-9;]*m")
for name, rc in progress.items():
    log = out / name / "run.log"
    if name == "seed" or not log.exists():
        continue
    losses: dict[int, float] = {}
    for line in ansi.sub("", log.read_text(errors="replace")).splitlines():
        m = re.search(r"step:\s*(\d+)\s+loss:\s*(-?[0-9.]+)\s+grad_norm:\s*([0-9.]+)", line)
        if m:
            step, loss = int(m.group(1)), float(m.group(2))
            losses[step] = max(losses.get(step, -1e9), loss)
    cols = " ".join(f"{losses.get(s, float('nan')):.5f}" for s in (1, 5, 10))
    print(f"{name:22s} rc={rc} steps={max(losses) if losses else 0:2d}  loss@1/5/10 {cols}")
