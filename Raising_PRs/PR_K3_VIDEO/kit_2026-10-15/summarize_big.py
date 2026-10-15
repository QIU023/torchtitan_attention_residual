"""Per cell: rc, loss and grad norm at steps 1/2/last (last pipeline stage), per-rank peak memory (max over steps), step time."""
import re
import sys
from datetime import datetime
from pathlib import Path

out = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
rcs = dict(re.findall(r"(\S+) rc=(\d+)", (out / "progress.txt").read_text()))
pat = re.compile(r"\[rank(\d+)\].*?(\d\d:\d\d:\d\d),\d+ - .*step:\s*(\d+)\s+loss:\s*(-?[0-9.]+)\s+grad_norm:\s*([0-9.]+)\s+memory:\s*([0-9.]+)GiB")
for name, rc in rcs.items():
    log = out / name / "run.log"
    if name == "seed" or not log.exists():
        continue
    loss, gn, mem, times = {}, {}, {}, {}
    for line in ansi.sub("", log.read_text(errors="replace")).splitlines():
        m = pat.search(line)
        if not m:
            continue
        rank, t, step, l, g, gib = int(m.group(1)), m.group(2), int(m.group(3)), float(m.group(4)), m.group(5), float(m.group(6))
        if l > loss.get(step, -1e9):
            loss[step], gn[step] = l, g
        mem[rank] = max(mem.get(rank, 0.0), gib)
        times.setdefault(step, t)
    steps = sorted(loss)
    if not steps:
        print(f"{name:14s} rc={rc} no steps")
        continue
    last = steps[-1]
    secs = None
    if len(steps) >= 3:
        t0 = datetime.strptime(times[steps[1]], "%H:%M:%S")
        t1 = datetime.strptime(times[last], "%H:%M:%S")
        secs = (t1 - t0).total_seconds() / (last - steps[1])
    cols = " ".join(f"{s}:{loss[s]:.5f}/{gn[s]}" for s in (1, 2, last) if s in loss)
    mems = " ".join(f"{mem[r]:.2f}" for r in sorted(mem))
    print(f"{name:14s} rc={rc} steps={last} {cols} | peak GiB by rank {mems} | {secs:.1f} s/step" if secs else f"{name:14s} rc={rc} steps={last} {cols} | peak GiB by rank {mems}")
