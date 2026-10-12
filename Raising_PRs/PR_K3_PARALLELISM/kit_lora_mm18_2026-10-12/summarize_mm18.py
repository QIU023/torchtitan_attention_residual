"""Summarize one or more 18-cell matrix arm dirs: per cell rc, steps, seed loads and loss at steps 1 / 5 / 10, grouped
by data-parallel degree (cells with the same dp read the same data), with the step-1 gap to the group's reference.
usage: summarize_mm18.py <arm dir> [<arm dir> ...]"""
import pathlib
import re
import sys

PAT = re.compile(r"\[rank(\d+)\]:.*?step:\s*(\d+)\s+loss:\s*([-\d.e]+)\s+grad_norm:\s*([-\d.e]+)")
CELLS = [  # name, dp
    ("dp1", 1), ("pp2", 1), ("pp4", 1), ("pp8", 1), ("tp2_ep2", 1), ("tp4_ep4", 1), ("cp2", 1), ("cp4", 1),
    ("ep2_tp2_pp2_cp2", 1),
    ("fsdp2", 2), ("ep2_fsdp2", 2), ("ep4_fsdp2_tp2_pp2", 2), ("ep2_fsdp2_tp2_pp2", 2), ("fsdp2_pp2_cp2", 2),
    ("ep2_fsdp2_pp2_cp2", 2), ("ep4_fsdp2_tp2_cp2", 2), ("ep2_fsdp2_tp2_cp2", 2),
    ("ep8_fsdp8", 8),
]


def read(arm: pathlib.Path, cell: str):
    log = arm / cell / "run.log"
    if not log.exists():
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    steps = {}
    for rank, step, loss, grad in PAT.findall(text):
        if not loss.startswith("-"):
            steps.setdefault(int(step), (loss, grad))
    return steps, text.count("Loading the checkpoint")


for arm_dir in sys.argv[1:]:
    arm = pathlib.Path(arm_dir)
    rcs = {}
    for line in (arm / "progress.txt").read_text().splitlines():
        m = re.search(r" (\S+) rc=(\d+) .*seed_loaded=(\d+)", line)
        if m:
            rcs[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    print(f"== {arm.name}")
    ok = 0
    refs = {}
    for cell, dp in CELLS:
        got = read(arm, cell)
        rc, loads = rcs.get(cell, (None, 0))
        if got is None or not got[0]:
            print(f"  {cell:22s} dp{dp} rc={rc} no loss lines")
            continue
        steps, _ = got
        last = max(steps)
        ok += rc == 0 and last == 10 and loads > 0
        refs.setdefault(dp, float(steps[1][0]))
        gap = abs(float(steps[1][0]) - refs[dp]) / refs[dp]
        cols = " / ".join(steps[s][0] if s in steps else "-" for s in (1, 5, 10))
        print(f"  {cell:22s} dp{dp} rc={rc} steps={last:2d} seed_loads={loads} loss 1/5/10 {cols}  step-1 gap to dp{dp} ref {gap:.1e}")
    print(f"  {ok} / {len(CELLS)} cells ran 10 steps from the seed")
