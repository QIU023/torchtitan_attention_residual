"""Pass/fail gate for one LoRA arm of the 18-cell matrix: every cell rc 0, 10 steps, the seed loaded, an EXPORT_CHECK
line from every rank with worst 0, and the merged-module total equal to modules * gpus / pp.
usage: gate_mm18.py <arm dir> <LoRA modules per full model>"""
import pathlib
import re
import sys

CELLS = {  # name: (gpus, pp)
    "dp1": (1, 1), "fsdp2": (2, 1), "pp2": (2, 2), "tp2_ep2": (2, 1), "ep2_fsdp2": (2, 1), "cp2": (2, 1),
    "pp4": (4, 4), "tp4_ep4": (4, 1), "cp4": (4, 1), "pp8": (8, 8), "ep8_fsdp8": (8, 1),
    "ep4_fsdp2_tp2_pp2": (8, 2), "ep2_fsdp2_tp2_pp2": (8, 2), "fsdp2_pp2_cp2": (8, 2), "ep2_fsdp2_pp2_cp2": (8, 2),
    "ep4_fsdp2_tp2_cp2": (8, 1), "ep2_tp2_pp2_cp2": (8, 2), "ep2_fsdp2_tp2_cp2": (8, 1),
}
CHECK = re.compile(r"EXPORT_CHECK rank (\d+): (\d+) LoRA modules merged, worst \|merged - expected\| ([-0-9.e+]+)")
arm, modules = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rows = {}
for line in (arm / "progress.txt").read_text().splitlines():
    m = re.search(r" (\S+) rc=(\d+) last step=(\d*) seed_loaded=(\d+)", line)
    if m:
        rows[m.group(1)] = (int(m.group(2)), m.group(3), int(m.group(4)))
passed = 0
for cell, (gpus, pp) in CELLS.items():
    rc, last, loads = rows.get(cell, (None, "", 0))
    log = arm / cell / "run.log"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace")) if log.exists() else ""
    checks = {int(r): (int(n), float(w)) for r, n, w in CHECK.findall(text)}
    total = sum(n for n, _ in checks.values())
    worst = max((w for _, w in checks.values()), default=float("nan"))
    ok = (rc == 0 and last == "10" and loads > 0 and sorted(checks) == list(range(gpus))
          and worst == 0.0 and total == modules * gpus // pp)
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'} {cell:20s} rc={rc} steps={last} seed_loads={loads} export_ranks={len(checks)}/{gpus} "
          f"modules={total}/{modules * gpus // pp} worst={worst:g}")
print(f"{arm.name}: {passed} / {len(CELLS)}")
sys.exit(0 if passed == len(CELLS) else 1)
