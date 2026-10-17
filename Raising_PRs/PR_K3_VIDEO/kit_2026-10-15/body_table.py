"""The video PR body's training table from a matrix run: loss / grad norm at steps 1, 5 and 10 per layout, blank when absent."""
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
ROWS = [
    ("dp1", "dp1"), ("pp2", "pp2"), ("pp2_vpp2", "pp2×vpp2"), ("pp4", "pp4"), ("pp8", "pp8"),
    ("tp2_ep2", "tp2×ep2"), ("tp4_ep4", "tp4×ep4"), ("cp2", "cp2"), ("tc_cp2", "cp2, type-checked"),
    ("ep2_tp2_pp2_cp2", "ep2×tp2×pp2×cp2"), ("cp4", "cp4, seq 512"),
    ("fsdp2", "fsdp2"), ("ep2_fsdp2", "ep2×fsdp2"), ("tc_ep2_fsdp2_tp2", "ep2×fsdp2×tp2, type-checked"),
    ("ep4_fsdp2_tp2_pp2", "ep4×fsdp2×tp2×pp2"), ("ep2_fsdp2_tp2_pp2", "ep2×fsdp2×tp2×pp2"),
    ("fsdp2_pp2_cp2", "fsdp2×pp2×cp2"), ("ep2_fsdp2_pp2_cp2", "ep2×fsdp2×pp2×cp2"),
    ("ep4_fsdp2_tp2_cp2", "ep4×fsdp2×tp2×cp2"), ("ep2_fsdp2_tp2_cp2", "ep2×fsdp2×tp2×cp2"),
    ("ep8_fsdp8", "ep8×fsdp8"), ("mix_ep2_fsdp2", "ep2×fsdp2, cc12m-test images mixed in"),
]
progress = re.sub(r"^\S+ ", "", (out / "progress.txt").read_text(), flags=re.M)
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"step:\s*(\d+)\s+loss:\s*(-?[0-9.]+)\s+grad_norm:\s*([0-9.]+)")
print("| Layout | Step 1 | Step 5 | Step 10 |\n|---|---|---|---|")
for name, label in ROWS:
    log = out / name / "run.log"
    vals: dict[int, tuple[str, str]] = {}
    done = re.search(rf"(^|rerun ){re.escape(name)} rc=0 last step=10 ", progress, re.M)
    if done and log.exists():
        for line in ansi.sub("", log.read_text(errors="replace")).splitlines():
            m = pat.search(line)
            if m and float(m.group(2)) >= 0:
                vals.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    cells = [" / ".join(vals[s]) if s in vals else "" for s in (1, 5, 10)]
    print(f"| {label} | " + " | ".join(cells) + " |")
