"""Build the #4380 body table from a results dir (run_matrix2 layout: <dir>/ag/<cell>/run.log, <dir>/ul/<cell>/run.log);
logbook kit only. Prints the markdown table and the facts the sentences below it state."""
import hashlib
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
STEP = re.compile(r"step: +(\d+) +loss: +([0-9.]+) +grad_norm: +([0-9.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def steps(group, cell):
    out = {}
    for line in (root / group / cell / "run.log").read_text(errors="replace").splitlines():
        line = ANSI.sub("", line)
        if line.startswith("[rank0]") and (m := STEP.search(line)):
            out[int(m.group(1))] = (m.group(2), m.group(3))
    return out


def first_diff(a, b):
    return next((s for s in sorted(a) if a[s] != b.get(s)), None)


def digest(t):
    return hashlib.md5(repr(sorted(t.items())).encode()).hexdigest()[:8]


cells = {
    "main, CP=2 all-gather": ("ag", "ag_main"),
    "main, CP=2 all-gather, second run": ("ag", "ag_main_b"),
    "#4380, CP=2 all-gather": ("ag", "ag_cpmm"),
    "#4380, CP=2 all-gather, every image split": ("ag", "ag_split"),
    "main, CP=2 Ulysses": ("ul", "ul_main"),
    "#4380, CP=2 Ulysses": ("ul", "ul_cpmm"),
}
data = {k: steps(*v) for k, v in cells.items()}
cols = (1, 10, 50, 100)
print("| Configuration | " + " | ".join(f"Step {c}" for c in cols) + " |")
print("|---|" + "---|" * len(cols))
for name, t in data.items():
    print(f"| {name} | " + " | ".join(f"{t[c][0]} / {t[c][1]}" if c in t else "missing" for c in cols) + " |")
print()
ref_ag, ref_ul = data["main, CP=2 all-gather"], data["main, CP=2 Ulysses"]
print("steps per cell:", {k: len(v) for k, v in data.items()})
print("noise floor (main twice) identical:", digest(ref_ag) == digest(data["main, CP=2 all-gather, second run"]))
print("#4380 AG first differs from main at step", first_diff(ref_ag, data["#4380, CP=2 all-gather"]))
print("#4380 UL first differs from main UL at step", first_diff(ref_ul, data["#4380, CP=2 Ulysses"]))
print("every-image-split first differs from main at step", first_diff(ref_ag, data["#4380, CP=2 all-gather, every image split"]))
print("main AG vs main UL first differ at step", first_diff(ref_ag, ref_ul))
swap = steps("ag", "ag_swap")
print("bank swap identical to main AG:", digest(swap) == digest(ref_ag), "steps", len(swap))
cp1m, cp1c = steps("ul", "cp1_main"), steps("ul", "cp1_cpmm")
print("CP=1 #4380 identical to main:", digest(cp1m) == digest(cp1c), "steps", len(cp1m), len(cp1c))
