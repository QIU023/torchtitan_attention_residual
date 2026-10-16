"""Bank-swap table on the review tree: logged loss and grad norm per step, and the step-1 gradient and final-parameter
dumps compared tensor by tensor (vision tower vs the rest)."""
import re
import sys
from pathlib import Path

import torch

O = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"step:\s*(\d+)\s+loss:\s*(-?[0-9.]+)\s+grad_norm:\s*([0-9.]+)")
cells = ["off", "whole", "split", "swap"]


def series(name):
    out = {}
    for line in ansi.sub("", (O / f"{name}_meas" / "run.log").read_text(errors="replace")).splitlines():
        m = pat.search(line)
        if m:
            out.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    return out


logs = {c: series(c) for c in cells}
print("step | " + " | ".join(f"{c} loss / gn" for c in cells))
for s in sorted(logs["off"]):
    print(f"{s:4d} | " + " | ".join(" / ".join(logs[c].get(s, ("-", "-"))) for c in cells))


def load(c, tag, r=0):
    return torch.load(O / f"{c}_meas" / "dump" / f"{tag}_rank{r}.pt")


def compare(a, b, label):
    tower = [k for k in a if "vision_encoder" in k]
    rest = [k for k in a if "vision_encoder" not in k]
    for group, keys in (("tower", tower), ("rest", rest)):
        same = sum(torch.equal(a[k], b[k]) for k in keys)
        worst, worst_k = 0.0, None
        for k in keys:
            x, y = a[k].double(), b[k].double()
            n = y.norm().item()
            rel = (x - y).norm().item() / n if n > 0 else (x - y).norm().item()
            if rel > worst:
                worst, worst_k = rel, k
        print(f"{label:28s} {group:5s} bitwise {same}/{len(keys)}  worst rel {worst:.3e} {worst_k or ''}")


g = {c: load(c, "grad1") for c in cells}
p = {c: load(c, "param_last") for c in cells}
print("\nstep-1 gradients")
compare(g["swap"], g["whole"], "swap vs whole")
compare(g["whole"], g["off"], "whole vs off")
compare(g["split"], g["whole"], "split vs whole")
compare(g["split"], g["off"], "split vs off")
print("\nparameters after step 10")
compare(p["swap"], p["whole"], "swap vs whole")
compare(p["whole"], p["off"], "whole vs off")
print("\nCP rank 0 vs rank 1, parameters after step 10")
for c in cells:
    compare(load(c, "param_last", 0), load(c, "param_last", 1), f"{c} rank0 vs rank1")
