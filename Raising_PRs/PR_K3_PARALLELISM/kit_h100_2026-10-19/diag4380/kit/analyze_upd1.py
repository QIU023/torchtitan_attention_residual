"""Main against the PR for each diagnosis variant: step-1 gradients, parameters after the first update, clip norm, step 2."""
import re
import sys
from pathlib import Path

import torch

D = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")


def log(cell):
    text = ansi.sub("", (D / cell / "run.log").read_text(errors="replace"))
    norm = re.search(r"UPD1_GRAD_NORM (\S+)", text)
    steps = dict(re.findall(r"step:\s*(\d+)\s+loss:\s*(\S+)", text))
    return (norm.group(1) if norm else "-"), steps


def compare(a, b):
    out = []
    for group in ("tower", "rest"):
        keys = [k for k in a if ("vision_encoder" in k) == (group == "tower")]
        same = sum(torch.equal(a[k], b[k]) for k in keys)
        rel = [
            (b[k].float() - a[k].float()).abs().max().item() / max(a[k].float().abs().max().item(), 1e-30)
            for k in keys
            if not torch.equal(a[k], b[k])
        ]
        out.append(f"{group} {same}/{len(keys)}" + (f" (max rel {max(rel):.2g})" if rel else ""))
    return ", ".join(out)


for variant in sorted({p.name.split("__")[1] for p in D.glob("*__*") if p.is_dir()}):
    ref, cells = f"main__{variant}", [p.name for p in D.glob(f"*__{variant}") if not p.name.startswith("main__")]
    if not (D / ref / "dump" / "grad1.pt").exists():
        continue
    rn, rs = log(ref)
    print(f"== {variant}: main clip norm {rn}, step 1 {rs.get('1')}, step 2 {rs.get('2')}")
    for cell in sorted(cells):
        if not (D / cell / "dump" / "param1.pt").exists():
            print(f"   {cell}: no dump")
            continue
        n, s = log(cell)
        g = compare(torch.load(D / ref / "dump" / "grad1.pt"), torch.load(D / cell / "dump" / "grad1.pt"))
        p = compare(torch.load(D / ref / "dump" / "param1.pt"), torch.load(D / cell / "dump" / "param1.pt"))
        print(f"   {cell}: clip norm {n} ({'same' if n == rn else 'DIFFERENT'}), step 1 {s.get('1')}, step 2 {s.get('2')}")
        print(f"      step-1 gradients vs main: {g}")
        print(f"      parameters after update 1 vs main: {p}")
