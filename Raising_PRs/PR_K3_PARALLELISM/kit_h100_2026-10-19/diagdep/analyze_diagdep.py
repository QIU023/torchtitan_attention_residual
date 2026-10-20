"""DEP on against DEP off, per tree: each rank's step-1 gradients and parameters after the first update, tower apart."""
import sys
from pathlib import Path

import torch

G = Path(sys.argv[1])


def compare(a, b):
    out = []
    for group in ("tower", "rest"):
        keys = [k for k in a if ("vision_encoder" in k) == (group == "tower")]
        if not keys:
            continue
        same = sum(torch.equal(a[k], b[k]) for k in keys)
        diffs = [(b[k].float() - a[k].float()) for k in keys if not torch.equal(a[k], b[k])]
        rels = [
            (b[k].float() - a[k].float()).abs().max().item() / max(a[k].float().abs().max().item(), 1e-30)
            for k in keys
            if not torch.equal(a[k], b[k])
        ]
        frac = sum((d != 0).sum().item() for d in diffs) / max(sum(a[k].numel() for k in keys), 1)
        out.append(f"{group} {same}/{len(keys)}" + (f" (max rel {max(rels):.3g}, {frac:.2%} of elements differ)" if rels else ""))
    return ", ".join(out)


for tree in ("new", "old"):
    ref = G / f"{tree}_w_dep_off" / "dump"
    if not ref.exists():
        print(f"{tree}: no DEP-off dump")
        continue
    for mode in ("w_dep_k25", "w_dep_bubble"):
        d = G / f"{tree}_{mode}" / "dump"
        if not d.exists():
            print(f"{tree} {mode}: no dump")
            continue
        print(f"== {tree}: {mode} vs DEP off")
        for r in range(4):
            g = compare(torch.load(ref / f"grad1_rank{r}.pt"), torch.load(d / f"grad1_rank{r}.pt"))
            p = compare(torch.load(ref / f"param1_rank{r}.pt"), torch.load(d / f"param1_rank{r}.pt"))
            print(f"   rank {r}: step-1 gradients {g}")
            print(f"           parameters after update 1 {p}")
