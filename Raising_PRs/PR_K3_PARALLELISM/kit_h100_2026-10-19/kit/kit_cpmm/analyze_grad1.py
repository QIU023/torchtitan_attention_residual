"""Step-1 gradients of the #4380 H100 cells against main, tower and the rest counted separately (logbook kit only)."""
import sys
from pathlib import Path

import torch

O = Path(sys.argv[1])
ref = torch.load(O / "main_a" / "dump" / "grad1_rank0.pt")
for cell in ["pr_off", "pr_whole", "pr_128", "pr_128_swap"]:
    f = O / cell / "dump" / "grad1_rank0.pt"
    if not f.exists():
        print(f"{cell}: no dump")
        continue
    g = torch.load(f)
    r1 = O / cell / "dump" / "grad1_rank1.pt"
    if r1.exists():
        g1 = torch.load(r1)
        print(f"{cell}: rank 1 equals rank 0 in {sum(torch.equal(g[k], g1[k]) for k in g)}/{len(g)}")
    for group in ("tower", "rest"):
        keys = [k for k in ref if ("vision_encoder" in k) == (group == "tower")]
        same = sum(torch.equal(ref[k], g[k]) for k in keys)
        rel = sorted(
            ((g[k].float() - ref[k].float()).abs().max().item() / max(ref[k].float().abs().max().item(), 1e-30), k)
            for k in keys
            if not torch.equal(ref[k], g[k])
        )
        med = rel[len(rel) // 2][0] if rel else 0.0
        worst = f"max {rel[-1][0]:.3g} ({rel[-1][1]})" if rel else "all bitwise"
        print(f"{cell} vs main, {group}: {same}/{len(keys)} bitwise; median rel {med:.3g}; {worst}")
