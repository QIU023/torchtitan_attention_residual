"""Step-1 and step-2 gradients, main against the PR with dynamic CP on and no split, and each rank's local image tokens."""
import re
import sys
from pathlib import Path

import torch

D = Path(sys.argv[1])
for cell in ("tokens",):
    lines = re.findall(r"UPD2_GATHER rank (\d) call (\d+) local_image_tokens (\d+) rows (\S+)", (D / f"g2_{cell}" / "run.log").read_text(errors="replace"))
    per_call = {}
    for rank, call, n, rows in lines:
        per_call.setdefault(int(call), {})[int(rank)] = f"{n} ({rows})"
    print(f"{cell}: local image tokens per gather call (rank 0 | rank 1): " + "; ".join(f"call {c}: {v.get(0)} | {v.get(1)}" for c, v in sorted(per_call.items())))
for step in (1, 2):
    a = torch.load(D / "g2_main" / "dump" / f"grad{step}.pt")
    b = torch.load(D / "g2_prwhole" / "dump" / f"grad{step}.pt")
    for group in ("tower", "rest"):
        keys = [k for k in a if ("vision_encoder" in k) == (group == "tower")]
        same = sum(torch.equal(a[k], b[k]) for k in keys)
        rels = sorted(
            ((b[k].float() - a[k].float()).abs().max().item() / max(a[k].float().abs().max().item(), 1e-30), k)
            for k in keys
            if not torch.equal(a[k], b[k])
        )
        dtypes = sorted({str(a[k].dtype) for k in keys})
        extra = f"; median rel {rels[len(rels) // 2][0]:.3g}, max {rels[-1][0]:.3g} ({rels[-1][1]})" if rels else ""
        print(f"step {step} {group}: {same}/{len(keys)} bitwise, dtypes {dtypes}{extra}")
