"""Compare two bitwise_dump.py output folders rank by rank: bitwise equality and the max abs gap per tensor."""

import glob
import os
import sys

import torch

a_dir, b_dir = sys.argv[1], sys.argv[2]
ok = True
for a_path in sorted(glob.glob(os.path.join(a_dir, "rank*.pt"))):
    name = os.path.basename(a_path)
    a, b = torch.load(a_path), torch.load(os.path.join(b_dir, name))
    for key in ("out", "grad_x", "grad_w13", "grad_w2", "grad_weights"):
        for i, (x, y) in enumerate(zip(a[key], b[key], strict=True)):
            same = torch.equal(x, y)
            ok &= same
            if not same:
                gap = (x.float() - y.float()).abs().max().item()
                rel = gap / max(y.float().abs().max().item(), 1e-30)
                print(f"{name} {key}[{i}] differs: max abs gap {gap:.3e} ({rel:.2e} of max |new|)")
    print(f"{name} peak alloc MiB: {a['peak_alloc'] / 2**20:.1f} -> {b['peak_alloc'] / 2**20:.1f}")
print("BITWISE EQUAL" if ok else "NOT EQUAL")
