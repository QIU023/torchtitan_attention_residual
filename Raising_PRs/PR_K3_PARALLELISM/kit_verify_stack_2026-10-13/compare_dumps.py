"""Compare two state dumps per rank: keys present in only one, and tensors that are not bitwise equal."""
import pathlib
import sys

import torch

a_dir, b_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for tag in ("pre6", "post6"):
    for a_file in sorted(a_dir.glob(f"{tag}_rank*.pt")):
        b_file = b_dir / a_file.name
        a, b = torch.load(a_file), torch.load(b_file)
        only = sorted(set(a) ^ set(b))
        diffs = []
        for key in sorted(set(a) & set(b)):
            x, y = a[key], b[key]
            if isinstance(x, torch.Tensor):
                if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
                    m = (x.double() - y.double()).abs().max().item() if x.shape == y.shape else float("nan")
                    diffs.append((key, tuple(x.shape), str(x.dtype), m))
            elif x != y:
                diffs.append((key, "value", repr(x)[:40], repr(y)[:40]))
        print(f"{a_file.name}: {len(a)} entries, only in one: {len(only)}, differing: {len(diffs)}")
        for d in only[:5]:
            print("   only:", d)
        for d in diffs[:12]:
            print("   diff:", d)
