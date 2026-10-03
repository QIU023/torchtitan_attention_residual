"""Max abs error against the fp32 dense reference, old tree vs new tree, per tensor kind (max over ranks and indices)."""

import glob
import os
import sys

import torch

old_dir, new_dir = sys.argv[1], sys.argv[2]
for key in ("out", "grad_x", "grad_w13", "grad_w2", "grad_weights"):
    errs = {}
    for tag, d in (("old", old_dir), ("new", new_dir)):
        e = 0.0
        for path in sorted(glob.glob(os.path.join(d, "rank*.pt"))):
            dump = torch.load(path)
            for x, r in zip(dump[key], dump["ref"][key], strict=True):
                e = max(e, (x.float() - r.float()).abs().max().item())
        errs[tag] = e
    print(f"{key:13s} max |x - fp32 ref|: old {errs['old']:.3e}  new {errs['new']:.3e}")
