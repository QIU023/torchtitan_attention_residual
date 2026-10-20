"""Owner tower (DEP off) running gradients against the replicas' per-micro-batch gradients (DEP on), new tree:
for each parameter, the first accumulate call whose value differs from the fp32 running sum of the replica gradients."""
import sys
from collections import Counter
from pathlib import Path

import torch

G = Path(sys.argv[1])
own = torch.load(G / "own_new" / "dump" / "ownergrad_rank0.pt")
rep = {}
for r in range(4):
    rep.update(torch.load(G / "mb_new" / "dump" / f"mbgrad_rank{r}.pt"))
rep_mbs = sorted({m for m, _ in rep})
lens = Counter(len(v) for v in own.values())
print(f"owner params {len(own)}, calls per param {dict(lens)}, dtypes {sorted({str(t.dtype) for v in own.values() for t in v})}; replica micro-batches {rep_mbs}")
first_bad = Counter()
for fqn, values in sorted(own.items()):
    name = fqn.split("vision_encoder.", 1)[1]
    keys = [k for k in {n for _, n in rep} if k.endswith(name) or name.endswith(k)]
    assert len(keys) == 1, (fqn, keys)
    key = keys[0]
    mbs = list(range(16)) if len(values) == 16 else rep_mbs
    running = None
    bad = None
    for k, (mb, v) in enumerate(zip(mbs, values)):
        g = rep.get((mb, key))
        if g is not None:
            running = g.float() if running is None else running + g.float()
        ref = running if running is not None else torch.zeros_like(v.float())
        if not torch.equal(v.float(), ref):
            rel = (v.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
            bad = (k, mb, rel, str(v.dtype))
            break
    first_bad[bad[:2] if bad else None] += 1
    if bad:
        print(f"   {fqn}: first differs at call {bad[0]} (micro-batch {bad[1]}), max rel {bad[2]:.3g}, value dtype {bad[3]}")
print("first differing (call, micro-batch) over parameters:", dict(first_bad))
