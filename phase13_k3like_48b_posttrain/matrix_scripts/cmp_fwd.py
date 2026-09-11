import glob, os, re, sys, torch, numpy as np

def ulps(a, b):
    """Distance in float32 units in the last place."""
    ai = a.detach().float().cpu().numpy().view(np.int32).astype(np.int64)
    bi = b.detach().float().cpu().numpy().view(np.int32).astype(np.int64)
    # map to a monotone ordering so negatives compare correctly
    ai = np.where(ai < 0, np.int64(-2**31) - ai, ai)
    bi = np.where(bi < 0, np.int64(-2**31) - bi, bi)
    return np.abs(ai - bi)

D = sys.argv[1] if len(sys.argv) > 1 else "/workspace/fwd_dump"
files = glob.glob(os.path.join(D, "f.ws*.pt"))
idx = {}
for f in files:
    m = re.search(r"\.ws(\d+)\.rank(\d+)\.l(\d+)\.c(\d+)\.pt$", f)
    ws, rk, l, c = (int(x) for x in m.groups())
    idx.setdefault(ws, {})[(l, c)] = f
if 1 not in idx or 2 not in idx:
    print("missing runs:", sorted(idx)); sys.exit(1)
pp_layers = sorted({l for (l, c) in idx[2]})
print(f"dp1 entries={len(idx[1])} pp2 entries={len(idx[2])}")
print(f"pp2 layers present: {pp_layers[0]}..{pp_layers[-1]}")
# the stage boundary: which layers came from rank 1
r1 = sorted({int(re.search(r'\.l(\d+)\.', f).group(1)) for f in idx[2].values() if '.rank1.' in f})
print(f"rank1 (stage 1) layers: {r1[0]}..{r1[-1]}" if r1 else "rank1: none")
print()
hdr = f"{'layer':>5} {'call':>4} | {'values differ':>28} | {'probs differ':>22} | {'out differ':>26}"
print(hdr); print("-" * len(hdr))
first_bad = None
for (l, c) in sorted(idx[2]):
    if (l, c) not in idx[1]:
        continue
    a = torch.load(idx[1][(l, c)], weights_only=False)
    b = torch.load(idx[2][(l, c)], weights_only=False)
    row = []
    for k in ("values_float", "probs", "out"):
        u = ulps(a[k], b[k]); n = int((u > 0).sum()); tot = u.size
        row.append(f"{n}/{tot} max {int(u.max())}u" if n else "identical")
    if first_bad is None and any("identical" != r for r in row):
        first_bad = (l, c, a, b)
    print(f"{l:>5} {c:>4} | {row[0]:>28} | {row[1]:>22} | {row[2]:>26}")
print()
if first_bad:
    l, c, a, b = first_bad
    print(f"FIRST DIFFERENCE at layer {l}, call {c}")
    for k in ("values_shape", "values_stride", "values_contig", "values_dtype",
              "block_shape", "block_stride", "block_contig", "has_partial"):
        print(f"  {k:>15}: dp1={a[k]}  pp2={b[k]}")
else:
    print("no difference anywhere in the dumped forward")
