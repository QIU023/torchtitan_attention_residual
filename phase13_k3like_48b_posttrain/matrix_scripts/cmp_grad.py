import glob, os, re, sys, torch, numpy as np

def ulps(a, b):
    ai = a.numpy().view(np.int32).astype(np.int64)
    bi = b.numpy().view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(-2**31) - ai, ai)
    bi = np.where(bi < 0, np.int64(-2**31) - bi, bi)
    return np.abs(ai - bi)

D = sys.argv[1] if len(sys.argv) > 1 else "/workspace/grad_dump24"
files = glob.glob(os.path.join(D, "g.ws*.pt"))
dp1, pp2 = {}, {}
for f in files:
    ws = int(re.search(r"\.ws(\d+)\.", f).group(1))
    d = torch.load(f, weights_only=False)
    (dp1 if ws == 1 else pp2).update(d)
print(f"dp1 tensors={len(dp1)} pp2 tensors={len(pp2)}")
common = [k for k in dp1 if k in pp2]
print(f"common={len(common)}  dp1-only={len(dp1)-len(common)}  pp2-only={len(pp2)-len(common)}")

def layer_of(n):
    m = re.search(r"layers\.(\d+)\.", n)
    return int(m.group(1)) if m else None

rows = []
for k in common:
    a, b = dp1[k], pp2[k]
    if a.shape != b.shape:
        rows.append((k, layer_of(k), -1, a.numel(), -1)); continue
    u = ulps(a, b)
    rows.append((k, layer_of(k), int((u > 0).sum()), int(u.size), int(u.max())))

non_layer = [r for r in rows if r[1] is None]
print("\nnon-layer tensors (head/embedding/aggregation):")
for k, _, n, tot, mx in sorted(non_layer):
    print(f"  {k:<52} {'identical' if n==0 else f'{n}/{tot} max {mx}u'}")

print("\nper-layer (share of elements differing, max ulps):")
print(f"{'layer':>5} {'tensors':>7} {'differing':>10} {'elements':>12} {'share':>8} {'maxulp':>7}")
bylayer = {}
for k, l, n, tot, mx in rows:
    if l is None: continue
    a = bylayer.setdefault(l, [0, 0, 0, 0])
    a[0] += 1; a[1] += n; a[2] += tot; a[3] = max(a[3], mx)
for l in sorted(bylayer, reverse=True):
    t, n, tot, mx = bylayer[l]
    print(f"{l:>5} {t:>7} {n:>10} {tot:>12} {100*n/tot:>7.2f}% {mx:>7}")

diffs = [r for r in rows if r[2] > 0]
print(f"\ntensors differing: {len(diffs)} of {len(rows)}")
if diffs:
    # "first" in backward order = the highest layer id / non-layer output side
    def order(r):
        return (-(r[1] if r[1] is not None else 10**6),)
    first = sorted(diffs, key=order)[0]
    print(f"first non-identical in backward order: {first[0]}  {first[2]}/{first[3]} elements, max {first[4]} ulp")
    deepest = max((r[1] for r in rows if r[1] is not None), default=None)
    print(f"deepest layer id present: {deepest}")
