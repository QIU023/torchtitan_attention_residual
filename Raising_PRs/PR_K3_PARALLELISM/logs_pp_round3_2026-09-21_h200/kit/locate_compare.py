"""Compare the locate dumps: python locate_compare.py <out> <ref> <cells...>. Per cell: step-1 gradients (before the clip)
and step-1 parameters (after the update) bitwise counts against the reference with the per-layer largest relative
difference, then the micro batch losses of step 1 and the total grad norm per step, hex against hex."""
import glob
import re
import sys

import torch

out, ref_nm, names = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(kind, nm):
    d = {}
    for f in sorted(glob.glob(f"{out}/{kind}_{nm}.rank*.pt")):
        d.update(torch.load(f))
    return d


def key(n):
    m = re.search(r"layers\.(\d+)\.", n)
    return (int(m.group(1)) if m else (-1 if "tok_emb" in n or "vision" in n else 10**6), n)


def compare(kind, nm):
    ref, g = load(kind, ref_nm), load(kind, nm)
    common = sorted(set(ref) & set(g), key=key)
    eq = [n for n in common if torch.equal(ref[n], g[n])]
    print(f"== {kind} {nm} vs {ref_nm}: {len(eq)}/{len(common)} bitwise (missing {len(set(ref) - set(g))}, extra {len(set(g) - set(ref))})")
    per = {}
    for n in common:
        if n in eq:
            continue
        a, b = ref[n].double(), g[n].double()
        rel = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
        per.setdefault(key(n)[0], []).append((rel, n))
    for layer in sorted(per):
        rel, n = max(per[layer])
        print(f"  layer {layer:>7}: {len(per[layer]):3d} differ, max rel {rel:.2e} ({n})")


def scalars(nm, tag):
    rows = {}
    for line in open(f"{out}/{nm}.log", errors="ignore"):
        m = re.match(rf".*{tag} r(\d+) (\d+) (\S+) (\S+)", line)
        if m:
            rows.setdefault(int(m.group(2)), {})[int(m.group(1))] = (m.group(3), m.group(4))
    return rows


for nm in names:
    compare("grads", nm)
    compare("params", nm)
    for tag in ("LOSSREPR", "GNREPR"):
        a, b = scalars(ref_nm, tag), scalars(nm, tag)
        same = [k for k in sorted(a) if k in b and all(v[1] == a[k][0][1] for v in b[k].values())]
        diff = [k for k in sorted(a) if k in b and k not in same]
        print(f"  {tag} {nm}: {len(same)}/{len(a)} equal to the reference in hex; first different: {diff[:5]}")
        for k in diff[:3]:
            print(f"    {k}: ref {a[k][0][0]} {a[k][0][1]} | cell " + " | ".join(f"r{r} {v[0]} {v[1]}" for r, v in sorted(b[k].items())))
