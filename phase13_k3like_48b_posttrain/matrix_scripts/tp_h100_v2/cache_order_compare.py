"""Compare step-1 gradient dumps (cache_order_gradprobe.sh): bitwise count, then per-layer max relative diff."""
import glob, re, sys
import torch
def load(out, nm):
    d = {}
    for f in sorted(glob.glob(f"{out}/grads_{nm}.rank*.pt")):
        d.update(torch.load(f))
    return d
out, ref_nm, names = sys.argv[1], sys.argv[2], sys.argv[3:]
ref = load(out, ref_nm)
def key(n):
    m = re.search(r"layers\.(\d+)\.", n)
    return (int(m.group(1)) if m else (-1 if "tok_emb" in n or "vision" in n else 10**6), n)
for nm in names:
    g = load(out, nm)
    common = sorted(set(ref) & set(g), key=key)
    eq = [n for n in common if torch.equal(ref[n], g[n])]
    print(f"\n== {nm} vs {ref_nm}: {len(eq)}/{len(common)} bitwise (missing {len(set(ref)-set(g))}, extra {len(set(g)-set(ref))})")
    per = {}
    for n in common:
        if n in eq: continue
        a, b = ref[n].double(), g[n].double()
        rel = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
        per.setdefault(key(n)[0], []).append((rel, n))
    for layer in sorted(per):
        rel, n = max(per[layer])
        print(f"  layer {layer:>7}: {len(per[layer]):3d} differ, max rel {rel:.2e} ({n})")
