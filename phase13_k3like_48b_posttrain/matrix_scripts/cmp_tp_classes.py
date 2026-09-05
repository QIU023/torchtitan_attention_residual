import sys, torch, collections
D, tag = sys.argv[1], sys.argv[2]; tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
a = torch.load(f"{D}/{tag}_dp1.rank0.pt")
def rel(x, y):
    x = x.float(); y = y.float(); return ((x - y).norm() / max(x.norm().item(), 1e-12)).item()
for label in ("tp2sp", "tp2nosp"):
    r0 = torch.load(f"{D}/{tag}_{label}.rank0.pt"); r1 = torch.load(f"{D}/{tag}_{label}.rank1.pt")
    cls = collections.defaultdict(list)
    for k, g in a.items():
        if k not in r0 or r0[k].shape != g.shape: cls["shape"].append((k, 0)); continue
        e0, e1 = rel(g, r0[k]), rel(g, r1[k]); es = rel(g, r0[k].float() + r1[k].float()); em = rel(g, (r0[k].float() + r1[k].float()) / 2)
        if max(e0, e1) < tol: cls["each==dp1"].append((k, max(e0, e1)))
        elif es < tol: cls["sum==dp1"].append((k, es))
        elif em < tol: cls["mean==dp1"].append((k, em))
        else: cls["wrong"].append((k, min(e0, e1, es, em)))
    print(f"== {label}: " + ", ".join(f"{c} {len(v)}" for c, v in cls.items()))
    for c in ("sum==dp1", "mean==dp1", "wrong", "shape"):
        names = sorted(set(k.replace('_checkpoint_wrapped_module.', '').split('.', 2)[-1] if k.startswith('layers.') else k for k, _ in cls[c]))
        if names: print(f"   {c} ({len(cls[c])}): {names[:24]}{' ...' if len(names) > 24 else ''}")
    if cls["wrong"]:
        w = sorted(cls["wrong"], key=lambda x: -x[1])[:6]
        print("   worst wrong (best-of rel):", [(k.replace('_checkpoint_wrapped_module.', ''), f"{e:.2e}") for k, e in w])
