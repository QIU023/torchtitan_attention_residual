import sys, torch, collections, re
D, tag = sys.argv[1], sys.argv[2]
a = torch.load(f"{D}/{tag}_dp1.rank0.pt")
def ratio(x, y): return (y.float().norm() / max(x.float().norm().item(), 1e-12)).item()
def rel(x, y): x = x.float(); y = y.float(); return ((x - y).norm() / max(x.norm().item(), 1e-12)).item()
def short(k): k = k.replace("_checkpoint_wrapped_module.", ""); return re.sub(r"^layers\.(\d+)\.", lambda m: f"L{int(m.group(1)):02d}.", k)
for label in ("tp2sp", "tp2nosp"):
    r0 = torch.load(f"{D}/{tag}_{label}.rank0.pt"); r1 = torch.load(f"{D}/{tag}_{label}.rank1.pt")
    cls = collections.defaultdict(list)
    for k, g in a.items():
        if k not in r0: continue
        ident = torch.equal(r0[k], r1[k]); q1 = ratio(g, r0[k]); qs = ratio(g, r0[k].float() + r1[k].float()); e1 = rel(g, r0[k]); es = rel(g, r0[k].float() + r1[k].float())
        if ident:
            c = "ident~1x" if 0.8 < q1 < 1.25 else ("ident~2x" if 1.6 < q1 < 2.5 else ("ident~0.5x" if 0.4 < q1 < 0.6 else "ident-other"))
        else:
            c = "diff,sum~1x" if 0.8 < qs < 1.25 and es < 0.5 else ("diff,each~1x" if 0.8 < q1 < 1.25 and e1 < 0.5 else "diff-other")
        cls[c].append((short(k), q1, qs))
    print(f"== {label}: " + ", ".join(f"{c}: {len(v)}" for c, v in sorted(cls.items())))
    for c in sorted(cls):
        if c == "ident~1x": continue
        fams = collections.Counter(re.sub(r"^L\d+\.", "", n) for n, _, _ in cls[c])
        print(f"   {c}: " + "; ".join(f"{f} x{n}" for f, n in sorted(fams.items())))
        for n, q1, qs in cls[c][:4]: print(f"      e.g. {n}: r0/dp1 {q1:.3f}, (r0+r1)/dp1 {qs:.3f}")
