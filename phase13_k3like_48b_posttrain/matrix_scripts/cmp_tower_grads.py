import sys, statistics, torch
D = sys.argv[1]
a = torch.load(f"{D}/g_dp1.rank0.pt")
def rel(x, y):
    x = x.float(); y = y.float(); return ((x - y).norm() / max(x.norm().item(), 1e-12)).item()
def ratio(x, y):
    return (y.float().norm() / max(x.float().norm().item(), 1e-12)).item()
for label in ("tp2sp", "tp2nosp"):
    r0 = torch.load(f"{D}/g_{label}.rank0.pt"); r1 = torch.load(f"{D}/g_{label}.rank1.pt")
    same = [k for k in a if k in r0 and a[k].shape == r0[k].shape]
    tower = [k for k in same if k.startswith("vision_encoder")]
    other = [k for k in same if not k.startswith("vision_encoder")]
    ident_t = sum(1 for k in tower if torch.equal(r0[k], r1[k]))
    ident_o = sum(1 for k in other if torch.equal(r0[k], r1[k]))
    dt = sorted(((rel(a[k], r0[k]), ratio(a[k], r0[k]), k) for k in tower), reverse=True)
    do = sorted(((rel(a[k], r0[k]), ratio(a[k], r0[k]), k) for k in other), reverse=True)
    print(f"== {label}: same-shape params {len(same)} (tower {len(tower)}, other {len(other)}); rank0==rank1: tower {ident_t}/{len(tower)}, other {ident_o}/{len(other)}")
    for name, d in (("tower", dt), ("other same-shape", do)):
        if not d: continue
        rels = [x[0] for x in d]; rats = [x[1] for x in d]
        print(f"  {name}: rel diff vs dp1 median {statistics.median(rels):.3e} max {d[0][0]:.3e} ({d[0][2]}); norm ratio tp2/dp1 median {statistics.median(rats):.4f} min {min(rats):.4f} max {max(rats):.4f}; >5% off: {sum(1 for r in rels if r > 0.05)}")
