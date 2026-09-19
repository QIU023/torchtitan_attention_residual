import json, sys, re, statistics
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print(f"loss: {sys.argv[1].split('/')[-1]}={a['loss']:.6f}  {sys.argv[2].split('/')[-1]}={b['loss']:.6f}")
ga, gb = a["grads"], b["grads"]
def grp(n):
    if "routed_experts.inner_experts" in n: return "experts(w1/w2/w3)"
    if ".router." in n: return "router"
    if "routed_down" in n or "routed_up" in n or "routed_norm" in n: return "latent proj/norm"
    if "shared_experts" in n: return "shared_experts"
    if "vision" in n: return "vision"
    if "attention" in n or "kda" in n or "mla" in n or "attn" in n: return "attention"
    if "embed" in n or "output" in n or "res_" in n: return "embed/output/res"
    return "other"
rows = []
for n in ga:
    x, y = ga[n], gb.get(n, float("nan"))
    rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
    rows.append((grp(n), n, x, y, rel))
ta = sum(v*v for v in ga.values()) ** 0.5; tb = sum(v*v for v in gb.values()) ** 0.5
print(f"total grad norm: {ta:.5f} vs {tb:.5f}")
print(f"{'group':20s} {'n':>4s} {'max rel':>9s} {'median rel':>11s} {'norm A':>10s} {'norm B':>10s}")
for g in sorted(set(r[0] for r in rows)):
    rs = [r for r in rows if r[0] == g]
    na = sum(r[2]**2 for r in rs) ** 0.5; nb = sum(r[3]**2 for r in rs) ** 0.5
    print(f"{g:20s} {len(rs):4d} {max(r[4] for r in rs):9.2e} {statistics.median(r[4] for r in rs):11.2e} {na:10.4f} {nb:10.4f}")
print("--- top 12 per-parameter relative differences ---")
for r in sorted(rows, key=lambda r: -r[4])[:12]:
    print(f"{r[4]:9.2e}  {r[2]:11.5f} {r[3]:11.5f}  {r[1]}")
