"""Compare two step-1 gradient dumps (GRAD_DUMP prefix A vs prefix B).
Merges every rank's file per side, then reports per layer: tensors, bitwise
tensors, fraction of differing elements, median/max ulps (in the dump dtype),
and the first differing tensor walking the backward (top layer down)."""
import glob, re, sys, torch
def load(prefix):
    d = {}
    for f in sorted(glob.glob(prefix + ".ws*.rank*.pt")):
        d.update(torch.load(f, weights_only=False))
    return d
def ords(t):
    bits = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype]
    i = t.contiguous().view(bits).long()
    lo = -(1 << (8 * t.element_size() - 1))
    return torch.where(i < 0, lo - i, i)
def layer(n):
    m = re.search(r"layers\.(\d+)\.", n); return int(m.group(1)) if m else None
A, B = load(sys.argv[1]), load(sys.argv[2])
stage_of = {}
if len(sys.argv) > 3:   # optional "L:S,L:S" map of layer -> stage for the pipeline side
    for kv in sys.argv[3].split(","):
        l, s_ = kv.split(":"); stage_of[int(l)] = int(s_)
common = sorted(set(A) & set(B))
print(f"A={len(A)} B={len(B)} common={len(common)} A-only={len(set(A)-set(B))} B-only={len(set(B)-set(A))}")
rows = {}
for k in common:
    a, b = A[k], B[k]
    L = layer(k); key = ("L%02d" % L) if L is not None else ("vision" if k.startswith("vision_encoder") else "embed" if k.startswith("tok_embeddings") else "head")
    r = rows.setdefault(key, [0, 0, 0, 0, [], 0, []])
    r[0] += 1
    if torch.equal(a, b): r[1] += 1; r[6].append(0.0); continue
    u = (ords(a) - ords(b)).abs()
    nz = u[u > 0].float()
    big = a.float().abs() > a.float().abs().median()   # elements above the median magnitude: no sign-flip noise
    r[2] += int((u > 0).sum()); r[3] += a.numel(); r[4].append(float(nz.median()))
    r[5] = max(r[5], int(u[big].max()) if big.any() else 0)
    r[6].append(float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-30)))
tot_equal = sum(r[1] for r in rows.values())
print(f"bitwise tensors: {tot_equal}/{len(common)}")
def sk(k): return (0, 0) if k == "head" else ((1, -int(k[1:])) if k.startswith("L") else (2, 0 if k == "embed" else 1))
print(f"{'group':6} {'stage':5} {'tensors':>7} {'bitwise':>7} {'diff frac':>9} {'med ulps':>8} {'max ulps>med|g|':>15} {'relL2 med':>9} {'relL2 max':>9}")
for k in sorted(rows, key=sk):
    n, eq, nd, ne, meds, mx, rl = rows[k]
    st = stage_of.get(int(k[1:]), "") if k.startswith("L") else ""
    frac = f"{nd/ne:.3f}" if ne else "0"
    med = f"{sorted(meds)[len(meds)//2]:.1f}" if meds else "-"
    rls = sorted(rl); rm = rls[len(rls)//2]
    print(f"{k:6} {str(st):5} {n:7d} {eq:7d} {frac:>9} {med:>8} {str(mx) if meds else '-':>15} {rm:9.1e} {max(rls):9.1e}")
