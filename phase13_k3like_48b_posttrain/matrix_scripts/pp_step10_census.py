"""Sign census of two step-1 gradient dumps (grad_tensor_dump_hack.py): how many elements
change sign between two runs, where they sit in magnitude, and what that does to Adam's
first update, which is lr * sign(g) per element at step 1.

usage: pp_step10_census.py <label> <prefix_a> <prefix_b>   (prefix = path without .rankN.pt)
"""
import glob, hashlib, sys, torch

def load(prefix):
    d = {}
    for f in sorted(glob.glob(prefix + ".rank*.pt")):
        d.update(torch.load(f, map_location="cpu"))
    return d

def group(name):
    for key, g in (("tok_embeddings", "embedding"), ("output", "head"), ("experts", "experts"),
                   ("router", "router"), ("shared", "shared experts"), ("inner_kda", "kda"),
                   ("kda", "kda"), ("attention", "attention"), ("norm", "norms")):
        if key in name:
            return g
    return "other"

label, pa, pb = sys.argv[1:4]
a, b = load(pa), load(pb)
common = [k for k in a if k in b]
tot = {}; same = 0; flips = 0; n_all = 0; rel = []
lowmag = 0  # flipped elements whose |g| is below 1e-2 of the tensor's rms
for k in common:
    ga, gb = a[k].float(), b[k].float()
    if hashlib.sha1(a[k].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == \
       hashlib.sha1(b[k].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest():
        same += 1
    fl = (torch.sign(ga) != torch.sign(gb))
    nf = int(fl.sum()); n = ga.numel()
    rms = ga.pow(2).mean().sqrt().item() or 1.0
    lowmag += int((fl & (ga.abs() < 1e-2 * rms)).sum())
    g = group(k); t = tot.setdefault(g, [0, 0, 0.0, 0.0]); t[0] += n; t[1] += nf
    t[2] += float((ga - gb).pow(2).sum()); t[3] += float(ga.pow(2).sum())
    flips += nf; n_all += n
    na, nb = ga.norm().item(), gb.norm().item()
    if max(na, nb) > 0:
        rel.append(abs(na - nb) / max(na, nb))
rel.sort(); q = lambda p: rel[min(len(rel) - 1, int(p * len(rel)))]
f = flips / n_all
print(f"### {label}")
print(f"parameters {len(common)}; sha1-identical {same}; per-parameter norm rel diff median/p90/max = {q(0.5):.1e} / {q(0.9):.1e} / {rel[-1]:.1e}")
print(f"elements {n_all:,}; sign flips {flips:,} = {100*f:.3f}% (of which {100*lowmag/max(flips,1):.1f}% sit below 1e-2 of their tensor's rms)")
print(f"Adam step-1 update (lr*sign(g)): relative L2 difference 2*sqrt(f) = {100*2*f**0.5:.2f}% of the update norm")
print("| group | elements | sign flips | gradient rel L2 diff |")
print("|---|---|---|---|")
for g, (n, nf, d2, s2) in sorted(tot.items()):
    print(f"| {g} | {n:,} | {100*nf/n:.3f}% | {100*(d2/s2)**0.5 if s2 else 0:.3f}% |")
