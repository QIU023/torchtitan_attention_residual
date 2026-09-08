"""Diff the routing decisions of the two hooks on the same data: fraction of tokens whose Top-k expert SET differs, per layer."""
import glob, os, sys, torch
OUT = sys.argv[1]
def load(name, step):
    files = sorted(glob.glob(f"{OUT}/routes_{name}/routes_step{step}_rank*.pt"))
    per_rank = [torch.load(f) for f in files]
    return per_rank
for pair in (("dp1_control", "dp1_qb"), ("dp2ep2_control", "dp2ep2_qb")):
    for step in (2, 10):
        a, b = load(pair[0], step), load(pair[1], step)
        if not a or not b or len(a) != len(b):
            print(f"{pair} step {step}: missing dumps ({len(a)} vs {len(b)} ranks)"); continue
        fr = []
        for ra, rb in zip(a, b):
            for key in sorted(ra):
                ida = torch.cat([t.reshape(-1, t.shape[-1]) for t in ra[key]]); idb = torch.cat([t.reshape(-1, t.shape[-1]) for t in rb[key]])
                if ida.shape != idb.shape: fr.append(float("nan")); continue
                sa, sb = ida.sort(dim=-1).values, idb.sort(dim=-1).values
                fr.append((sa != sb).any(dim=-1).float().mean().item())
        vals = [f for f in fr if f == f]
        print(f"{pair[0]} vs {pair[1]} step {step}: tokens whose routed expert set differs, mean over layers {sum(vals)/len(vals):.3%} (min {min(vals):.3%}, max {max(vals):.3%}, {len(vals)} layer-ranks)")
