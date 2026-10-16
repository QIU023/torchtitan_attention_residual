"""Compare the probe's dumps: CP-rank agreement of tower grads (step 1) and params (after step 1), and sum-over-ranks
of the dynamic-CP-on grads against the dynamic-CP-off grad."""
import sys

import torch

root = sys.argv[1]
load = lambda run, tag, r: torch.load(f"{root}/{run}/dump/{tag}_rank{r}.pt")
for run in ("off", "on"):
    pre = [load(run, "pre1", r) for r in (0, 1)]
    post = [load(run, "post1", r) for r in (0, 1)]
    grads = [k for k in pre[0] if k.startswith("grad.")]
    params = [k for k in post[0] if k.startswith("param.")]
    gdiff = max((pre[0][k] - pre[1][k]).abs().max().item() for k in grads)
    gsame = sum(torch.equal(pre[0][k], pre[1][k]) for k in grads)
    pdiff = max((post[0][k] - post[1][k]).abs().max().item() for k in params)
    psame = sum(torch.equal(post[0][k], post[1][k]) for k in params)
    print(f"{run}: tower grads equal across CP ranks {gsame}/{len(grads)} (max diff {gdiff:.3e}); "
          f"params after step 1 equal {psame}/{len(params)} (max diff {pdiff:.3e})")
off = [load("off", "pre1", r) for r in (0, 1)]
on = [load("on", "pre1", r) for r in (0, 1)]
ratios, coss = [], []
for k in off[0]:
    if not k.startswith("grad."):
        continue
    ref = off[0][k].flatten()
    total = (on[0][k] + on[1][k]).flatten()
    if ref.norm() == 0:
        continue
    ratios.append((total.norm() / ref.norm()).item())
    coss.append(torch.nn.functional.cosine_similarity(total, ref, dim=0).item())
ratios.sort(); coss.sort()
n = len(ratios)
print(f"sum over CP ranks of the on-grads vs the off-grad, {n} tensors: norm ratio min {ratios[0]:.3f} median {ratios[n//2]:.3f} max {ratios[-1]:.3f}; cos min {coss[0]:.5f} median {coss[n//2]:.5f}")
half = [((on[0][k] + on[1][k]) / 2).flatten() for k in off[0] if k.startswith("grad.")]
