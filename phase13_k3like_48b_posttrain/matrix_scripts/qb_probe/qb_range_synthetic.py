"""Fixed [-1,1] histogram range (the PR) vs the report's adaptive range [b_min-1, b_max+1] (App. D), iterated to a fixed
point on the module's own skewed synthetic router (n=16, k=2, m=4096); plus the exact quantile as the reference."""
import sys, torch
sys.path.insert(0, "/tmp/wt_qbrebase")
from torchtitan.components.quantile_balance import (topk_with_cutoff, margin_histogram, quantile_balance_bias_histogram,
                                                     quantile_balance_bias, expert_loads)
torch.manual_seed(0)
n, k, m = 16, 2, 4096
logits = torch.randn(m, n) * 0.5 + torch.linspace(1.5, -1.5, n)  # skewed: early experts favoured
scores = torch.sigmoid(logits)
def cv(bias):
    l = expert_loads(scores, bias, k).float(); return (l.std(unbiased=False) / l.mean()).item()
def run(mode, bins, iters=60):
    bias = torch.zeros(n); hist = []
    for t in range(iters):
        _, cutoff = topk_with_cutoff(scores, bias, k)
        if mode == "exact":
            bias = quantile_balance_bias(scores, cutoff, k)
        else:
            if mode == "fixed":
                lo, hi = -1.0, 1.0
            else:  # adaptive, the report's range for the required bias r = alpha - s mirrored onto margins s - alpha
                bmin, bmax = bias.min().item(), bias.max().item(); lo, hi = -(1.0 + bmax), 1.0 - bmin
            counts = margin_histogram(scores, cutoff, num_bins=bins, lo=lo, hi=hi)
            bias = quantile_balance_bias_histogram(counts, k, lo=lo, hi=hi)
        hist.append(cv(bias))
    return hist
print(f"start cv = {cv(torch.zeros(n)):.3f}")
for mode, bins in (("exact", 0), ("fixed", 512), ("fixed", 1000), ("adaptive", 512), ("adaptive", 1000)):
    h = run(mode, bins)
    print(f"{mode:8s} bins={bins:4d}: cv after 1/5/20/60 updates = {h[0]:.3f} / {h[4]:.3f} / {h[19]:.3f} / {h[59]:.3f}   bias range at end: see below")
# the bias magnitude the adaptive range must cover
bias = torch.zeros(n)
for t in range(60):
    _, cutoff = topk_with_cutoff(scores, bias, k); bias = quantile_balance_bias(scores, cutoff, k)
print(f"exact-quantile fixed point: bias in [{bias.min():.3f}, {bias.max():.3f}]  (margins then span [{-(1+bias.max()):.2f}, {1-bias.min():.2f}] vs the fixed [-1, 1])")
