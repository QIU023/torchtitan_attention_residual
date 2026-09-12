# The gradient of one block of the AttnRes stack under pp2 x vp2 (rank 0 = stages 0, 2; rank 1 =
# stages 1, 3). a[k] are the gradients of stage k's reads of the block, in backward order.
# Autograd adds each read onto whatever gradient already arrived: a running fold, top-down.
import torch

torch.manual_seed(0)
a = [[torch.randn(4096).mul(10.0 ** torch.randint(-2, 2, (4096,))).bfloat16() for _ in range(3)]
     for _ in range(4)]

def fold(start, reads):          # ((start + r0) + r1) + r2, the order autograd accumulates in
    g = start
    for r in reads:
        g = r.clone() if g is None else g + r
    return g

no_pp = fold(None, a[3] + a[2] + a[1] + a[0])            # one graph: every read, top-down

g = fold(None, a[3])                                      # cache off: each hop hands the running
for k in (2, 1, 0):                                       # sum back and the next stage keeps folding
    g = fold(g, a[k])
cache_off = g

d3, d2 = fold(None, a[3]), fold(None, a[2])               # cache on: stages 2, 3 read the block from
g1 = fold(None, a[1]) + d3                                # their rank's store and fold from zero; stage 1
cache_on = fold(g1 + d2, a[0])                            # adds rank 1's deposit, stage 0 rank 0's, then folds

print("cache off == no PP:", torch.equal(cache_off, no_pp))
print("cache on  == no PP:", torch.equal(cache_on, no_pp), "| elements that differ:",
      int((cache_on != no_pp).sum()), "of", no_pp.numel())
a64 = [[r.double() for r in s] for s in a]
exact = sum(r for s in a64 for r in s)
print("max |fold - exact sum| in float64 terms: no PP %.2e, cache on %.2e"
      % ((no_pp.double() - exact).abs().max(), (cache_on.double() - exact).abs().max()))
