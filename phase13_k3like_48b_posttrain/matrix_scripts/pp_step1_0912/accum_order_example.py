"""One committed block b, read by 3 layers on each of 4 stages (pp2 x vp2:
rank0 = stages 0,2; rank1 = stages 1,3).  Three ways to get d loss / d b:
  dp1    : one autograd graph, every read of the stack accumulates in engine order
  naive  : cache off, every hop carries the stack; each stage's input is a fresh
           leaf (as assemble_stack does) and its .grad goes back over the wire
  cached : cache on, stages 2 and 3 read b from their rank's store; their grads
           are deposited (real PPRankLocalCache) and collected by the stage that
           brought b onto the rank (stage 1 on rank1, stage 0 on rank0)
"""
import torch
from torchtitan.models.kimi_k3.pipeline_stage import PPRankLocalCache

def run(dtype, seed=0, T=64, D=16, reads=3):
    g = torch.Generator().manual_seed(seed)
    b0 = torch.randn(T, D, generator=g).to(dtype)
    # per-read weights with spread magnitudes so rounding is visible
    W = [[(torch.randn(T, D, generator=g) * 10 ** torch.randint(-2, 2, (1,), generator=g)).to(dtype)
          for _ in range(reads)] for _ in range(4)]
    def stage_loss(S, k):          # S: stack [T,1,D]; the layers of stage k read it
        return sum(((S[:, 0] * w).float().sum()) for w in W[k])

    # dp1
    x = b0.clone().requires_grad_(True)
    S = x.unsqueeze(1)
    sum(stage_loss(S, k) for k in range(4)).backward()
    dp1 = x.grad.clone()

    # naive: stage k gets a leaf, sends a stack onward, receives the stack grad
    x = b0.clone().requires_grad_(True)
    leaves, outs, losses = [], [], []
    inp = x.unsqueeze(1)
    for k in range(4):
        leaf = inp if k == 0 else inp.detach().requires_grad_(True)
        losses.append(stage_loss(leaf, k)); outs.append(torch.stack([leaf[:, 0]], 1)); leaves.append(leaf)
        inp = outs[-1]
    grad_next = None
    for k in reversed(range(4)):
        ts, gs = [losses[k]], [None]
        if grad_next is not None: ts.append(outs[k]); gs.append(grad_next)
        torch.autograd.backward(ts, gs)
        if k: grad_next = leaves[k].grad
    naive = x.grad.clone()

    # cached: b travels 0 -> 1 once; stages 2, 3 read it from their rank's store
    x = b0.clone().requires_grad_(True)
    store = {0: PPRankLocalCache(), 1: PPRankLocalCache()}
    S0 = x.unsqueeze(1); l0 = stage_loss(S0, 0); out0 = torch.stack([S0[:, 0]], 1)
    store[0].put(0, 0, S0[:, 0].detach())
    S1 = out0.detach().requires_grad_(True); l1 = stage_loss(S1, 1)
    store[1].put(0, 0, S1[:, 0].detach())
    S2 = torch.stack([store[0].blocks(0)[0]], 1).requires_grad_(True); l2 = stage_loss(S2, 2)
    S3 = torch.stack([store[1].blocks(0)[0]], 1).requires_grad_(True); l3 = stage_loss(S3, 3)
    l3.backward(); store[1].deposit(0, 0, S3.grad[:, 0])       # stage 3 deposits on rank1
    l2.backward(); store[0].deposit(0, 0, S2.grad[:, 0])       # stage 2 deposits on rank0
    l1.backward(); gw = S1.grad.clone()
    gw[:, 0].add_(store[1].collect(0, 0)[0])                   # stage 1 collects rank1's deposit
    gw = gw.clone(); gw[:, 0].add_(store[0].collect(0, 0)[0])  # stage 0 collects rank0's deposit
    torch.autograd.backward([l0, out0], [None, gw])
    cached = x.grad.clone()
    return b0, W, dp1, naive, cached

def ulps(a, b):
    ia = a.view(torch.int16).int(); ib = b.view(torch.int16).int()
    ia = torch.where(ia < 0, -32768 - ia, ia); ib = torch.where(ib < 0, -32768 - ib, ib)
    return (ia - ib).abs()

_, W, dp1, naive, cached = run(torch.bfloat16)
print("bf16  naive == dp1 bitwise:", torch.equal(naive, dp1))
print("bf16  cached == dp1 bitwise:", torch.equal(cached, dp1),
      "| differing elements:", int((cached != dp1).sum()), "of", dp1.numel(),
      "| max ulps:", int(ulps(cached, dp1).max()))
_, _, d64, n64, c64 = run(torch.float64)
print("fp64  max|naive-dp1| = %.1e   max|cached-dp1| = %.1e   (|grad| ~ %.1e)"
      % ((n64 - d64).abs().max(), (c64 - d64).abs().max(), d64.abs().mean()))
# one element, spelled out
i = int((cached != dp1).flatten().nonzero()[0])
r = [sum(w.flatten()[i] for w in W[k][::-1]) for k in range(4)]  # each stage's reads, summed top-down in bf16
bf = lambda v: torch.tensor(float(v)).to(torch.bfloat16)
print(f"element {i}: per-stage partial sums r0..r3 =", [f"{float(v):.6g}" for v in r])
print(f"  dp1/naive fold ((r3 + r2) + r1) + r0 -> {float(dp1.flatten()[i]):.6g}")
print(f"  cached    ((r1 + r3) + r2) + r0     -> {float(cached.flatten()[i]):.6g}")
