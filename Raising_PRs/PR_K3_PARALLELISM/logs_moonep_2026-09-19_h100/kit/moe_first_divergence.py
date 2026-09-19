"""Characterise the first diverging MoE layer: rounding order, or dropped tokens?"""
import sys
import torch

a = torch.load(sys.argv[1], weights_only=False)["capture"]
b = torch.load(sys.argv[2], weights_only=False)["capture"]
name = sys.argv[3]

xa, xb = a[name]["x"], b[name]["x"]
oa, ob = a[name]["out"], b[name]["out"]
print(f"module: {name}")
print(f"input identical      : {bool(torch.equal(xa, xb))}  shape={tuple(xa.shape)}")
print(f"output shape         : {tuple(oa.shape)}")

d = (oa - ob).abs()
print(f"output max |a|       : {float(oa.abs().max()):.6e}   rms {float(oa.pow(2).mean().sqrt()):.6e}")
print(f"output max abs diff  : {float(d.max()):.6e}   rms {float(d.pow(2).mean().sqrt()):.6e}")
print(f"elements differing   : {int((d > 0).sum())} of {d.numel()} ({100.0 * float((d > 0).float().mean()):.2f}%)")

# One bf16 ulp of the larger operand: bf16 keeps 8 total mantissa bits (7 stored).
m = torch.maximum(oa.abs(), ob.abs())
exp = torch.floor(torch.log2(m.clamp_min(1e-30)))
ulp = torch.pow(2.0, exp - 7)
within1 = (d <= ulp) | (m == 0)
print(f"within one bf16 ulp  : {int(within1.sum())} of {d.numel()} ({100.0 * float(within1.float().mean()):.2f}%)")
worst = int((d / ulp.clamp_min(1e-30)).argmax())
print(f"worst element        : {float(d.flatten()[worst]):.6e} = {float((d / ulp.clamp_min(1e-30)).flatten()[worst]):.2f} ulp")

# A dropped token shows as a row that is all zero on one side only.
za = (oa.abs().sum(dim=-1) == 0)
zb = (ob.abs().sum(dim=-1) == 0)
print(f"all-zero rows        : std {int(za.sum())}, moonep {int(zb.sum())}, differing {int((za ^ zb).sum())}")
rn = (oa - ob).norm(dim=-1) / oa.norm(dim=-1).clamp_min(1e-30)
print(f"per-token rel norm   : max {float(rn.max()):.3e}  median {float(rn.median()):.3e}")
