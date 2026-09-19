"""Compare two MoE traces layer by layer and name the first divergence."""
import re
import sys
import torch

a = torch.load(sys.argv[1], weights_only=False)
b = torch.load(sys.argv[2], weights_only=False)
print(f"loss: {sys.argv[1].split('/')[-1]}={a['loss']:.6f}  {sys.argv[2].split('/')[-1]}={b['loss']:.6f}")
ca, cb = a["capture"], b["capture"]

names = sorted(set(ca) & set(cb), key=lambda n: (int(re.search(r"layers\.(\d+)", n).group(1)), n))
if set(ca) != set(cb):
    print(f"WARNING: module sets differ, {len(set(ca) ^ set(cb))} only on one side")


def mx(x, y):
    return float((x - y).abs().max()) if x.shape == y.shape else float("nan")


print(f"{'module':<34} {'in maxabs':>11} {'out maxabs':>11} {'ids flips':>10} {'scores maxabs':>14}")
first_in, first_ids, first_out = None, None, None
for n in names:
    x, y = ca[n], cb[n]
    if "rows" in x:
        continue
    if "ids" in x:
        din = mx(x["x"], y["x"])
        flips = int((x["ids"] != y["ids"]).sum())
        dsc = mx(x["scores"], y["scores"])
        print(f"{n:<34} {din:>11.3e} {'':>11} {flips:>10d} {dsc:>14.3e}")
        if first_in is None and din > 0:
            first_in = (n, din)
        if first_ids is None and flips > 0:
            first_ids = (n, flips, int(x["ids"].numel()))
    else:
        din, dout = mx(x["x"], y["x"]), mx(x["out"], y["out"])
        print(f"{n:<34} {din:>11.3e} {dout:>11.3e} {'':>10} {'':>14}")
        if first_in is None and din > 0:
            first_in = (n, din)
        if first_out is None and dout > 0:
            first_out = (n, dout)

print()
print(f"first input difference : {first_in}")
print(f"first output difference: {first_out}")
print(f"first routing flip     : {first_ids}")
