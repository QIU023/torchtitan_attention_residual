"""Cell 9: do tokens land only in this rank's home rows and its prefetch slots?"""
import re
import sys
import torch

blob = torch.load(sys.argv[1], weights_only=False)
rank, world = 0, int(sys.argv[2])
cap = {k: v for k, v in blob["capture"].items() if "rows" in v}
print(f"loss {blob['loss']:.6f}   layers captured {len(cap)}")

names = sorted(cap, key=lambda n: int(re.search(r"layers\.(\d+)", n).group(1)))
total = len(names)
n_rows = int(cap[names[0]]["rows"].numel())
bad, checked = [], 0
E = None
for n in names:
    rows = cap[n]["rows"].to(torch.int64)
    if E is None:
        # rows is [E + B]; B = E / world, so E = n_rows * world / (world + 1).
        E = n_rows * world // (world + 1)
        B = n_rows - E
        lo, hi = rank * (E // world), (rank + 1) * (E // world)
        print(f"rows per layer {n_rows} = E {E} + B {B}; rank {rank} homes experts [{lo}, {hi})")
    foreign = torch.cat([rows[:lo], rows[hi:E]])
    checked += 1
    if int(foreign.sum()) != 0:
        bad.append((n, int(foreign.sum())))

home = torch.stack([cap[n]["rows"][:E][lo:hi].sum() for n in names])
slots = torch.stack([cap[n]["rows"][E:].sum() for n in names])
print(f"layers with tokens in a foreign expert's row: {len(bad)} of {checked}")
for n, v in bad[:5]:
    print(f"    {n}: {v} tokens")
print(f"home rows per layer : min {int(home.min())} max {int(home.max())}")
print(f"slot rows per layer : min {int(slots.min())} max {int(slots.max())}, layers using a slot {int((slots > 0).sum())} of {total}")
