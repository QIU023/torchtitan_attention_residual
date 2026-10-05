"""Compare step-1 gradient dumps. Per parameter: d = ||a - b|| / ||a||. Pairs: the floor (standard twice), MoonEP
against standard at efsdp 1, 2 and HSDP; for efsdp 2 and HSDP every parameter's d is also divided by its d at
efsdp 1, the accepted baseline, and the routed experts (the MoonEP-owned gradients) are listed on their own."""
import sys

import torch


def rel(a, b):
    out = {}
    for key in a:
        na = a[key].norm().item()
        out[key] = ((a[key] - b[key]).norm().item() / na) if na > 0 else (a[key] - b[key]).norm().item()
    return out


def routed(key):
    return "experts" in key and "shared" not in key


d = sys.argv[1]
g = {n: torch.load(f"{d}/{n}.pt", map_location="cpu") for n in
     ("standard", "standard_b", "moonep", "standard_ep2", "moonep_ep2", "standard_hsdp", "moonep_hsdp")}
pairs = {
    "floor, standard twice (FSDP 4 x EP 4)": rel(g["standard"], g["standard_b"]),
    "efsdp 1, MoonEP vs standard (FSDP 4 x EP 4)": rel(g["standard"], g["moonep"]),
    "efsdp 2, MoonEP vs standard (dp_shard 4 x EP 2)": rel(g["standard_ep2"], g["moonep_ep2"]),
    "HSDP, MoonEP vs standard (2 x 2 x EP 2)": rel(g["standard_hsdp"], g["moonep_hsdp"]),
}
base = pairs["efsdp 1, MoonEP vs standard (FSDP 4 x EP 4)"]
for name, r in pairs.items():
    keys = list(r)
    zero = sum(1 for k in keys if r[k] == 0.0)
    rk = [k for k in keys if routed(k)]
    med = sorted(r.values())[len(r) // 2]
    print(f"{name}: {len(keys)} parameters, {zero} bitwise equal; median d {med:.2e}; max d {max(r.values()):.2e} "
          f"({max(keys, key=r.get)}); routed experts ({len(rk)}) max d {max((r[k] for k in rk), default=0):.2e}")
    if "MoonEP" in name and "efsdp 1" not in name:
        ratio = {k: r[k] / max(base[k], 1e-12) for k in keys}
        worst = sorted(keys, key=ratio.get, reverse=True)[:5]
        print(f"    against efsdp 1 per parameter: max ratio {ratio[worst[0]]:.2f}; parameters above 3x and d > 1e-2: "
              f"{sum(1 for k in keys if ratio[k] > 3 and r[k] > 1e-2)}")
        for k in worst:
            print(f"      {ratio[k]:6.2f}x  d {r[k]:.2e} (efsdp 1: {base[k]:.2e})  {k}")
