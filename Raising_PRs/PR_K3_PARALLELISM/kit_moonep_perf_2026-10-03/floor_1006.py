"""Step-1 per-parameter gradient noise floor from the 10-05 dumps: standard EP under another expert partition (same data)."""
import torch

d = "/root/mep/results/moonep_final/grads"
g = {n: torch.load(f"{d}/{n}.pt", map_location="cpu") for n in ("standard", "moonep", "standard_ep2", "standard_hsdp")}


def rel(a, b):
    return {k: ((a[k].float() - b[k].float()).norm() / a[k].float().norm()).item() for k in a if a[k].float().norm() > 0}


def routed(k):
    return "experts" in k and "shared" not in k


mep = rel(g["standard"], g["moonep"])
for name in ("standard_ep2", "standard_hsdp"):
    floor = rel(g["standard"], g[name])
    keys = sorted(set(mep) & set(floor))
    med = lambda r, ks: sorted(r[k] for k in ks)[len(ks) // 2]
    rk = [k for k in keys if routed(k)]
    above = [k for k in keys if mep[k] > floor[k]]
    print(f"floor = standard FSDP 4 x EP 4 vs {name}: {len(keys)} parameters")
    print(f"  median d: floor {med(floor, keys):.2e}, MoonEP {med(mep, keys):.2e}")
    print(f"  routed experts ({len(rk)}): floor max {max(floor[k] for k in rk):.2e} median {med(floor, rk):.2e}; MoonEP max {max(mep[k] for k in rk):.2e} median {med(mep, rk):.2e}")
    print(f"  parameters where MoonEP's d exceeds the floor's: {len(above)} of {len(keys)}; routed experts among them: {sum(routed(k) for k in above)}")
    worst = sorted(keys, key=lambda k: mep[k] / max(floor[k], 1e-12), reverse=True)[:4]
    for k in worst:
        print(f"    MoonEP {mep[k]:.2e} vs floor {floor[k]:.2e}  {k}")
