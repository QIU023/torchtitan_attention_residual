"""dp1's stream against the cp2 rank shards, layer by layer (first micro-batch of step 1)."""
import sys, torch
D = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cp_layer"
a = torch.load(f"{D}/dp1.rank0.pt"); r = [torch.load(f"{D}/cp2.rank{i}.pt") for i in (0, 1)]
T = a["tokens"].shape[0]; L = T // 2
print("tokens match:", all(torch.equal(a["tokens"][i*L:(i+1)*L], r[i]["tokens"]) for i in (0, 1)), "T", T)
keys = [k for k in a if k != "tokens"]
print(f"{'layer':6s} {'rank0: rel diff':>16s} {'max|d|':>9s} {'bitwise':>8s} | {'rank1: rel diff':>16s} {'max|d|':>9s} {'bitwise':>8s}")
for k in keys:
    row = f"{k:6s}"
    for i in (0, 1):
        x = a[k][i*L:(i+1)*L].float(); y = r[i][k].float()
        rel = ((x - y).norm() / max(x.norm().item(), 1e-30)).item(); mx = (x - y).abs().max().item()
        row += f" {rel:16.3e} {mx:9.3e} {str(torch.equal(x, y)):>8s}" + (" |" if i == 0 else "")
    print(row)
