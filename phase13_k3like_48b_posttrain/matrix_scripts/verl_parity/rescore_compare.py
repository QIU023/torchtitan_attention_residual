import os, torch
D = os.environ.get("DUMP", "/workspace/logprob_dump2"); TAG = os.environ.get("TAG", "")
b = torch.load(f"{D}/batch.pt"); tt = torch.load(f"{D}/rescore_tt.pt"); vl = torch.load(f"{D}/rescore_vllm{TAG}.pt")
pairs = {"engine(old_log_probs) vs rollout(vLLM server)": [], "engine vs plain torchtitan": [], "rollout vs offline vLLM": [],
         "plain torchtitan vs offline vLLM": []}
first = {k: [] for k in pairs}
for i in sorted(tt):
    Lr = tt[i]["Lr"]; e = b["old_log_probs"][i, :Lr]; r = b["rollout_log_probs"][i, :Lr]; t = tt[i]["lp"]; v = vl[i]
    for k, (x, y) in zip(pairs, [(e, r), (e, t), (r, v), (t, v)]):
        d = (x - y).abs(); pairs[k].append(d); first[k].append(d[:6])
for k, ds in pairs.items():
    d = torch.cat(ds); print(f"{k:48s} mean|d|={d.mean():.4f}  frac>0.5={(d > 0.5).float().mean():.3f}  max={d.max():.3f}  first6(seq0)={[round(x, 3) for x in first[k][0].tolist()]}")
# cross-check the row alignment of the engine array: does row i match another row's plain rescore better?
i0 = sorted(tt)[0]; e = b["old_log_probs"][i0]
best = min(((( e[:tt[j]["Lr"]] - tt[j]["lp"]).abs().mean().item(), j) for j in tt))
print("engine row", i0, "closest plain-rescore row:", best)
