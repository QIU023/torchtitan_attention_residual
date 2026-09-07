"""Near-fp32 torchtitan reference on verl's sampled sequences (fp32 everything but the MLA attention op, which has no fp32
flex configuration on this card), compared with the bf16 rescore, the engine's old_log_probs and the rollout's log-probs."""
import glob, os, time, torch
EXPORT = os.environ.get("EXPORT", "/root/models/kimi-k3-debug-nt"); D = os.environ.get("DUMP", "/workspace/logprob_dump2")
b = torch.load(f"{D}/batch.pt"); prompts, plen, resp, rmask = b["prompts"], b["prompt_lengths"], b["responses"], b["response_mask"]
B = prompts.shape[0]; N = int(os.environ.get("N", "40")); idx = list(range(0, B, max(1, B // N)))[:N]
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from safetensors.torch import load_file
cfg = model_registry("rl").model; model = cfg.build()
hf = {}
for f in sorted(glob.glob(f"{EXPORT}/*.safetensors")): hf.update(load_file(f))
model.load_state_dict(KimiK3StateDictAdapter(cfg, hf_assets_path=None).from_hf(hf), strict=False)
model = model.float().cuda().eval()
def _bf16_attention(inner):
    orig = inner.forward
    def fwd(q, k, v, **kw): return orig(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), **kw).to(q.dtype)
    inner.forward = fwd
for layer in model.layers.values():
    if layer.attention is not None: _bf16_attention(layer.attention.inner_attention)
out = {}
for i in idx:
    Lp = int(plen[i]); Lr = int(rmask[i].sum())
    seq = torch.cat([prompts[i, :Lp], resp[i, :Lr]]).cuda(); T = seq.numel(); pos = torch.arange(T, device="cuda")
    with torch.no_grad(): logits = model(seq, positions=pos, attention_masks=model.get_attention_masks(positions=pos))
    lp = torch.log_softmax(logits.float(), -1); out[i] = lp[torch.arange(Lp - 1, Lp + Lr - 1, device="cuda"), seq[Lp:Lp + Lr]].cpu()
torch.save(out, f"{D}/rescore_tt_fp32.pt"); print("fp32 reference saved", flush=True)
while not os.path.exists(f"{D}/rescore_tt.pt"): time.sleep(20)
tt = torch.load(f"{D}/rescore_tt.pt")
def stat(name, pairs):
    d = torch.cat([(x - y).abs() for x, y in pairs]); print(f"{name:40s} mean|d|={d.mean():.4f} frac>0.5={(d > 0.5).float().mean():.3f} max={d.max():.3f}")
E = {i: b["old_log_probs"][i, :tt[i]["Lr"]] for i in idx}; R = {i: b["rollout_log_probs"][i, :tt[i]["Lr"]] for i in idx}
stat("fp32 ref vs bf16 plain torchtitan", [(out[i], tt[i]["lp"]) for i in idx])
stat("fp32 ref vs engine old_log_probs", [(out[i], E[i]) for i in idx])
stat("fp32 ref vs rollout log-probs", [(out[i], R[i]) for i in idx])
stat("engine vs bf16 plain torchtitan", [(E[i], tt[i]["lp"]) for i in idx])
stat("rollout vs bf16 plain torchtitan", [(R[i], tt[i]["lp"]) for i in idx])
print("VARIANTS-DONE")
