"""Rescore verl's own sampled sequences with a plain single-sequence torchtitan forward (rl flavor, HF export)."""
import glob, os, sys, torch
EXPORT = os.environ.get("EXPORT", "/root/models/kimi-k3-debug-nt")
D = os.environ.get("DUMP", "/workspace/logprob_dump2")
b = torch.load(f"{D}/batch.pt")
prompts, plen, resp, rmask = b["prompts"], b["prompt_lengths"], b["responses"], b["response_mask"]
B = prompts.shape[0]; N = int(os.environ.get("N", "40")); idx = list(range(0, B, max(1, B // N)))[:N]
print("batch", B, "prompt lens", plen[:8].tolist(), "resp lens", rmask.sum(1)[:8].tolist(), flush=True)
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from safetensors.torch import load_file
cfg = model_registry("rl").model; model = cfg.build()
hf = {}
for f in sorted(glob.glob(f"{EXPORT}/*.safetensors")): hf.update(load_file(f))
sd = KimiK3StateDictAdapter(cfg, hf_assets_path=None).from_hf(hf)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("load: missing", len(missing), missing[:6], "unexpected", len(unexpected), unexpected[:6], flush=True)
model = model.to(torch.bfloat16).cuda().eval()
out = {}
for i in idx:
    Lp = int(plen[i]); Lr = int(rmask[i].sum())
    seq = torch.cat([prompts[i, :Lp], resp[i, :Lr]]).cuda(); T = seq.numel(); pos = torch.arange(T, device="cuda")
    masks = model.get_attention_masks(positions=pos)
    with torch.no_grad(): logits = model(seq, positions=pos, attention_masks=masks)
    lp = torch.log_softmax(logits.float(), -1)
    r = lp[torch.arange(Lp - 1, Lp + Lr - 1, device="cuda"), seq[Lp:Lp + Lr]].cpu()
    out[i] = {"lp": r, "Lp": Lp, "Lr": Lr, "seq": seq.cpu()}
    if len(out) % 10 == 0: print("done", len(out), flush=True)
torch.save(out, f"{D}/rescore_tt.pt"); print("saved", len(out), "sequences", flush=True)
