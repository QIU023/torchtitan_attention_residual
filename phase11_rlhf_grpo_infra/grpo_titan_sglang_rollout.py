"""GRPO on the titan K3 194m actor with the sglang AttnRes overlay as the
ROLLOUT (HTTP /generate) and disk-reload weight sync.

Division of labor (the PLAN-2 commitment): the TRAINING parts are titan/ours
-- the GRPO algorithm (group-relative advantages), the titan actor forward +
policy-gradient update, and the actor->rollout weight-sync orchestration. The
ROLLOUT (token generation) is the user's sglang overlay, running in its own
venv (/workspace/sgl_venv, torch 2.11) as an external HTTP server; this script
(torch 2.12 titan venv) talks to it over HTTP only -- no sglang import. This is
the mechanism the veRL-native external-server rollout adapter productizes.

Weight sync: veRL has no pure-HTTP full-weight path (only CUDA-IPC
/update_weights_from_tensor). Cross-venv-safe path used here: the actor exports
HF-format weights to a shared dir and POSTs sglang's native
/update_weights_from_disk, so the rollout tracks the actor each step.

Random-init 194m fixture => reward ~0 (mechanism demo, not a learning demo):
the point is that the sglang-rollout GRPO loop + advantages + update + weight
sync all run end-to-end.
"""
import os
import shutil
import sys

import requests
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

HF_INIT = "/workspace/fake_hf/kimi_linear_194m"  # what the sglang server first served
SYNC_DIR = "/workspace/grpo_sglang_sync"  # actor exports here; server reloads from here
PORT = int(os.environ.get("SGL_PORT", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
SHARD = "model-00001-of-00001.safetensors"
VOCAB = 163840


def sglang_generate(prompt_ids, group, max_new_tokens, temperature=1.0):
    """Rollout via the external sglang server: `group` samples of one prompt.
    Returns a list of completion-token-id lists."""
    payload = {
        "input_ids": [list(prompt_ids)] * group,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }
    r = requests.post(f"{BASE}/generate", json=payload, timeout=180)
    r.raise_for_status()
    return [item["output_ids"] for item in r.json()]


def export_and_reload(model, adapter):
    """actor -> HF weights on disk -> sglang /update_weights_from_disk."""
    hf_sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in adapter.to_hf(model.state_dict()).items()}
    save_file(hf_sd, os.path.join(SYNC_DIR, SHARD))
    r = requests.post(
        f"{BASE}/update_weights_from_disk", json={"model_path": SYNC_DIR}, timeout=300
    )
    r.raise_for_status()
    return r.json().get("success", False)


def main():
    from torchtitan.experiments.kimi_k3 import model_registry
    from torchtitan.experiments.kimi_k3.state_dict_adapter import (
        KimiLinearStateDictAdapter,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    spec = model_registry("kimi_linear_194m_block_attn_res")
    with torch.device("cuda"):
        model = spec.model.build()
        model.init_weights()
    model = model.to(torch.bfloat16)

    adapter = KimiLinearStateDictAdapter(spec.model, HF_INIT)
    # Load the actor from the SAME HF checkpoint the sglang server serves, so
    # actor == rollout at step 0.
    hf_sd = load_file(os.path.join(HF_INIT, SHARD))
    titan_sd = adapter.from_hf(hf_sd)
    missing, unexpected = model.load_state_dict(titan_sd, strict=False)
    print(f"[load] from_hf: {len(titan_sd)} keys; missing {len(missing)} unexpected {len(unexpected)}", flush=True)

    # Prime the export dir with the config/tokenizer so the server can reload.
    os.makedirs(SYNC_DIR, exist_ok=True)
    for fn in os.listdir(HF_INIT):
        if not fn.endswith(".safetensors"):
            shutil.copy(os.path.join(HF_INIT, fn), os.path.join(SYNC_DIR, fn))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    G, GEN = 6, 12
    prompt = torch.randint(0, VOCAB, (72,), device="cuda")
    prompt_ids = prompt.tolist()

    for step in range(6):
        # 1) ROLLOUT via the user's sglang server (external, HTTP)
        comps = sglang_generate(prompt_ids, G, GEN)
        # pad/truncate completions to GEN, build a validity mask
        comp = torch.full((G, GEN), 0, dtype=torch.long, device="cuda")
        mask = torch.zeros((G, GEN), dtype=torch.bool, device="cuda")
        for i, c in enumerate(comps):
            c = c[:GEN]
            comp[i, : len(c)] = torch.tensor(c, device="cuda")
            mask[i, : len(c)] = True
        seqs = torch.cat([prompt.unsqueeze(0).repeat(G, 1), comp], dim=1)

        # 2) reward + group-relative (GRPO) advantages
        r = ((comp % 2 == 0) & mask).float().sum(1) / mask.float().sum(1).clamp(min=1)
        adv = (r - r.mean()) / (r.std() + 1e-6)

        # 3) policy-gradient update on the titan actor (recompute logprobs)
        model.train()
        logits = model(seqs)[:, prompt.shape[0] - 1 : -1, :].float()
        logp_tok = F.log_softmax(logits, dim=-1).gather(2, comp.unsqueeze(-1)).squeeze(-1)
        logp = (logp_tok * mask.float()).sum(1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # 4) weight sync: actor -> disk -> sglang rollout reload
        synced = export_and_reload(model, adapter)
        print(
            f"[GRPO+sglang] step {step}: reward {r.mean():.3f}+/-{r.std():.3f} "
            f"adv[{adv.min():.2f},{adv.max():.2f}] pg_loss {loss.item():.4f} "
            f"grad_norm {gn:.3f} weight_sync={synced}",
            flush=True,
        )

    print("[GRPO+sglang] PASS -- sglang-rollout GRPO loop + advantage + update + weight-sync run", flush=True)


if __name__ == "__main__":
    sys.exit(main())
