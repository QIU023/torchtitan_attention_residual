"""Single-process GRPO — no monarch, no torchstore.

Monarch's actor-mesh first-rollout spawn hangs ~100% on this box, so the whole
GRPO loop runs in one process:
  - policy VLM (frozen SigLIP + trainable Projector + Kimi-Linear AttnRes LM) on
    cuda:0 for teacher-forcing logprob + backward
  - sglang Engine on GPU1 (base_gpu_id=1, fa3 backend) for rollout generation
  - verifiable exact-match GQA reward (pure python)

Modes: load -> gen -> one -> loop.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

REPO = "/home/torchtitan_attention_residual"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/torchtitan")

HF = REPO + "/phase11_rlhf_grpo_infra/hf/popefix_447m"
DCP = REPO + "/phase5_vlm_multimodal_sft/runs/popefix_447m/checkpoint/step-593"
GQA = "/home/.hf_home/gqa_rl/gqa_rl_train.json"
IMG_ROOT = "/home/.hf_home/LLaVA-Instruct/images"
LM_CONFIG = "kimi_linear_447m_aligned_block_attn_res_n4"
VISION = "google/siglip-base-patch16-224"
TOKENIZER = "NousResearch/Meta-Llama-3.1-8B"
N_VIS = 196


def log(*a):
    print(f"[grpo-sp {time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_policy(device="cuda:0"):
    import torch
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModel
    from torchtitan.experiments.kimi_linear import config_registry as cr
    from phase5_vlm_multimodal_sft.multimodal_model import Projector
    dt = torch.bfloat16
    log(f"vision {VISION} (frozen)")
    vision = AutoModel.from_pretrained(VISION).vision_model.to(device, dt).eval()
    for p in vision.parameters():
        p.requires_grad_(False)
    log(f"LM {LM_CONFIG}")
    spec = getattr(cr, LM_CONFIG)().model_spec
    minfo = spec.model
    lm = minfo.build().to(device, dt)
    proj = Projector(vision.config.hidden_size, minfo.kimi_config.hidden_size).to(device, dt)
    log(f"dcp.load {DCP}")
    # popefix DCP was saved by train_mm.py: LM keys under 'model.', projector
    # under 'mm_state.projector.'. Load into the live tensors in place.
    # DCP layout (verified): LM keys are TOP-LEVEL (embed_tokens, layers.*, norm,
    # lm_head, final_attn_res_*); projector under mm_state.projector.*.
    lm_sd = dict(lm.state_dict())  # bare keys match DCP top level
    proj_sd = {f"mm_state.projector.{k}": v for k, v in proj.state_dict().items()}
    load_sd = {**lm_sd, **proj_sd}
    dcp.load(load_sd, checkpoint_id=DCP)
    lm.load_state_dict({k: v for k, v in load_sd.items() if not k.startswith("mm_state.")}, strict=False)
    proj.load_state_dict({k[len("mm_state.projector."):]: v for k, v in load_sd.items()
                          if k.startswith("mm_state.projector.")}, strict=False)
    log("policy loaded")
    return vision, lm, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["load", "gen", "one", "loop"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()
    os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
    os.environ.setdefault("SGLANG_DISABLE_SHM_MM", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    # fla KDA Triton kernel JIT-parse fails in sglang venv ("Unsupported
    # function referenced: next_power_of_2") with a stale triton disk cache;
    # disabling line-info side-steps the parse path. Verified policy fwd OK.
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "1")

    import torch
    import torch.nn.functional as F
    from phase5_vlm_multimodal_sft.multimodal_dataset import IMAGE_TOKEN_ID

    dev = "cuda:0"
    vision, lm, proj = build_policy(dev)
    if args.mode == "load":
        log("LOAD OK"); return

    log("sglang Engine GPU1 (fa3)")
    from sglang.srt.models import attn_res_vl_overlay  # noqa
    from sglang.srt.configs.kimi_attn_res_vl import KimiAttnResVLConfig  # noqa
    from sglang.srt.entrypoints.engine import Engine
    eng = Engine(model_path=HF, tp_size=1, base_gpu_id=1, trust_remote_code=True,
                 mem_fraction_static=0.55, disable_cuda_graph=True,
                 attention_backend="fa3", log_level="warning")
    log("engine ready")

    from transformers import AutoTokenizer, AutoProcessor
    from PIL import Image
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    improc = AutoProcessor.from_pretrained(VISION).image_processor
    recs = json.load(open(GQA))
    rng = random.Random(0)

    def sample():
        r = rng.choice(recs)
        p = "<image>\n" + r["question"] + "\nAnswer the question using a single word or phrase."
        return p, os.path.join(IMG_ROOT, r["image"]), r["answer"].strip().lower()

    if args.mode == "gen":
        p, img, gold = sample()
        out = eng.generate(prompt=[p], sampling_params={"temperature": 0.0, "max_new_tokens": args.max_new}, image_data=[img])
        log("GEN OK:", out[0].get("text"), "| gold:", gold); return

    def pix(img_path):
        im = Image.open(img_path).convert("RGB")
        return improc(images=im, return_tensors="pt").pixel_values.to(dev, torch.bfloat16)

    def grpo_loss(prompt_text, img_path, comps, adv):
        with torch.no_grad():
            vf = vision(pixel_values=pix(img_path)).last_hidden_state
        vemb = proj(vf)  # [1,196,dim]
        ptxt = prompt_text.replace("<image>", "")
        pids = tok(ptxt, add_special_tokens=False).input_ids
        imgids = [IMAGE_TOKEN_ID] * N_VIS
        total = torch.zeros((), device=dev, dtype=torch.float32)
        for c, a in zip(comps, adv):
            aids = tok(c, add_special_tokens=False).input_ids
            if not aids:
                continue
            ids = torch.tensor([imgids + pids + aids], device=dev)
            mask = (ids == IMAGE_TOKEN_ID)
            logits = lm(ids, vision_embeds=vemb, image_mask=mask).float()
            n = len(aids)
            tgt = ids[0, -n:]
            pred = logits[0, -n - 1:-1]
            lp = F.log_softmax(pred, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()
            total = total - a * lp
        return total / max(len(comps), 1)

    trainable = [p for p in lm.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    log(f"AdamW lr={args.lr} {sum(p.numel() for p in trainable)/1e6:.0f}M params")

    n = 1 if args.mode == "one" else args.steps
    for step in range(n):
        t0 = time.time()
        p, img, gold = sample()
        outs = eng.generate(prompt=[p],
                            sampling_params={"temperature": 1.0, "max_new_tokens": args.max_new, "n": args.group},
                            image_data=[img])
        comps = [o["text"] for o in (outs if isinstance(outs, list) else [outs])]
        rew = []
        for c in comps:
            full = " ".join(c.lower().replace(".", " ").split())
            ok = (" " + gold + " ") in (" " + full + " ") or set(gold.split()).issubset(set(full.split()))
            r = 1.0 if ok else 0.0
            wc = len(full.split())
            if wc > 12:
                r -= min(0.3, 0.02 * (wc - 12))
            rew.append(r)
        rt = torch.tensor(rew, dtype=torch.float32)
        adv = ((rt - rt.mean()) / (rt.std() + 1e-6)).to(dev)
        loss = grpo_loss(p, img, comps, adv)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        log(f"step {step:3d}  loss={loss.item():.4f}  reward_mean={rt.mean().item():+.3f}  dt={time.time()-t0:.1f}s")
        if args.mode == "one":
            log("ONE-STEP OK"); break
    log("DONE")


if __name__ == "__main__":
    main()
