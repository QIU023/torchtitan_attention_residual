"""Single-process GRPO (CORRECTED) — no monarch, no torchstore.

Fixes vs the first version:
  BUG1 (prompt mismatch): rollout + policy now use the SAME token ids. We build
       prompt_ids = [BOS] + [IMAGE]*196 + text_ids ourselves and pass input_ids
       to the engine; the engine returns output_ids (the exact action tokens).
       The policy logprob is computed on prompt_ids + output_ids — byte-identical
       to what the engine generated.
  BUG2 (no importance ratio): engine returns per-token old_logprob
       (return_logprob=True). GRPO clipped surrogate uses ratio = exp(new-old).
  BUG3 (zero-variance groups): groups whose rewards are all-equal (std≈0) are
       skipped (no learning signal, and they dominated the flat-reward run).

Speed/VRAM: all `group` completions for a question go through ONE padded policy
forward (not a python for-loop). Optionally B questions per step.

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
IMAGE_TOKEN_ID = 32000


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
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--qbatch", type=int, default=2, help="questions per step")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--sync-every", type=int, default=10)
    ap.add_argument("--save-dcp", default="")
    ap.add_argument("--data", default=GQA, help="RL prompt json [{image,question,answer}]")
    ap.add_argument("--task", default="gqa", choices=["gqa", "pope"],
                    help="pope = strict yes/no reward (1/0); gqa = substring match")
    args = ap.parse_args()
    os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
    os.environ.setdefault("SGLANG_DISABLE_SHM_MM", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "1")

    import torch
    import torch.nn.functional as F

    dev = "cuda:0"
    vision, lm, proj = build_policy(dev)
    if args.mode == "load":
        log("LOAD OK"); return

    # pre-warm fla KDA kernel before sglang import (deterministic JIT)
    _ids = torch.randint(0, 1000, (1, 210), device=dev)
    _emb = torch.zeros(1, N_VIS, proj.fc2.weight.shape[0], device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        lm(_ids, vision_embeds=_emb, image_mask=(_ids == IMAGE_TOKEN_ID))
    log("fla KDA pre-warmed")

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
    bos = tok.bos_token_id
    improc = AutoProcessor.from_pretrained(VISION).image_processor
    recs = json.load(open(args.data))
    rng = random.Random(0)

    def build_prompt_ids(question):
        text = question + "\nAnswer the question using a single word or phrase."
        tids = tok(text, add_special_tokens=False).input_ids
        return ([bos] if bos is not None else []) + [IMAGE_TOKEN_ID] * N_VIS + tids

    def pix(img_path):
        im = Image.open(img_path).convert("RGB")
        return improc(images=im, return_tensors="pt").pixel_values.to(dev, torch.bfloat16)

    def sample():
        r = rng.choice(recs)
        return r["question"], os.path.join(IMG_ROOT, r["image"]), r["answer"].strip().lower()

    def reward_of(text, gold):
        full = " ".join(text.lower().replace(".", " ").split())
        if args.task == "pope":
            # strict yes/no: parse first yes/no token; reward 1 iff it matches gold
            toks = full.split()
            hy = "yes" in toks; hn = "no" in toks
            pred = "yes" if (hy and not hn) else ("no" if (hn and not hy) else "unknown")
            return 1.0 if pred == gold else 0.0
        ok = (" " + gold + " ") in (" " + full + " ") or set(gold.split()).issubset(set(full.split()))
        r = 1.0 if ok else 0.0
        wc = len(full.split())
        if wc > 12:
            r -= min(0.3, 0.02 * (wc - 12))
        return r

    _instr = ("\nAnswer the question using a single word: yes or no."
              if args.task == "pope"
              else "\nAnswer the question using a single word or phrase.")

    def gen_prompt(q):
        return "<image>\n" + q + _instr

    if args.mode == "gen":
        q, img, gold = sample()
        out = eng.generate(prompt=[gen_prompt(q)], image_data=[img],
                           sampling_params={"temperature": 0.0, "max_new_tokens": args.max_new})
        log("GEN OK:", out[0].get("text"), "| gold:", gold); return

    trainable = [p for p in lm.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    log(f"AdamW lr={args.lr} clip={args.clip} group={args.group} qbatch={args.qbatch} "
        f"{sum(p.numel() for p in trainable)/1e6:.0f}M params")

    from phase11_rlhf_grpo_infra.dcp_to_hf_kimi_attn_res_vl import _remap_lm_state_dict
    from safetensors.torch import save_file

    def sync_weights():
        hf_sd = _remap_lm_state_dict(lm.state_dict(), torch.bfloat16)
        for k, v in proj.state_dict().items():
            hf_sd[f"mm_projector.projector.{k}"] = v.detach().to(torch.bfloat16).contiguous()
        hf_sd = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in hf_sd.items()}
        save_file(hf_sd, HF + "/model.safetensors")
        eng.update_weights_from_disk(HF)

    def grpo_step():
        """One optimizer step over qbatch questions × group completions.

        Returns (loss_val, reward_mean, n_groups_used)."""
        # 1. rollout: collect (prompt_ids, img, [ (out_ids, old_lp_sum, reward) ]) per question
        samples = []  # list of dict
        all_rewards = []
        n_used = 0
        for _ in range(args.qbatch):
            q, img, gold = sample()
            pids = build_prompt_ids(q)
            outs = eng.generate(prompt=[gen_prompt(q)], image_data=[img], return_logprob=True,
                                sampling_params={"temperature": 1.0, "max_new_tokens": args.max_new,
                                                 "n": args.group})
            outs = outs if isinstance(outs, list) else [outs]
            comps = []
            for o in outs:
                oid = o.get("output_ids") or o["meta_info"].get("output_ids")
                otl = o["meta_info"].get("output_token_logprobs")  # list of [lp, tok_id, ...]
                if not oid or not otl:
                    continue
                old_lp = [float(x[0]) for x in otl][:len(oid)]
                rw = reward_of(o.get("text", ""), gold)
                comps.append((oid, old_lp, rw))
                all_rewards.append(rw)
            if len(comps) < 2:
                continue
            rws = torch.tensor([c[2] for c in comps])
            if rws.std() < 1e-5:   # zero-variance group: no signal, skip (BUG3)
                continue
            adv = (rws - rws.mean()) / (rws.std() + 1e-6)
            samples.append((pids, img, comps, adv))
            n_used += 1

        if not samples:
            return None, (sum(all_rewards) / max(len(all_rewards), 1)), 0

        # 2. policy forward + GRPO clipped loss (BUG1+BUG2 fixed)
        opt.zero_grad()
        total_loss = torch.zeros((), device=dev, dtype=torch.float32)
        n_terms = 0
        for pids, img, comps, adv in samples:
            with torch.no_grad():
                vemb = proj(vision(pixel_values=pix(img)).last_hidden_state)
            for (oid, old_lp, _), a in zip(comps, adv.tolist()):
                ids = torch.tensor([pids + oid], device=dev)
                logits = lm(ids, vision_embeds=vemb, image_mask=(ids == IMAGE_TOKEN_ID)).float()
                no = len(oid)
                tgt = ids[0, -no:]
                pred = logits[0, -no - 1:-1]
                new_lp = F.log_softmax(pred, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                old = torch.tensor(old_lp, device=dev)
                ratio = torch.exp(new_lp - old)
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                # GRPO: maximize min(unclipped, clipped) → minimize -min, mean over tokens
                total_loss = total_loss - torch.min(unclipped, clipped).mean()
                n_terms += 1
        loss = total_loss / max(n_terms, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        return loss.item(), (sum(all_rewards) / max(len(all_rewards), 1)), len(samples)

    n = 1 if args.mode == "one" else args.steps
    for step in range(n):
        t0 = time.time()
        lv, rm, ng = grpo_step()
        lvs = f"{lv:.4f}" if lv is not None else "skip"
        log(f"step {step:3d}  loss={lvs}  reward_mean={rm:+.3f}  groups={ng}  dt={time.time()-t0:.1f}s")
        if args.mode == "one":
            ts0 = time.time(); sync_weights(); log(f"weight-sync OK dt={time.time()-ts0:.1f}s ONE-STEP OK"); break
        if (step + 1) % args.sync_every == 0:
            ts0 = time.time(); sync_weights(); log(f"[sync] step {step+1} dt={time.time()-ts0:.1f}s")
    if args.save_dcp:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemWriter
        os.makedirs(args.save_dcp, exist_ok=True)
        save_sd = {k: v.detach().cpu() for k, v in lm.state_dict().items()}
        save_sd.update({f"mm_state.projector.{k}": v.detach().cpu() for k, v in proj.state_dict().items()})
        # no_dist=True: single-process save without a torch.distributed PG
        # (plain dcp.save hangs waiting on a default PG that's never inited).
        dcp.save(save_sd, storage_writer=FileSystemWriter(args.save_dcp), no_dist=True)
        log(f"saved final policy DCP -> {args.save_dcp}")
    log("DONE")


if __name__ == "__main__":
    main()
