"""Phase 9 PPO production post-training — minimal real-ckpt actor+ref smoke.

Loads the phase4 step-12500 trained 447M Kimi Linear AttnRes ckpt as
both the actor (trainable) and frozen reference, runs a single PPO
training step:

    1. Actor + ref forward on a synthetic prompt+response batch.
    2. KL = mean(actor_logprob - ref_logprob).
    3. Mock reward: random scalar per sample.
    4. Advantage = reward - mean(reward).
    5. PPO clipped surrogate loss + KL penalty.
    6. Backward + optimizer step on actor only.

This validates the dual-model fwd/bwd path on the trained ckpt; it
does NOT include the rollout phase (SGLang generation + weight sync
back to torchtitan), which needs monarch/torchstore — separate
multi-day scope. With rollout, this becomes proper PPO.

Run:
    NGPU=8 torchrun --nproc_per_node=$NGPU \
        phase9/ppo_actor_ref_real_ckpt.py \
        --ckpt phase4/runs/lm_447m_base/checkpoint/step-12500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))
sys.path.insert(0, str(_WS / "torchtitan"))


def _build_model():
    from torchtitan.experiments.kimi_linear.config_registry import (
        kimi_linear_447m_aligned_block_attn_res_n4,
    )
    from torchtitan.experiments.kimi_linear.attn_res_model import (
        KimiLinearAttnResModel,
    )
    cfg = kimi_linear_447m_aligned_block_attn_res_n4()
    spec = cfg.model_spec.model
    m = KimiLinearAttnResModel(spec.kimi_config, num_blocks=spec.num_blocks)
    m.init_weights()
    return spec.kimi_config, m


def _logprobs(logits, target_ids):
    """Compute per-token logprobs along the response side (right-shifted)."""
    # logits: (B, T, V), target_ids: (B, T)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.2)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"[ppo] world_size={world} ckpt={args.ckpt}")

    # Build actor (trainable) and ref (frozen).
    cfg, actor = _build_model()
    _, ref = _build_model()
    actor = actor.cuda().bfloat16()
    ref = ref.cuda().bfloat16()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Load ckpt into both via DCP (single-host single-rank load — small ckpt).
    if rank == 0:
        print(f"[ppo] loading DCP into actor + ref ...")
    sd = actor.state_dict()
    dcp.load(sd, checkpoint_id=str(args.ckpt))
    actor.load_state_dict(sd, strict=False)
    ref.load_state_dict(sd, strict=False)
    if rank == 0:
        print(f"[ppo] loaded ({sum(p.numel() for p in actor.parameters())/1e6:.0f}M params)")

    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, betas=(0.9, 0.95))

    # FSDP-shard the actor (ref stays replicated per rank — small enough).
    # actor.layers is a ModuleDict (str → layer), not ModuleList.
    from torch.distributed._composable.fsdp import fully_shard
    for layer in actor.layers.values():
        fully_shard(layer)
    fully_shard(actor)

    actor.train()
    ref.eval()

    for step in range(args.steps):
        # Synthetic prompt+response batch (random ids — no rollout phase here).
        torch.manual_seed(step * 100 + rank)
        ids = torch.randint(
            1, cfg.vocab_size, (args.batch, args.seq), device="cuda"
        )
        # Right-shifted target = ids itself (next-token prediction).
        target = ids
        # Mock reward: random scalar per sample.
        reward = torch.randn(args.batch, device="cuda")

        t0 = time.perf_counter()

        # --- Actor forward (with grad) ---
        actor_logits = actor(ids)            # (B, T, V)
        actor_lp = _logprobs(actor_logits, target).sum(-1)  # (B,)

        # --- Ref forward (no grad) ---
        with torch.no_grad():
            ref_logits = ref(ids)
            ref_lp = _logprobs(ref_logits, target).sum(-1)  # (B,)

        # --- KL (per-sample) ---
        kl = (actor_lp - ref_lp)            # (B,)

        # --- Advantage: centered reward ---
        # In real PPO this is GAE; here we just z-score.
        adv = reward - reward.mean()
        adv = adv / (adv.std() + 1e-8)

        # --- PPO clipped surrogate (single-step "ratio" = 1 since fresh) ---
        ratio = torch.exp(actor_lp - actor_lp.detach())  # = 1 fresh; placeholder
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
        ppo_loss = -torch.min(s1, s2).mean()
        kl_pen = args.kl_coef * kl.mean()
        loss = ppo_loss + kl_pen

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        # Reduce metrics across world for clean rank-0 print.
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(kl_pen, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(
                f"[ppo] step {step:2d}  "
                f"loss={loss.item():.4f}  "
                f"ppo={ppo_loss.item():.4f}  "
                f"kl_pen={kl_pen.item():.6f}  "
                f"gnorm={gnorm.item():.4f}  "
                f"dt={dt:.0f}ms"
            )

    if rank == 0:
        print("[ppo] DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
