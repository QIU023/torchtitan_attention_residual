# Phase 9-B PPO Trace — Deferred (Future Work)

## Status: Deferred (re-scoped after user correction)

Originally planned as 50-step PPO infrastructure smoke for fabric
profile (multi-model NCCL pattern: actor + ref + reward + critic
sub-meshes + cross-mesh logprob exchange). Skipped due to setup
blocker, but the original blocker analysis was wrong.

## Original (over-scoped) blocker

torchtitan's `experiments/rl/simple_grpo_sum_digits.py` (one example
RL entry point) requires vLLM + monarch + torchstore. These are NOT
PPO/RLHF requirements — they are speed/scale optimizations:

- **vLLM** is a fast batched-inference server for the rollout phase
  (sample tokens from actor). Pure PyTorch `model.generate()` works,
  just slower.
- **monarch.actor** is for cross-mesh actor scheduling. Not needed
  for a single-host smoke.
- **torchstore** is for cross-mesh weight sync. Not needed when actor
  and ref share the same mesh / are loaded into the same process.

## Re-scoped path (vLLM-free smoke, ~4-6 h)

A minimal PPO/GRPO trace smoke that captures the unique fabric
pattern (multi-model fwd/bwd, KL exchange) without vLLM:

1. Load `v11_4d_*/checkpoint/step-5000` twice (actor + frozen ref),
   both wrapped in the same FSDP+PP mesh.
2. Mock reward (random scalar per sample, or "all-positive" sanity
   reward to avoid div/zero in advantage).
3. For each training step:
   a. **Slow rollout** via `actor.generate(max_new_tokens=64)` (no
      KV cache for simplicity; ~2-3× slower than training fwd).
   b. **Compute logprobs** under both actor and frozen ref → KL.
   c. **PPO loss** = `clip_ratio * advantage - kl_coef * KL`.
   d. Backward + optimizer step on actor only.
4. Tier_b NCCL trace at steps 30-50 → captures:
   - Actor training fwd/bwd (same as v11 pattern)
   - **Frozen ref forward** (FSDP allgather without RS — distinctive
     fabric signature for "inference-only" model)
   - **Cross-model exchange** (logits broadcast actor→ref or KL
     reduce across actor+ref ranks if they're in different meshes)

This adds **dual-model fwd + KL exchange** to the catalog, which is
the fabric signature unique to RLHF.

## Alternatives considered

| Path | Realistic time | Why skipped |
|---|---|---|
| OpenRLHF | 2-3 days | Same setup overhead; needs custom kimi_linear adapter |
| TRL (HuggingFace) | 1-2 days | Single-mesh focus, weak multi-mesh signal |
| Manual 2-model GRPO-lite (load two v11 ckpts, KL term, mock advantage) | 4-6 hours | Captures *partial* fabric pattern (dual-model fwd) but no rollout phase, not really PPO |

## Recommended path forward (when revisiting)

1. **Reserve 2-day window** with vLLM + monarch installed in advance
2. **Use Qwen3-0.6B (already supported)** for first PPO smoke — don't
   try to plug kimi_linear in until vanilla path works
3. Capture tier_b trace during steps 30–50 (post-warmup)
4. Run `phase7/extract_collectives.py` + flows + ixia to add PPO
   pattern to the fabric catalog

## Fabric pattern coverage WITHOUT 9-B

Without 9-B we still have:
- v11 4D pretrain (FSDP+PP+TP+EP)
- v12 4D pretrain (FSDP+PP+EP+dp_rep, no TP)
- SFT 4D (same mesh as v11, post-train fabric)

This covers **single-model 4D fabric** comprehensively (PP send/recv,
FSDP allgather/RS, EP all-to-all, optional TP allreduce). Missing:
- **Multi-model exchange** (actor↔ref logprob transfer)
- **Rollout phase** (inference-style burst with KV cache allgather)
- **RM-actor coupling** (reward injection into training step)

These are the unique additions PPO would bring.
