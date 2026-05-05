# Phase 9-B PPO Trace — Deferred (Future Work)

## Status: Deferred

Originally planned as 50-step PPO infrastructure smoke for fabric
profile (multi-model NCCL pattern: actor + ref + reward + critic
sub-meshes + cross-mesh logprob exchange). Skipped due to setup
blocker.

## Setup blocker

torchtitan's `experiments/rl/simple_grpo_sum_digits.py` (the cleanest
RL entry point in this repo) requires:

- **vLLM** for actor rollout (not installed; `pip install vllm` ~2 GB
  + dep conflicts with our PyTorch 2.11 nightly possible)
- **monarch.actor** (Meta's distributed actor framework, beta) — not
  installed
- **torchstore** for cross-mesh weight sync — not installed
- **Qwen3-0.6B** base ckpt — needs HF download (~1 GB) and
  ``hf_assets_path`` config

Realistic setup + first-run debug = **1-2 days**. Doesn't fit the
remaining 18h budget after phase 9-A SFT + phase 8 qual eval.

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
