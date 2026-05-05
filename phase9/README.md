# Phase 9 — Post-training (SFT + PPO trace)

Two scope-distinct subgoals, both on top of v11 step-5000 (4D pretrain
ckpt) which already has projector aligned via LLaVA-Pretrain.

## Subphase 9-A: SFT on LLaVA-Instruct-150K

* **Goal**: visual instruction tuning — turn v11 from "vision-aligned
  next-token predictor" into "vision-aware instruction follower"
* **Data**: `liuhaotian/llava-instruct-150k` (single-split JSON, 150K
  multi-turn QA on COCO 2017 train images)
* **Mesh**: same 4D as v11 (FSDP=2 PP=2 TP=2 EP=2) — preserves
  pretraining infrastructure investment
* **Hyperparams**:
    * LR = 2e-5 (10× v11's 1e-5; SFT standard)
    * 1 epoch ≈ 1200 steps at GBS=128 (LLaVA-1.5 recipe)
    * micro=20 LBS=200 same as v11 (memory profile known good)
* **Trace**: TRACE_TIER=tier_b 50 steps captured for phase 7 ext
* **Cost**: ~3-4h on 8×RTX 5090 PCIe at v11's TPS=2400

## Subphase 9-B: PPO smoke (infra-only, traffic profiling)

* **Goal**: capture multi-model RLHF infrastructure NCCL traffic for
  fabric profiling. Traffic patterns (multi-mesh, rollout vs update,
  cross-model logprob exchange) are the deliverable, NOT trained
  policy quality.
* **Setup**: OpenRLHF (PyTorch-native, easiest 4-model PPO infra)
    * Actor (= v11 step-5000 + SFT'd from 9-A)
    * Reference frozen (= same checkpoint)
    * Reward Model: **mock** (random reward) — enough to drive PPO
      loss term without needing a real preference-trained RM
    * Critic: shared with actor or a small dedicated (TBD)
* **Mesh**: 4D for actor (PP=2 FSDP=2 TP=2 EP=2), 1D each for ref/RM
* **Run**: 50 PPO steps + tier_b trace
* **Cost**: ~30-60 min infra setup + 15 min trace = 1-2h

## Why SFT first then PPO

* SFT model = standard PPO reference for KL constraint
* SFT-tuned actor produces less garbage rollouts → PPO rollout phase
  has reasonable token distributions (more representative trace)
* For trace-only purpose SFT could be skipped; we keep it because
  the user wants real post-training quality alongside trace

## Phase 7 ext (free byproduct)

Both 9-A and 9-B traces feed into `phase7/extract_collectives.py +
expand_to_flows.py + flows_to_ixia.py` to produce post-training fabric
patterns alongside the existing pretraining patterns from v11/v12.

## Files (to be written)

```
phase9/run_sft_pretrain.sh        # SFT entry, modeled on run_v11
phase9/multimodal_sft_dataset.py  # LLaVA-Instruct conversational format
phase9/run_ppo_smoke.sh           # OpenRLHF launcher
phase9/openrlhf_config.yaml       # 4-model + mesh setup
```
