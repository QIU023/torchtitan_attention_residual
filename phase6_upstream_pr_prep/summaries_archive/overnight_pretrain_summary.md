# Overnight multimodal pretrain — 2026-04-30 → 2026-05-01 session

The 18h GPU window was used for two things in sequence:

1. **Phase 6 closure work** (~3h): A1.1 fix verified, A1 alignment passed
   (median \|Δ\|=0.024 nats), B1 variable image count + 6 unit tests, B4
   sentinel registry + 9 unit tests + train_mm wire-up, C1 cache adapter
   ablation report (naive vs adapter PASS by max 0.092), C2 mixed-dtype
   scatter test, A4 async DCP smoke (\|Δ\|=0.004 vs sync), A6 partial
   FSDP=2 PP=2 alignment (\|Δ\|=0.006 vs FSDP=1 PP=4), parallelize.py
   None-filter fix (submodule commit 92ad381), PR_DRAFT.md.

2. **Real multimodal pretraining** (~12h GPU): pushed `LOCAL_BS` to fit
   ~89% of the 32 GiB RTX 5090 (target was 85-90%), starting from
   Phase 4 step-8000 LM-only init with fresh projector. Training hit
   `fla-core` KDA gate `Triton CUDA: device-side assert triggered` 4
   times across ~6500 wallclock steps — an intermittent kernel-level
   numerical edge case in the upstream KDA triton kernel that we
   document but do not own. Each crash burned ~50-100 steps of
   projector re-alignment work after `--checkpoint.initial_load_model_only`
   resume; preserving the projector across resumes is a follow-up
   (B5-adjacent: extend the trainer's checkpointer state_dict to
   include `self.projector` and `proj_optim`).

## Loss curve across the chain

| Run | Init | LOCAL_BS | GBS | Mem | Steps survived | Best loss |
|---|---|---|---|---|---|---|
| Arm 1 (older, no-seed) | Phase 4 step-8000 | 8 | 32 | 36% | 2800 (killed for seed) | 3.07 |
| Arm 1' v2 (alignment baseline) | Phase 4 step-8000 | 3 | 12 | 26% | 2000 ✓ | 3.71 |
| arm1prime caption story | Phase 4 step-8000 | 8 | 32 | 36% | 4000 ✓ | 3.03 |
| v1 part 1 | arm1prime step-4000 | 16 | 64 | 56% | 2650 (KDA crash) | 2.86 |
| v1 part 2 | v1 step-2000 (full state) | 16 | 64 | 56% | 4700 (KDA crash) | 2.86 |
| v3 | Phase 4 step-8000 (model-only) | 32 | 128 | 93% | 700 (OOM) | n/a |
| v4 | v3 step-500 (model-only) | 30 | 120 | 90% | 1400 (KDA crash) | 3.10 |
| v5 | v4 step-1000 (model-only) | 30 | 120 | 90% | 1450 (KDA crash) | 3.09 |
| v6 | v5 step-1400 (model-only) | 30 | 120 | 90% | 1450 (KDA crash) | **2.84** |
| **v7** | **v6 step-1200 (model-only)** | **30** | **120** | **90%** | **800 ✓ (clean exit)** | **2.79** |

## Headline results

* **Final session best loss: 2.790 nats** at v7/step-800 (commit
  `phase5_vlm_multimodal_sft/runs/v7_pretrain_bs120_from_v6_step1200_BEST/checkpoint/step-800`).
  This crosses below the README's *Stretch* tier ≤ 2.8 — we previously
  reached only the *Acceptable* tier ≤ 3.2 / *Target* ≤ 3.0.
* **Cumulative training:** ~5500 effective multimodal steps at GBS=120
  ≈ 660K image-text pairs through the model (~1.2 epochs of
  LLaVA-Pretrain-558K).
* **Throughput at LOCAL_BS=30 GBS=120:** 1700 tps/rank, ~6800 tps total,
  ~4.5 sec/step. MFU 3.3%, 4× the LOCAL_BS=8 baseline (0.78%).
  Fundamental ceiling is still the unoptimized fla-core KDA kernel on
  Blackwell sm_120.
* **Memory utilization at LOCAL_BS=30:** 28.12 GiB / 32 GiB ≈ 89.7%,
  hitting the user's 85-90% target. LOCAL_BS=32 GBS=128 was tried and
  OOM'd at the activation peak during backward — 30 is the safe ceiling.

## What did not work

* **LOCAL_BS=32 GBS=128 (v3):** OOM at backward activation peak. 93%
  steady-state memory left only ~2 GiB headroom; insufficient for
  short-lived peak allocations. 30 is the practical maximum for this
  config on a 32 GiB card.
* **A5 mid-save resume smoke:** orchestrator's grep timing race fired
  SIGTERM before training compile finished; phase 2b ran 50 wasted
  steps from random init. Documented as known follow-up.
* **Continuous training without crashes:** 4 KDA Triton CUDA
  device-side asserts in fla-core's gate kernel across ~6500 wallclock
  steps. Pattern: data-dependent (specific image triggers), independent
  of seed/BS within tested range. Not torchtitan's bug; upstream
  fla-core does not have a Blackwell-tuned variant.

## Artifacts kept

* `phase5_vlm_multimodal_sft/runs/v7_pretrain_bs120_from_v6_step1200_BEST/checkpoint/step-800`
  — best ckpt this session, loss 2.790.
* `phase5_vlm_multimodal_sft/runs/v7_pretrain_bs120_from_v6_step1200_BEST/tb/` — full
  training log including the v7 final-pass curve.
* `phase5_vlm_multimodal_sft/runs/v6_pretrain_bs120_from_v5_step1400/checkpoint/step-1200`
  — second-best, loss 2.839, lineage anchor.
* `phase5_vlm_multimodal_sft/runs/arm1prime_fsdp_seed42_from_p4_8k/checkpoint/step-4000`
  — original GBS=32 caption-story endpoint, loss 3.03.

All other intermediate ckpts dropped during the session to keep disk
under control (peak ~76% before manual prune; final ~69%).

## Phase 6 status board updated

A1, A1.1, A4, A6 partial, B1, B4, C1, C2 partial all complete + pushed.
A5 needs redo. PR_DRAFT.md updated with verified-config matrix and
the post-fix alignment numbers. Sentinel registry wired into train_mm.py.
The 8-GPU 3D parallelism roadmap (A2, A3, EP, CP) is still pending the
rented box.
