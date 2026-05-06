# phase5/runs/ — categorized index

Top-level `phase5/runs/` flat layout dates back to early Phase 5
when there were only a handful of runs. It has accumulated 50+
sub-directories across pretrain (Phase 5), alignment (Phase 6),
production training (v10/v11/v12), SFT (Phase 9-A), and Phase 10
inference + RLHF work.

Re-arranging the directories themselves would break hardcoded paths
in many `phase{6,9,10}/run_*.sh` and committed `ixia_config.json`
files. Instead, this README catalogs the existing layout by
era / purpose so the structure becomes navigable without moves.

## Active production checkpoints (DO NOT MOVE)

These are referenced by current launchers + Phase 10 work:

| Directory | What | Source |
|---|---|---|
| `v11_4d_fsdp2_pp2_tp2_ep2_continue_8gpu_from_p4_step8000/` | v11 4D pretrain (FSDP=2×PP=2×TP=2×EP=2), 5000 steps | `phase6/run_v11_pretrain.sh` |
| `v12_4d_fsdp2_dp2_pp2_ep2_continue_8gpu_from_p4_step8000/` | v12 4D EP-replace-TP, 5000 steps | `phase6/run_v12_pretrain.sh` |
| `sft_v11_llava_instruct_150k_4d/` | SFT step-490 on LLaVA-Instruct from v11 | `phase9/run_sft_pretrain.sh` |

## Phase 7 / 8 / 9 fabric profiling

| Directory | What |
|---|---|
| `5d_mode_b_llama3_pp_fsdp_cp/` | 5D MODE=B trace (llama3 PP+FSDP+CP) |
| `v12_trace_50step/` | v12 trace post-hoc 50-step capture |
| `ppo_smoke_no_vllm/` | Phase 9-B toy PPO cross-mesh KL smoke |

## Phase 10 inference + RLHF

| Directory | What |
|---|---|
| `inference_torchtitan_phase4_step8000/` | Stage D 4D inference baseline (FSDP=4×TP=2×EP=2) |
| `inference_autoregressive_growing/` | Stage J no-cache growing-prefix autoregressive |
| `inference_autoregressive_single_token/` | Stage J single-token blocked by KDA Triton (run.log only) |
| `inference_two_phase_real/` | Stage K real-model two-phase RS+AG injection |
| `ppo_real_torchtitan/` | Stage F real PPO smoke (kimi_linear actor + ref) |
| `two_phase_tp_allreduce/` | Stage I synthetic TP AllReduce baseline |
| `two_phase_tp_rs_ag/` | Stage I synthetic TP RS+AG two-phase |
| `workload_short_high_bs/` | Stage L sustained workload BS=16 seq=256 |
| `workload_mid/` | Stage L sustained BS=4 seq=1024 |
| `workload_long/` | Stage L sustained BS=2 seq=4096 |
| `workload_prod/` | Stage L sustained BS=8 seq=2048 |

## Earlier alignment experiments (Phase 6 era)

`8gpu_a2_*`, `8gpu_a3_*`, `8gpu_b0_*` — alignment sweeps with
small-batch GBS=16 at fixed mesh shapes.

`align_*`, `arm1_*`, `arm2_*`, `a2_*`, `a3_*`, `a4_*`, `a5_*`,
`a6_*` — Phase 6 alignment-arm experiments. Tier_a/b/c trace
directories under most of these are referenced by
`phase7/FINAL_CATALOG.md`.

## Earlier pretrain experiments (Phase 5 era, mostly superseded)

`v3_*`, `v4_*`, `v5_*`, `v6_*`, `v7_*` — pre-PP era pretrain
sweep. v7 was the BEST checkpoint at the time.

`v8_pretrain_resilient_*` — added crash-resilient retry-loop scaffolding.

`v9_continue_from_v8_step10000/` — v9 resume that hit the cublas
grouped_mm device-side assert (documented in
`phase6/summaries_archive/v9_final_summary.md`).

`v10_4d_*`, `v10_fsdp2_dp2_*`, `v10_fsdp4_pp2_*` — v10 ablations,
explored several mesh shapes for 4D. v11 superseded these.

`overnight_mm_pretrain_*` — early multimodal SFT sweeps before the
4D mesh stabilized.

## Diagnostic / one-off

`a4_async_dcp_smoke/`, `a5_mid_save_resume_smoke/`,
`a5_redo_resume_smoke/` — checkpoint-system smoke runs.

`arm1prime_fsdp_seed42_from_p4_8k/` — Arm 1 deterministic
re-run seed.

## Format conventions across runs

Each run directory typically contains:
- `train.log` (or `run.log` for inference)
- `recipe.json` (frozen hyperparams)
- `tb/<timestamp>/events.out.tfevents.*` (TensorBoard, sometimes
  multiple if resumed)
- `checkpoint/step-N/__*.distcp` + `.metadata` (DCP sharded)
- `tier_{a,b,c}_trace/` with NCCL collective traces (if profiled)

Phase 10 runs follow the same convention but with `run.log` instead
of `train.log` and only `tier_b_trace/` (since pure inference doesn't
need multi-tier).
