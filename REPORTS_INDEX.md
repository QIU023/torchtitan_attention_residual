# Reports + Deliverables Index

A flat list of what's documented where, for quick navigation.

## Cross-phase summaries

| File | Scope |
| --- | --- |
| `PHASE9_10_11_SUMMARY.md` | 5-day comprehensive summary: SFT, SGLang foundation, AttnRes optimization, RLHF framework |
| `PROJECT_STATUS.md` | (older) per-phase status table — pre-overnight |

## Phase 11 deliverable docs

| File | Scope |
| --- | --- |
| `phase11/SGLANG_ATTNRES_INFERENCE_SUMMARY.md` | **Standalone PR-targeted summary** of SGLang AttnRes inference optimization |
| `phase11/PHASE11_SGLANG_REPORT.md` | Original phase 11 report (pre-overnight) |
| `phase11/SGLANG_ATTNRES_AUDIT.md` | Item-by-item audit of every blog claim + closure status |
| `phase11/PROFILING_REPORT.md` | Kineto profile + corrected v3 bench (kernel firing) |
| `phase11/B5_ATTNRES_INFERENCE_KV_CACHE.md` | Design note: AttnRes block reps are intra-forward state |
| `phase11/TORCHTITAN_VAST_AI_PATCHES.md` | Env-compat patch rationale (torch 2.9 stable) |
| `phase11/torchtitan_vast_ai_env_compat.patch` | Applicable diff (5 files, 239 lines) |
| `phase11/rlhf/README.md` | RLHF framework README + status board |

## Upstream PR / RFC drafts

| File | Target |
| --- | --- |
| `torchtitan/torchtitan/experiments/rl/RFC_SGLANG_GENERATOR.md` | Engine-agnostic Generator RFC for upstream torchtitan |
| (sglang fork branch `attention_residual_inference`) | 3 commits PR-ready for sglang upstream |

## Code (PR-ready)

* `sglang/python/sglang/srt/layers/attn_res.py` — algorithm + Triton kernel
* `sglang/python/sglang/srt/models/{attn_res,qwen3_attn_res}_overlay.py` — carriers
* `torchtitan/torchtitan/experiments/rl/actors/sglang_generator.py` — RL Generator
* `torchtitan/torchtitan/experiments/rl/models/sglang_wrapper.py` — model glue
* `phase11/rlhf/run_grpo_{sum_digits,llava_caption}.py` — entry points
* `phase11/profile_attn_res.py` — kineto profiler harness
* `phase11/bench_attn_res.py` — 4-mode bench harness

## Trained checkpoints (kept)

| Path | Stage | Size |
| --- | --- | --- |
| `phase4/runs/kimi_447m_aligned_block_attn_res_fsdp_paperhparams/checkpoint/step-12500` | LM pretrain final | 17 GB DCP |
| `phase11/hf_aligned_447m_step12500/` | Same, HF safetensors | 3 GB |
| `phase5/runs/v_fsdp8_447m_aligned_continue_from_step12500/checkpoint/step-2500` | Multimodal continued pretrain | 17 GB DCP |
| `phase5/runs/sft_v_fsdp8_447m_llava_pretrain/checkpoint/step-500` | SFT final | 33 GB DCP |

## Bench / trace artifacts

| Path | Content |
| --- | --- |
| `phase11/bench_results_v3_kernel_actually_on/` | Bench v3 with Triton kernel firing — final numbers |
| `phase11/profile_results/` | Kineto traces + JSON summary (kernel-by-kernel) |
| `phase11/trace_kimi_tp8_shard{0,1}/` | NCCL trace TP=8 with/without seq-shard |
| `phase11/trace_kimi_3d_shard{0,1}/` | Same on 3D mesh (TP×PP×EP) |
| `phase11/rlhf/trace_grpo_sum_digits/` | 50-step GRPO production trace (40K NCCL ops) |
| `phase11/rlhf/trace_grpo_sum_digits_200steps/` | 200-step GRPO trace (44K NCCL ops) |
| `phase11/rlhf/trace_ppo_sum_digits/` | 50-step PPO with KL — 2× Send/Recv vs GRPO |
| `phase5/runs/*/tb/` | TensorBoard logs across all training runs |
