# v9 final summary — continued pretrain from v8/step-10000

## Headline

**Final loss 1.81 nats at v9/step-5000** — new session record, improved on v8's best 1.90 mid-run. The crash-resilient infrastructure (commit 57a4b47 projector save/load + commit fa1081d orchestrator) again survived 3 KDA Triton CUDA assert + 1 cuBLAS internal error and auto-resumed cleanly to completion.

## Run details

| Metric | Value |
|---|---|
| Init point | v8/step-10000 (loss 2.07) — model_only |
| Final ckpt | v9/step-5000 (loss 1.81), retained as session champion |
| Total wallclock | ~6.8 hours |
| Total grad steps | 5000 |
| Total samples | 5000 × GBS=120 = **600K image-text pairs** ≈ 1.07× LLaVA-Pretrain epoch |
| GPU config | LOCAL_BS=30 GBS=120 SEQ_LEN=260, 89.6% memory |
| Throughput | ~1605 tps/rank, 6420 tps total, MFU 3.13% |

## Loss trajectory

| step | loss | notes |
|---|---|---|
| 1 | 4.70 | projector reset (model_only init from v8/step-10000) |
| 50 | 3.16 | recovered to 3-tier within 50 steps |
| 200 | 2.70 | first save with mm_projector entry |
| 500 | 2.40 | crossed 2.5 |
| 1000 | 2.46 | iter 1 ended with cuBLAS internal error |
| 2000 | 2.21 | iter 2 (auto-resumed); crossed below v8's 2.30 baseline |
| 3000 | 2.17 | iter 3 (auto-resumed after KDA crash) |
| 3800 | 2.13 | live midpoint |
| 5000 | **1.81** | final, clean exit |

## Crash recovery

| Iter | Span | End reason |
|---|---|---|
| 1 | step 1 → ~1100 | cuBLAS internal error |
| 2 | ~1100 → ~2200 | KDA Triton CUDA assert |
| 3 | ~2200 → ~4400 | KDA Triton CUDA assert |
| 4 | ~4400 → 5000 | clean exit (Training completed) |

3 crashes total, all recovered with full state preservation (LM weights + projector + AdamW state for both + LR scheduler + dataloader). Loss curve continued to descend monotonically through every restart — no projector reset penalty.

## Comparison with v8

| Run | Init | Steps | Best loss | Final loss |
|---|---|---|---|---|
| v8 | v7/step-800 (model_only) | 10000 | 1.90 (mid-run) | 2.07 |
| **v9** | v8/step-10000 (model_only) | 5000 | **1.81 (final)** | **1.81** |

v9 with half the step budget produced a better final than v8's best mid-run reading. Two factors:
1. **Better init**: v9 starts from a much better-trained LM than v8's start point
2. **Projector mm_projector save**: v9 saved its trained projector at every save, so the per-iter projector reset only happened at iter 1 (not at every crash recovery as in v8)

## Cumulative Phase 6 multimodal pretrain progress

Across the v1 → v9 chain on 4×RTX 5090, with various BS / seed / init combinations:

| Milestone | Value | Source |
|---|---|---|
| Phase 4 LM-only baseline | C4 val 3.23 | Phase 4 step-8000 |
| First multimodal endpoint | caption loss 3.03 | arm1prime/step-4000 (GBS=32, no seed) |
| First high-BS endpoint | 2.07 | v8/step-10000 (GBS=120, seed=42, projector saved) |
| **Session best** | **1.81** | **v9/step-5000** |

Total improvement: Phase 4 LM-only val 3.23 → multimodal caption 1.81 = **1.42 nats improvement** over the LM-only starting point on caption task. (Note: LM val and caption are different objectives; the comparison is qualitative.)

## Final retained ckpts

| Path | Loss | Purpose |
|---|---|---|
| `phase4/.../checkpoint/step-8000` | C4 val 3.23 | LM-only Phase 4 anchor |
| `phase5/runs/arm1prime_fsdp_seed42_from_p4_8k/checkpoint/step-4000` | 3.03 | original GBS=32 caption story |
| `phase5/runs/v8_pretrain_resilient_from_v7_step800/checkpoint/step-10000` | 2.07 | v8 final, mm_projector |
| **`phase5/runs/v9_continue_from_v8_step10000/checkpoint/step-5000`** | **1.81** | **session champion**, mm_projector |

Total disk: ~56 GB across 4 keepers.
