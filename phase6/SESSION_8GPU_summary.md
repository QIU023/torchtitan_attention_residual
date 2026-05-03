# 8-GPU session summary (Phase 6 + Phase 7) — 2026-05-03

This file records what landed during the 8-GPU rental box session
(2026-05-03 onward), distinct from the prior 4-GPU work captured in
`SESSION_SUMMARY_zh.md`.

## Setup
- Hardware: 8× RTX 5090 PCIe (32 GiB / card, Blackwell sm_120).
- torch 2.11.0+cu130, fla-core 0.5.0, transformers + sentencepiece (added).
- HF cache at `/workspace/.hf_home/` (LLaVA-Pretrain 558K + 660 image dirs,
  SigLIP-base, Llama-3.1 tokenizer; 30 C4 shards prefetched).
- `phase4/runs/.../step-8000` ckpt scp'd from 4-GPU box (15 GB).
- `torchtitan/assets/hf/Llama-3.1-8B` symlinked to the workspace tokenizer dir.
- 124 CPU tests still pass (27 phase5 + 97 torchtitan).

## Code changes landed in `torchtitan/torchtitan/experiments/kimi_linear/parallelize.py`

| Function | What it adds |
|---|---|
| `apply_tp_kimi_linear` | Minimum-viable Tensor Parallel: dense MLP gate/up `ColwiseParallel` + down `RowwiseParallel`. KDA / MLA / AttnRes / embed / lm_head left replicated (KDA fla-core kernels not validated for sharded heads; MLA's asymmetric Q/K/V dim split is fragile; `AttnResProjection`'s out_features=1 cannot be sharded; embed/lm_head SP+loss-parallel deferred). Fires one all-reduce per dense-MLP forward. |
| `apply_ep_kimi_linear` | Expert Parallel: applies `ExpertParallel()` from `torchtitan.distributed.expert_parallel` to every MoE layer's `ffn._moe.experts` ModuleList. Fires all-to-all on the EP mesh for token dispatch + combine. Works on the existing `kimi_linear_436m_block_attn_res_n4` flavor (which already has `first_k_dense_replace=1` so layer 0 is dense and 1+ are MoE). |
| CP path | Replaces the prior unconditional `NotImplementedError` with a documented one explaining the fla-core blocker: KDA's `chunk_kda` triton kernel is a causal recurrence over seq dim, and CP shards seq dim → would need ring-recurrence in fla-core upstream. Until that lands, `context_parallel_degree > 1` raises a clear error. |
| `parallelize_kimi_linear` | Now applies (when enabled) TP → EP → compile → FSDP in that order. Old code raised `NotImplementedError` for both TP and CP. |

## 8-GPU launcher infrastructure

| File | Role |
|---|---|
| `phase6/launch_8gpu_mm.sh` | Generic mesh + recipe + tier driver. Required env: `OUT_DIR FSDP PP TP CP EP STEPS LOCAL_BS GLOBAL_BS`. Optional: `V ADAPTER FLAVOR STUDENT_CKPT SEED DETERMINISTIC COMPILE TRACE_TIER TRACE_STEPS`. Dumps `recipe.json` per run + opens NCCL trace if `TRACE_TIER` set. Used by everything below. |
| `phase6/run_remaining_8gpu.sh` | Phase 6 alignment matrix orchestrator — runs B0 → A2 → A3 → A6 → P7C4 → P7C5, then chains alignment reports → trace extraction → tier B → tier A → catalog gen. Writes to `phase6/orchestrator_8gpu.log`. |
| `phase6/run_v10_pretrain.sh` | 5000-step continued multimodal pretrain (GBS=120 LBS=15 FSDP=8) from step-8000. Optional; user starts manually after alignment matrix. |
| `phase6/run_alignment_reports.sh` | Wraps `phase5/compare_pp_vs_fsdp.py` for each config; emits `phase6/alignment_8gpu_<cfg>.{txt,csv,png}`. Resolves the timestamped TB subdir automatically. |
| `phase6/check_8gpu_status.sh` | One-shot status snapshot (running procs, GPU state, last-step + ERROR markers per run, orchestrator tail). |
| `phase7/extract_collectives.py` | Parses `NCCL_DEBUG=INFO COLL` logs into a structured CSV (`opname, count, dtype, bytes, size_bucket, nranks, root`) per trace dir. Verified on B0: 232k+ rows, FSDP all-gather/reduce-scatter dominate. |
| `phase7/run_tier_b_a_traces.sh` | Tier B (GBS=120, 50 steps) + Tier A (GBS=384, 100 steps) trace recording across 6 configs. ~12 h wallclock total. |
| `phase7/build_pattern_catalog.py` | Walks all `phase5/runs/8gpu_*/tier_{a,b,c}_trace/collective_summary.csv`, emits `phase7/pattern_catalog.md` with replay-priority table + per-config tier histograms + cross-config Tier A comparison. |

## Recipe consistency rule (locked-in)

All alignment configs use **GBS = 16, LBS = 1** (B0 baseline uses LBS = 2
on FSDP=8 to keep GBS = 16). This is the cross-mesh minimum: the largest
3D mesh (FSDP=2 × PP=4 V=2) needs GBS ≥ FSDP × V × PP = 16. Going lower
(e.g. GBS=8 or 12) underflows the Interleaved1F1B microbatch count and
produces the `loss = -log(vocab) ≈ -12` constant-output sentinel — we
hit this on the first attempt before fixing.

| Tier | GBS | LBS / config |
|---|---|---|
| Alignment | 16 | B0=2; everyone else=1 |
| Tier B | 120 | B0=15; PP-bearing configs=5 (8 mb / dp rank); P7C5=15 |
| Tier A | 384 | All=8 (universal microbatch slack) |

## Alignment matrix status

| Config | Mesh | Status | Final step / loss |
|---|---|---|---|
| B0 anchor | FSDP=8 (LBS=2) | ✅ done | step 500 / 3.523 |
| A2 | FSDP=2 × PP=4 V=2 + adapter | ⏳ in flight | — |
| A3 | FSDP=2 × PP=2 × TP=2 V=2 + adapter | queued | — |
| A6 | FSDP=2 × PP=2 × EP=2 V=2 + adapter | queued | — |
| P7C4 | TP=2 × PP=2 × EP=2 V=2 (FSDP=1) | queued | — |
| P7C5 | FSDP=2 × TP=2 × EP=2 (PP=1) | queued | — |
| CP=2 | — | out of scope (fla-core blocker) | n/a |

Pass criterion: max\|Δ\| ≤ 0.13 nats vs B0 over matched steps (Phase 3
established noise band, applied in Phase 6 A1 headline result).

## Trace tiers status

| Tier | Status |
|---|---|
| C (alignment-slice, free) | B0 done (232k collective rows). Others auto-collected as alignment runs hit step 500. |
| B | queued — runs after alignment matrix completes |
| A | queued — runs after Tier B completes |
| pattern_catalog.md | regenerated automatically after each tier batch |

## What's NOT in this session

- v10 multimodal pretrain (script ready at `phase6/run_v10_pretrain.sh`,
  user launches manually after alignment matrix done; ~3.5 h on FSDP=8).
- KDA-side TP shard (out of scope; fla-core kernel work needed upstream).
- KDA-side CP ring-recurrence (out of scope; same upstream dependency).
- MLA TP plan (deferred; asymmetric head_dim split makes a single
  ColwiseParallel registration risky without per-call shape audit).
- Embed / lm_head SP + loss-parallel (deferred; current FSDP=8 baseline
  doesn't need it for alignment claim).

## Path to upstream PR-merge readiness

After alignment matrix passes (max\|Δ\| ≤ 0.13 nats for all configs)
and tier A traces are collected, the deliverables for the eventual
upstream PR are:

1. `parallelize.py` with TP + EP plans (this session) + the existing
   FSDP plan (from prior phases).
2. Verified-config matrix in `PR_DRAFT.md` extended with the 5 new
   8-GPU 3D entries (B0, A2, A3, A6, P7C4, P7C5).
3. `phase7/pattern_catalog.md` as documentation of which collectives
   fire under which config at production tensor sizes.
4. Resolution-or-documentation of the fla-core CP dependency (this
   session: documented via `NotImplementedError`).
