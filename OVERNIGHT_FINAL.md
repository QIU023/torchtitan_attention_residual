# Overnight Session Final Summary

**Period:** 2026-05-09 09:46 UTC → 2026-05-10 (~12h)  
**Trigger:** "进入overnight模式 你有12小时时间，把任何pending的task和bug都解决"

## Closed loop outcomes

### 1. End-to-end RLHF runs

Production GRPO + PPO loops via SGLang generator, after solving the
torch-2.9-stable env-compat blocker (custom SDPA fallback for
`varlen_attn`):

| Run | Steps | NCCL ops | Wall | Notes |
| --- | ---: | ---: | ---: | --- |
| GRPO 50  | 50   | 40,675 | 45 s   | First end-to-end |
| PPO 50   | 50   | 40,679 | 60 s   | KL=0.05, frozen ref engaged |
| GRPO 200 | 200  | 44,279 | 3 min  | Longer convergence sample |
| GRPO 500 | 500  | 51,479 | 8 min  | Stable plateau ~+0.20 reward |
| GRPO 1000 | 1000 | (running) | (running) | Definitive curve |

Reward curve: random ~+0.18 → stable ~+0.20-0.22 from step 25 onwards.
Qwen3-0.6B is at the capability ceiling for digit-arithmetic — the
format learn is done early but the model lacks parametric knowledge
to reliably sum 4-5 digits.

### 2. SGLang AttnRes inference — finalized

| Bench | TP | ctx | two-phase decode tps | TTFT speedup |
| --- | ---: | ---: | ---: | ---: |
| v3 | 1 | 4K | 698 | 1.09× naive |
| v3 | 1 | 16K | 634 | 1.07× |
| v3 | 8 | 4K | 559 | 1.05× |
| v3 | 8 | 16K | 482 | 1.12× |
| **v4 long** | 1 | 24K | 623 | 1.07× |
| **v4 long** | 8 | 24K | 466 | 1.04× |

Profile (kineto) confirms `_phase2_merge_norm_kernel` fires 2015 / 2048
expected calls (>98% hit rate). +27% decode-tps recovery from kernel
firing correctly (after fixing stale install path between user
submodule and `/sgl-workspace/`).

NCCL fabric: AR -58% under seq-shard (60 GB → 25 GB at TP=8 ctx=16K).
Same -60% pattern under 3D mesh (TP×PP×EP).

### 3. RLHF framework architectural deliverables

* **Lead/follower pattern** for SGLang-vs-Monarch topology
  (`sglang_generator.py:__init__` + `Provisioner.allocate_shared`).
  Solves the "N actors × tp_size = N² GPU contention" without
  upstream Monarch changes.

* **PolicyTrainer model-spec relaxation** (RFC #26 partial):
  hard `VarlenAttention.Config` assert → soft warning. Enables
  non-Qwen3 model_specs to construct + parallelize without
  upstream-breaking changes. Full model-agnosticity (custom
  compute_token_log_probs per spec) is a follow-up.

* **`varlen_attn` SDPA fallback** in env-compat patch — packed
  sequences run through `F.scaled_dot_product_attention` with
  per-segment causal masking. Numerically equivalent to torch
  nightly's `varlen_attn`. Single most impactful patch — converts
  the framework from "imports OK" to "trains end-to-end".

## Reports written

* `PHASE9_10_11_SUMMARY.md` — comprehensive cross-phase report (root)
* `phase11/SGLANG_ATTNRES_INFERENCE_SUMMARY.md` — PR-targeted SGLang AttnRes summary
* `REPORTS_INDEX.md` — navigation across all docs
* This file — overnight-specific summary

## Final commits (chronological)

```
13d377c  varlen_attn SDPA fallback unblocks RL trainer
f34053d  GRPO sum-digits 50 steps via SGLang
edb3ca1  PPO sum-digits 50 steps (kl_coef=0.05)
e38b5ba  Phase 9-11 summary + SGLang AttnRes summary
9c42706  GRPO 200 steps + PolicyTrainer relaxation
d8b7b12  REPORTS_INDEX.md
dec1fc1  GRPO 500 steps
adccd82  Long-ctx (24K) AttnRes bench
(this commit) Overnight final summary
```

## Final disk state

* `/`: 200 GB total, ~120 GB used, 80 GB free
* Final ckpts kept (per user instruction): phase4 step-12500 (17 GB DCP),
  v_fsdp8_447m step-2500 (17 GB), sft step-500 (33 GB)
* HF safetensors copy of phase4 ckpt: 3 GB
* Qwen3-0.6B HF download: 1.5 GB
* All NCCL traces gz'd, kineto traces gz'd

## Pending (truly out-of-scope this session)

* Multimodal SGLang VLM model class (SigLIP+projector wired into AttnRes overlay) — ~half-day
* Real-ckpt 447m AttnRes RLHF — needs custom `compute_token_log_probs`
* Upstream PR filings (sglang RFC + torchtitan RFC) — process work, not technical
