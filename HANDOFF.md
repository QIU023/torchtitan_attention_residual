# seq-KD Pipeline — Handoff (this box abandoned, continue on H200)

**Date:** 2026-05-30. This container became unreliable (12h offline); distillation is
being restarted on a separate H200. All code is on GitHub; this doc captures the
non-git knowledge (specs, hyperparameters, eval targets, results, backup paths).

---

## 1. Pipeline (4-stage, where we are)

1. ~~Phase A (mix665k epoch-2 SFT)~~ — **cut**. Live A/B showed it was redundant
   (MMBench 36.4→36.8 = noise; flat training loss from step 1 = no new signal).
2. ✅ **seq-KD data** — Qwen3-VL-30B-A3B teacher rewrote all assistant turns of the
   full 665,298-sample mix665k. Offline (sequence-level KD; teacher NOT in the student
   training loop).
3. 🔄 **Student SFT (S5)** — 447M Kimi-AttnRes on the distilled data. On this box reached
   **step 3450/5200 (66% of 1 epoch)** then abandoned. **Partial eval below is NEGATIVE.**
4. ⏳ eval → token-OPD (Mantis) → GRPO.

---

## 2. seq-KD hyperparameters

### Teacher generation (offline)
| param | value |
|---|---|
| teacher | `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` (4-bit AWQ) |
| serving | vLLM, **single-GPU per replica (TP=1, NO tensor-parallel)**, 8 data-parallel replicas |
| temperature | 0 (greedy) |
| max_new_tokens | 512 |
| `max_pixels` | 1,003,520 (≈1280 vision tokens — cap to avoid >max_model_len crash) |
| method | regenerate each `gpt` turn conditioned on the ORIGINAL preceding dialogue |
| code | `phase11_rlhf_grpo_infra/seq_kd/{gen_worker.py, run_seqkd_gen.sh}` |

### Student SFT (S5)
| param | value |
|---|---|
| init | stage2 step-5200 (= pre-seq-KD VLM base, MMBench 36.4) |
| lr | **2e-5** |
| seq_len | 1024 on this box (dropped from 1536 due to sm_120 conv crash — **use 1536 on H200**, see §5) |
| gbs / lbs / grad_accum | 128 / 8 / 2 |
| steps | 5200 (= 1 epoch over 665,298) |
| warmup | 156 steps; cosine, decay_ratio 0.2, min_lr 0 |
| weight_decay / max_norm | 0.0 / 1.0 |
| loss | **gpt-only CE** (mask prompt; assistant tokens only). NO KL/temperature — that is the point of seq-KD (tokenizer-agnostic). |
| AC / parallel | full activation checkpoint / 1D FSDP=8 |
| code | `phase5_vlm_multimodal_sft/run_seqkd_sft_autoresume.sh` → `launch_stage2.sh` |

---

## 3. Standard eval spec + data volumes

Pipeline: `phase5_vlm_multimodal_sft/eval_benchmarks/run_all_evals.sh`
(takes a DCP ckpt directly; `*_LIMIT=0` = full set; FSDP=8 greedy decode).

**Core 3 (have pre-seq-KD baseline anchors):**
| benchmark | full N | metric | baseline (step-5200) |
|---|---|---|---|
| GQA test-dev-balanced | 12,578 | exact-match acc | 12.3 |
| MMBench-EN-dev | 4,377 | 4-way MC acc | **36.4** |
| POPE (random+popular+adversarial) | 8,910 | F1 / acc | 50 (always-"no", F1=0) |

**Breadth (repo-supported):** ScienceQA-IMG test 2,017 · MMMU val 900.

**Teacher eval (to complete the baseline/teacher/student triangle):** run the teacher
through the same suite **at the same `max_pixels=1003520`** used in generation
(else full-res inflates it — not apples-to-apples).

**NOTE:** I only ever reported 500-sample SMOKE subsets in chat; the standard run is
`*_LIMIT=0` (full). GQA + POPE eval data were deleted on this box — re-download on H200.

---

## 4. ⚠️ Partial eval result (step-3450, 66%) — NEGATIVE on MMBench

| | MMBench-EN-dev (full 4,377) |
|---|---|
| baseline (pre-seq-KD step-5200) | **0.364** |
| S5 step-3450 (66% seq-KD) | **0.271** (parse_rate 0.9965) |

**seq-KD REGRESSED MMBench by ~9pp** (0.271 ≈ near 4-way random 0.25). parse_rate 99.65%
→ format is fine; the model genuinely selects worse.

**Likely mechanism (= same family as the OPD failure: task/format mismatch):** the teacher
rewrote *every* answer in a verbose 512-token reasoning style. MMBench is single-letter
multiple-choice; the original mix665k taught crisp MC answering, and seq-KD pulled the
student toward the teacher's verbose distribution → drifted away from decisive MC selection.

**Caveats:** (a) 66% partial, cosine LR not decayed — final could differ, but −9pp + near-random
is not encouraging; (b) only MMBench measured (GQA/POPE data deleted, ScienceQA didn't finish);
(c) this box is untrusted.

---

## 5. H200 redo — key guidance

1. **The fla causal_conv1d sm_120 crash does NOT exist on H200.** It is a Blackwell
   sm_120 (RTX 5090) Triton autotuner bug. H200 is Hopper sm_90 → no crash. So on H200:
   - use **seq_len 1536** (full distilled data, no truncation; distilled p99 = 1355 + 196 vision),
   - no autoresume churn / SAVE_FREQ=50 needed; standard checkpointing.
   - Do NOT apply the vendored_fla shadows (PR13/b/c) — they target sm_120 and PR13b/c were
     a measured REGRESSION (reverted, commit 9818635). Run stock fla on H200.
2. **Address the negative MMBench result before committing to full seq-KD:** have the teacher
   give **task-appropriate-length** answers — concise for MC/short-answer prompts (GQA, MMBench-style),
   verbose only for caption/complex-reasoning prompts. A blanket verbose rewrite hurts crisp axes.
   Alternatively keep original answer *format*, let the teacher only correct *content*.
3. GQA + POPE eval data must be re-downloaded (deleted here). POPE always-"no" (yes_ratio=0)
   should get a 5-min harness sanity check before concluding it's a true model bias.

---

## 6. Backup manifest (absolute paths — non-git artifacts)

```
🔴 critical:
  /workspace/torchtitan_attention_residual/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_full.json   1.1G
      └─ the offline KD dataset (665,298 recs, teacher-rewritten). MOST EXPENSIVE artifact;
         with it you can skip re-distillation and go straight to student SFT.
  /workspace/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200/             3.0G
      └─ pre-seq-KD VLM student base (HF) = baseline MMBench 36.4. DCP form was deleted; this HF is the only copy.
  /workspace/torchtitan_attention_residual/phase10_ckpt_dcp_to_hf/hf_step9700_paperalign_C/             3.0G
      └─ 447M LM pretrain base (HF), before VLM SFT.

🟡 optional (this box's intermediates; being regenerated on H200):
  /workspace/.../phase5_vlm_multimodal_sft/runs/seqkd_sft_447m/checkpoint/step-3450/                    17G  (66% partial S5, DCP)

🟢 DriveLM (separate project, memory-flagged keep):
  /workspace/DriveLM_VLM_Project/deploy/                  2.7G  (TRT engines + B.5'' demo)
  /workspace/DriveLM_VLM_Project/checkpoints_qwen25/      45G   (VLA SFT ckpts)
```

## 7. Repos / branches (code — already on GitHub)
- `QIU023/torchtitan_attention_residual` @ `main` — seq-KD scripts, PR13 series, eval pipeline, this doc
- `QIU023/torchtitan` @ `attention_residual_dev` (submodule)
- `QIU023/sglang` @ `attention_residual_inference`
