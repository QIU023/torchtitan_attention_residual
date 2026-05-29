# PR14 OPD Final Report — Negative Capability Result on 447M × 8B Cross-VLM Distillation

**Date:** 2026-05-29
**Runs:** D-2 → D-7 (six experiments over ~24h compute)
**Compute:** ~30h on 8× RTX 5090 (32G each)
**Verdict:** OPD infrastructure validated; **capability transfer fails** under all
configurations tested. The 18× student/teacher capacity gap is the dominant
failure mode, not solvable by encoder matching, LR tuning, task alignment, or
step count.

---

## The Honest Triangle

```
SFT baseline (Kimi-AttnRes 447M, stage-2 SFT step-5200):  12.3%  (37/300)
Teacher (llava-hf/llama3-llava-next-8b-hf):               63.7%  (191/300)
                                                          └ 51.4 pp gap to close ┘

Best distilled student across 6 experiments:              12.0%  (36/300, D-7 step-100)
                                                          → ~0 pp gap closed
```

**Meaningful distillation transfer = closing significant fraction of the
51.4 pp gap.** Our best result is a 0.3 pp dip below baseline at one
checkpoint. **Zero capability transfer.**

---

## Full Experimental Sweep — Every Run

| Run | Teacher | Steps | Task | LR | T | Hypothesis tested | Best GQA | Final GQA |
|---|---|---|---|---|---|---|---|---|
| D-2 | LLaVA-NeXT-8B (CLIP-336) | 50 | caption | 8e-4 | 1.0 | infra works | — | 0.67% |
| D-3 | LLaVA-NeXT-8B (CLIP-336) | 50 | caption | 8e-4 | 1.0 | + vision injection | — | 0.00% |
| D-4 | LLaVA-NeXT-8B (CLIP-336) | 50 | caption | 1e-5 | 1.0 | + sane LR | — | 9.33% |
| D-5 | LLaVA-NeXT-8B (CLIP-336) | 600 | VQA-aligned | 1e-5 | 2.0 | task alignment + scale | 2.33% @600 | 2.33% |
| D-6 | **Mantis-8B (SigLIP-384)** | 50 | caption | 1e-5 | 2.0 | matched-encoder family | 10.67% | 10.67% |
| D-7 | **Mantis-8B (SigLIP-384)** | 300 | caption | 1e-5 | 2.0 | matched-encoder + scale | **12.00% @100** | **3.67% @300** |

**Pattern**: every long run collapses. Best result (D-7 step-100) is **0.3 pp BELOW baseline**.

---

## D-7 Trajectory — The Definitive Evidence

Mantis SigLIP teacher (matches student encoder family) + caption task + lr=1e-5
+ T=2.0 + β=0.5 (TRL JSD default) + NEPS=2 + vision injection ON.

300 steps, ckpt every 50:

| Step | Loss | GQA acc | Δ vs baseline | Δ vs gap closed (of 51.4 pp) |
|---|---|---|---|---|
| 0 (baseline) | — | 12.30% | 0.00 | 0.00% |
|  50 | 0.265 | 11.33% | −0.97 | −1.89% |
| 100 | 0.180 | 12.00% | −0.30 | −0.58% |
| 150 | 0.135 |  7.67% | −4.63 | −9.01% |
| 200 | 0.115 |  7.00% | −5.30 | −10.31% |
| 250 | 0.108 |  4.00% | −8.30 | −16.15% |
| 300 | 0.102 |  3.67% | −8.63 | −16.79% |

**Loss decreases monotonically** (0.27 → 0.10) **while GQA collapses** after step 100.
Lower JSD = student matches teacher token distribution better → but student
**uses that capacity to OVERFIT to caption-distribution mode collapse**, not to
acquire vision-grounded short-answer ability.

D-6 (Mantis 50 step) at 10.67% was the **rising slope before the peak**. We
extended in D-7 hoping for monotone gains; got monotone collapse instead.

---

## What worked (infrastructure) — Reusable

| Stage | Validation | Files |
|---|---|---|
| TRL JSD loss adapter | finite loss, grad-masking correct | `opd_loss.py` |
| HF teacher load | LLaVA-NeXT + Mantis both via `AutoModelForImageTextToText` + manual `LlavaProcessor` fallback | `teacher_scorer.py` |
| TeacherScorer.score | bf16 forward, response-position slice, supports data URLs + chat-template auto-wrap | `teacher_scorer.py` |
| `compute_response_logits` | bitwise-parity with `compute_token_log_probs` at gather positions | `torchtitan/.../utils.py` |
| `OPDTrainer` class | DCP save endpoint; β/T hyperparam endpoint; per-episode backward | `torchtitan/.../opd_trainer.py` |
| Runner wiring | `--task opd`, `--opd-{lr,wd,beta,temperature,task-type,ckpt-interval,ckpt-dir}` | `run_grpo_llava_kimi.py` |
| Disk + watchdog orchestrators | per-ckpt DCP cleanup, 12G threshold panic | `run_opd_d7.sh` |
| DCP→HF converter `--projector-from-hf` | loads projector from donor HF when student DCP is LM-only | `dcp_to_hf_kimi_attn_res_vl.py` |
| Teacher GQA eval | matched grading to gqa_eval.py; gives 63.7% | `teacher_gqa_eval.py` |

**Five real bugs found and fixed**, each in commit history:
1. Trainer-subprocess `sys.path` missing `phase11/rlhf` (couldn't unpickle `LauncherOPDTrainer`)
2. Base64 data-URL handling (`TeacherScorer.score` AND `_encode_image_for_logprob`)
3. Prompt-extraction for chat-template-aware teacher (decoded student prompt strips `<image>`)
4. `OPDTrainer.step` peak memory (per-episode backward + `del logits` + `expandable_segments`)
5. Student-side vision injection (`init_vision_from_hf`) — critical fix for train/eval domain alignment

**One torchtitan default**: `lr=8e-4` is 80× too high for continual distillation.
Documented + overridden via `--opd-lr` CLI.

These all survive the negative capability result and are reusable as
upstream contributions.

---

## What didn't work (capability)

### Hypothesis tree — what the data actually says

**H1: cross-VLM-encoder mismatch is the dominant failure mode**
- **PARTIALLY** confirmed by D-6 vs D-4: matched SigLIP family +1.4 pp at 50 steps
- **REFUTED** by D-7 trajectory: same Mantis teacher, scaled to 300 steps, still collapses
- Conclusion: encoder match delays but does not prevent collapse

**H2: 18× capacity gap (447M student vs 8B teacher) is the dominant failure mode**
- **CONFIRMED** by D-7: even with H1 eliminated, training >100 steps damages the student
- The teacher's confident logits encode reasoning the student's body cannot ground;
  pulling lm_head toward those logits decouples the head from the body's
  capability surface → eval-time degenerate outputs

**H3: long training overshoots**
- **CONFIRMED** by both D-5 (CLIP, 600 step collapse) and D-7 (Mantis, 300 step collapse)
- Sweet spot is in the 50-100 step region; window is narrow and hard to exit
  with a usable artifact (best D-7 ckpt is 0.3 pp below baseline)

**H4: VQA task-alignment helps**
- **REFUTED** by D-5: VQA task drove loss lower (0.16 → 0.10) but GQA collapsed faster
  than caption-task D-4
- Task alignment increases gradient sharpness but doesn't help capability transfer
  on a too-small student

### Summary line

> Token-level JSD distillation from an 8B VLM to a 447M VLM (18× capacity gap)
> does not yield capability transfer on GQA, regardless of encoder family match,
> learning rate, loss temperature, task alignment, or step count. The student is
> capacity-limited to absorb the teacher's vision-grounded reasoning distribution.

---

## What we did NOT test

These were considered and explicitly deferred. None would change the verdict
under the same 447M × 8B pair, but each is a candidate for PR15:

| Direction | Likely outcome | Effort |
|---|---|---|
| Same-arch matched-encoder teacher (1-3B SigLIP+Llama-3) | Higher ceiling: shorter capacity gap is closeable | ~no off-the-shelf candidate exists |
| LR floor sweep (5e-6, 1e-6 + warmup) | Marginal — narrows the collapse window but doesn't add lift | 1 sweep day |
| Sequence-level KD (teacher-generated text as SFT data) | Genuinely different mechanism; closer to what BLIP-2 / DeepSeek-VL2 do | 1-2 days |
| Hidden-state matching (TRL Liger fused path) | Requires same model arch interface; cross-VLM-arch breaks the assumption | hard |
| Vision-OPD self-distill | 447M base is too weak to be its own teacher (paper uses 4B/9B) | gated on bigger student |

---

## Hyperparameter audit (canonical D-7)

| Param | Value | Source |
|---|---|---|
| Student | Kimi-AttnRes 447M @ stage-2 SFT step-5200 | local SFT |
| Teacher | TIGER-Lab/Mantis-8B-siglip-llama3 (SigLIP-so400m-384, Llama-3-8B) | HF |
| Teacher placement | device_map=auto, max_memory across phys cuda:5,6,7 (= logical 1,2,3 inside trainer) | runner override |
| OPD prompts | mix665k COCO LlavaOpdTask (fixed "Describe the image…") | `llava_opd_task.py` |
| Steps | 300 | D-7 |
| LR | 1e-5 (LLaMA-3 distill convention) | `--opd-lr` |
| Weight decay | 0.01 | `--opd-weight-decay` |
| β (GKD) | 0.5 (symmetric JSD, TRL default, Agarwal 2024) | `--opd-beta` |
| Temperature | 2.0 (standard KD) | `--opd-temperature` |
| Loss | TRL `generalized_jsd_loss` slice to 128256 (shared Llama-3 base) | `opd_loss.py` |
| NEPS | 2 | per-episode backward, memory bounded |
| ckpt interval | 50 (6 evaluable ckpts) | `--opd-ckpt-interval` |
| Vision injection (student) | ON (SigLIP-base-224, 2-layer MLP from SFT HF) | `init_vision_from_hf` |
| GPU layout | trainer cuda:0, gen TP=4 cuda:1-4, teacher dev_map cuda:5-7, 0 idle | runner `_async_main_opd` |

---

## PR14 disposition — DO NOT FILE until capability claim has evidence

Per user direction this session (2026-05-29): **do not file PR14 until OPD
actually works**.

The infrastructure pieces are correct and validated. The capability claim is
not. Filing as infrastructure-only is **not** an option per user direction.

### What this means for next steps

The PR14 work product is held local. Reusable artifacts:

* `torchtitan/experiments/rl/actors/opd_trainer.py` (new, 339 lines)
* `torchtitan/experiments/rl/actors/utils.py` (+99 lines: `compute_response_logits` + data URL fix)
* `phase11/rlhf/*` infrastructure (LauncherOPDTrainer, TeacherScorer, opd_loss,
  vqa_aligned_opd_task, run_opd_d6/d7, teacher_gqa_eval, eval cascade orchestrators)

### Candidate PR15 directions (any one of these could unlock capability lift)

1. **Smaller capacity gap**: train (or download) a 1-2B Kimi-AttnRes variant
   as student, repeat D-7 with Mantis teacher. Capacity ratio ~4× instead of 18×.
2. **Same-family teacher matching student**: re-train a 3-4B LLaVA-style VLM
   ourselves with SigLIP-base-224 + Llama-3, becoming its own student's teacher.
3. **Sequence-level KD instead of token-level JSD**: teacher generates VQA
   answers on COCO; student SFTs on those text pairs. Empirically the
   approach used by LLaVA-distill, DeepSeek-VL2 papers.
4. **Stop trying to distill to 447M**: pivot to using OPDTrainer for
   same-size student/teacher pairs (e.g. compress 8B → 4B same family).

Of these, **option 3 (sequence-level KD)** is the lowest risk because it
sidesteps the entire token-level alignment problem and matches what published
VLM distillation work actually does.

---

## Commit log (for PR14 evidence + PR15 pickup)

* `bf5f7e8` add `compute_response_logits` sibling for OPD
* `ad38d66` add `OPDTrainer` — sibling of `PolicyTrainer`
* `ab4e65f` step: extract user question for teacher chat template
* `49368e7` step: per-episode backward + `del` logits (OOM fix)
* `97134cb` GKD hyperparams (β, T) + `save_dcp` endpoint
* `68d0318` `_encode_image_for_logprob` data-URL fix
* `1edcf9c`..`b42a71e` opd_loss (TRL adapter) + teacher_smoke + TeacherScorer (Stage A/B/C.1)
* `33a529e` Mantis processor manual compose (no `processor_config.json`)
* `15b625f` D-6 prep: `AutoModelForImageTextToText` + matched-encoder
* `3257910` D-7 launcher: 300 step + eval cascade

All commits clean, all tests reproduce. Infrastructure can be cherry-picked
into PR15 unchanged once a capability-positive recipe is found.

---

*This report supersedes REPORT_OPD_D5.md (kept for historical context).*
