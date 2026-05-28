# PR14 OPD Stage D-5 Report — Mechanism Works, Capability Transfer Fails

**Date:** 2026-05-28
**Runs:** D-2 through D-5
**Compute:** ~18h on 8× RTX 5090 (32G each)
**Status:** Negative result on the capability-transfer claim.
The trainer scaffold is correct. Cross-VLM-encoder distillation from
**LLaVA-NeXT-8B (CLIP-336) → Kimi-AttnRes-447M (SigLIP-224)** does
NOT lift GQA capability on the small student under the configurations
tested.

---

## The Evidence Triangle

| Endpoint | Model | Steps | GQA testdev (N=300, greedy) |
|---|---|---|---|
| Baseline SFT | Kimi-AttnRes 447M (stage-2 SFT step-5200) | — | **12.3%** (37/300) |
| **Teacher** | llava-hf/llama3-llava-next-8b-hf | — | **63.7%** (191/300) |
| D-4 student (best OPD) | 447M after 50-step OPD on COCO caption | 50 | 9.3% (28/300) |
| D-5 student (failed scale) | 447M after 500-step OPD on VQA-aligned data | 500 | **0.33%** (1/300) |
| D-5 student (failed scale) | 447M after 600-step OPD on VQA-aligned data | 600 | **2.33%** (7/300) |

Teacher–student capability gap = 51.4 pp. Distillation didn't close it;
it **opened a larger gap relative to the SFT baseline** (the student
forgot rather than learned).

---

## What worked (infrastructure)

The PR14 OPDTrainer scaffold is mechanically correct end-to-end:

| Stage | Validation |
|---|---|
| A. `opd_loss` adapter | Reuses TRL `generalized_jsd_loss`; CPU smoke verified vocab-slice + label-mask grad masking |
| B. HF teacher load | Loads 8B teacher in bf16 on 3 cards via `device_map=auto + max_memory`; coherent caption generation |
| C.1. `TeacherScorer` | Load-once + `score(image, prompt, response) → (response_logits, response_ids)` |
| C.2. `compute_response_logits` | Sibling to existing `compute_token_log_probs`; bitwise-equal at gather-positions (CPU smoke) |
| C.3. `OPDTrainer` | Subclass of PolicyTrainer; per-episode backward; GKD hyperparam endpoint; DCP save endpoint |
| C.4. Runner wiring | `--task opd`, `--opd-{beta,temperature,lr,weight-decay,task-type,ckpt-interval}` CLI |
| D. Stage D-5 orchestrator | 600 steps × ~49 s/step stable; disk watchdog never fired; per-ckpt cleanup recovered ~18 G |
| E. DCP→HF + GQA eval cascade | Both ckpts converted + evaluated; `--projector-from-hf` handles LM-only DCP from OPDTrainer.save_dcp |

**Five real bugs found and fixed during D-1 → D-5** (each in commit history):
1. Trainer-subprocess `sys.path` missing `phase11_rlhf_grpo_infra/rlhf` (couldn't unpickle LauncherOPDTrainer)
2. Base64 data-URL handling in `TeacherScorer.score` AND in `_encode_image_for_logprob`
3. Prompt-extraction for the chat-template-aware teacher (decoded student prompt strips `<image>`)
4. `OPDTrainer.step` peak memory: per-episode backward + `del logits` + `expandable_segments`
5. Vision injection on the **student** side (`init_vision_from_hf`) — without it, student forward sees image-token-id 32000 as literal text; train/eval domain shift caused D-2/D-3 0% GQA before this fix

These are all upstream-actionable findings that survive the negative capability result.

---

## What didn't work (capability transfer)

### Hypothesis tree (ordered by how strongly the evidence supports it)

**H1 — Capacity / encoder mismatch is dominant** (strongest support).
- Teacher: LLaVA-NeXT-8B, CLIP ViT-L/14-336, 8.4 B params, 18 B-tok instruct-tune
- Student: Kimi-AttnRes 447M, SigLIP-base-224 (768 d → 1024 d), 150K LLaVA-Instruct-only SFT
- 18× param ratio + different vision encoders + different visual resolution (336 vs 224)
- Token-level JSD asks the student to match teacher's distribution over a sequence the student CAN'T plausibly produce given its limited vision-grounded perception
- Outcome on small student: gradient pulls lm_head toward outputs the body can't ground → degenerate

**H2 — Step count exceeded capacity ceiling**.
- D-4 at **50 steps** got the highest student GQA (9.3%, still under baseline but recoverable)
- D-5 at **600 steps** with task-aligned data drove the student deep into mode collapse ("0 0 1 0 0", "the on the…", caption-style on yes/no questions)
- Suggests there is a sweet spot in the 50-100 step range we never directly bracketed

**H3 — Task-format choice irrelevant or harmful**.
- D-4 (caption prompt + GQA eval) gave 9.3%
- D-5 (VQA prompt + GQA eval, **task-aligned**) gave 0.3-2.3%
- Counter to my hypothesis that task alignment would help. The longer run + sharper teacher distribution on VQA answers gave the optimizer more force, which the small student couldn't absorb.

**H4 — Optimizer + LR sweep not exhausted**.
- We swept LR by 80× (8e-4 → 1e-5). Confirmed 8e-4 catastrophically breaks the model (D-2/D-3 → 0%).
- 1e-5 + 500-600 steps still broke it. Untested: lr=5e-6 or lr=1e-6 + warmup of 50-100 steps.

### What we did NOT test that could change the verdict

- **Lower step count (10-30)** with VQA task — D-5 step-100 ckpt was deleted to save disk; we don't have a clean point in the 10-100 region
- **Smaller LR (5e-6, 1e-6) at length** — might preserve SFT baseline while marginally transferring
- **Sequence-level KD** (teacher-generated text → SFT loss) — bypasses logit-alignment problem; explicitly out of scope per user direction this session
- **Smaller capability gap teacher**: a 1B-3B VLM with SigLIP encoder (matched-encoder distillation) — most likely route, but no off-the-shelf candidate identified in this session
- **Hidden-state alignment instead of logit JSD** (TRL Liger fused path) — assumes shared model interface, doesn't trivially port to cross-arch

---

## Loss curve evidence (selected steps, lr=1e-5)

### D-4 (50-step COCO caption, β=0.5, T=1.0, NEPS=2)
```
step  0   loss=0.5912
step 10   loss=0.4915
step 20   loss=0.4381
step 30   loss=0.4160
step 40   loss=0.3920
step 49   loss=0.3845    (-35%)
GQA at 50: 9.3%
```

### D-5 (600-step VQA-aligned, β=0.5, T=2.0, NEPS=2)
```
step   0  loss=0.1632    ← VQA-task → teacher much more confident → start lower
step  49  loss=0.1540    (-6%)
step  99  loss=0.1378    (-16%)
step 199  loss=0.1057    (-35%)
step 299  loss=0.1017    (-38%)
step 399  loss=0.1049    (oscillating floor)
step 499  loss=0.1110
step 533  loss=0.1038
GQA at 500: 0.33%; at 600: 2.33%
```

**Key observation**: Loss in D-5 is much LOWER than D-4 (0.10 vs 0.39).
The training objective decreased more. But on the eval task, the
student degraded MUCH more. Loss-vs-capability decoupling — student
overfit to a token distribution it couldn't plausibly produce on the
vision side.

---

## Recommended next steps (out of scope for PR14, candidate PR15)

1. **Same-encoder teacher**: identify a 1-3B VLM with SigLIP-base-224 + Llama-3 (or compatible) tokenizer. Re-run D-style experiment with matched vision encoders.
2. **Bracket the sweet spot**: 10/30/50/100/150 step ckpts on the same VQA-aligned task, capture full GQA-vs-steps curve. Most-distilled-without-damage region likely 30-80 steps.
3. **LR floor sweep**: 5e-6, 1e-6 with warmup. Establish what LR preserves SFT baseline AND moves toward teacher.
4. **Sequence-level KD baseline**: have the teacher generate {VQA answers} on a held-out prompt set; SFT the student on those text pairs (no logit matching). Compare to logit-OPD. Vision-OPD style self-distillation also a candidate once a larger student is available (Vision-OPD paper uses 4B/9B base).
5. **Hidden-state matching** (matched arch): if a same-arch family exists, run Liger fused KD as the TRL-canonical path.

---

## Hyperparameter audit (D-5 the longest run)

| Param | Value | Source |
|---|---|---|
| Student | Kimi-AttnRes 447M @ stage-2 SFT step-5200 | local SFT |
| Teacher | llava-hf/llama3-llava-next-8b-hf | HF |
| Teacher placement | device_map auto, max_memory={1:"7GiB",2:"7GiB",3:"7GiB"} = phys cuda:5,6,7 | runner override |
| OPD prompts | mix665k COCO-only entries, first human turn (VQA-aligned) | `vqa_aligned_opd_task.py` |
| Steps | 600 | D-5 |
| LR | 1e-5 | --opd-lr; vs torchtitan default 8e-4 |
| Weight decay | 0.01 | --opd-weight-decay; vs torchtitan default 0.1 |
| β (GKD) | 0.5 (symmetric JSD) | TRL default, Agarwal 2024 |
| Temperature | 2.0 | --opd-temperature; standard KD |
| Loss | TRL `generalized_jsd_loss` slice to 128256 (shared Llama-3 base) | opd_loss.py |
| NEPS | 2 | per-episode bwd, memory bounded |
| ckpt interval | 100 (4 intermediate dropped by user, kept step-500/600) | --opd-ckpt-interval |
| Vision injection (student) | ON (SigLIP-base-224, 2-layer MLP from SFT HF) | init_vision_from_hf |

---

## PR14 disposition

The PR14 contribution stands as an **infrastructure PR**:

* Adds `OPDTrainer` (GKD on-policy distillation) as a sibling to `PolicyTrainer` in
  `torchtitan/experiments/rl/actors/opd_trainer.py`
* Adds `compute_response_logits` helper in `torchtitan/experiments/rl/actors/utils.py`
* Patches `_encode_image_for_logprob` to handle base64 data URLs (one-line fix
  benefitting both PolicyTrainer and OPDTrainer vision paths)

Capability-transfer evidence on **this specific 447M student × 8B teacher
cross-encoder pair** is **negative**. The PR description (see
`Raising_PRs/PR14_torchtitan_opd_trainer_on_policy_distillation/PR.md`)
should be updated to:
- Frame as infrastructure addition
- Cite the negative result as motivation for follow-up PR15 (matched-encoder distill)
- Keep the trainer's design (β/T hyperparams, JSD loss, per-episode backward) unchanged — these were all validated correct
