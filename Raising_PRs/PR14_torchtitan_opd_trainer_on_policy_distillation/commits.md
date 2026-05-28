# Backing commits — PR #14 `OPDTrainer` on-policy distillation

## Discovered in

**Phase 11 (RLHF / GRPO)** — GRPO on our 447M Kimi-AttnRes student
collapsed to flat reward / zero within-group variance after a few
hundred steps on multimodal VQA-style prompts. Root cause: the SFT
recipe before GRPO only used the 150K LLaVA-Instruct *conversation*
subset (no academic VQA), so the student is a captioner not an
answerer — every group produced narrative-style outputs none of which
parsed as VQA gold, so rule-based reward saturated and GRPO's
advantage went to 0.

Pivoted to on-policy distillation (OPD, Agarwal et al. 2024 GKD)
against `llava-hf/llama3-llava-next-8b-hf` as the teacher. The natural
home is `torchtitan/experiments/rl/actors/` next to `PolicyTrainer` —
everything except the loss and the "return logits instead of gathered
log-probs" forward is already there.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/torchtitan.git` |
| Branch | `attention_residual_dev` |
| Commit (trainer / utils — TODO) | **not yet on fork** — pending Stages C.2 / C.3 / C.4 below |
| Files implicated | `torchtitan/experiments/rl/actors/opd_trainer.py` (new), `torchtitan/experiments/rl/actors/utils.py` (add `compute_response_logits`) |

## Backing commits already in the fork (foundation pieces, all
in `attention_residual_dev`, **not** in the torchtitan submodule
itself — they live in the parent repo's `phase11_rlhf_grpo_infra/rlhf/`
folder and validate the design before the torchtitan-side changes
land):

| Commit | File | What it proves |
|---|---|---|
| `1edcf9c` | `phase11_rlhf_grpo_infra/rlhf/opd_loss.py` | TRL `generalized_jsd_loss` reusable as-is; vocab slice (163840 → 128256) and label-masking correct. Smoke: finite loss, grad on response shared-vocab positions, zero grad on masked prompts and student padding dims. |
| `7dd3769` | `phase11_rlhf_grpo_infra/rlhf/teacher_smoke.py` | `LlavaNextForConditionalGeneration` loads in ~3s, produces coherent caption on a known GQA image (`A snowboarder in mid-air…`), forward returns `[T_resp, 128320]` logits aligned to Llama-3 token ids. |
| `b42a71e` | `phase11_rlhf_grpo_infra/rlhf/teacher_scorer.py` | `TeacherScorer` class — load-once, `score(image, prompt_text, response_text) → (logits, ids)`. Reused across rollouts; 2nd call OK on shape `(5, 128320)`. |

These three commits live in the **parent** repo
(`workspace/torchtitan_attention_residual`), not the torchtitan
submodule. The PR's torchtitan-side commits (`opd_trainer.py` +
`utils.py::compute_response_logits`) are still pending — see "Stages
remaining" below.

## Stages remaining before the PR can be filed

| Stage | File | What it does | Status |
|---|---|---|---|
| C.2 | `torchtitan/experiments/rl/actors/utils.py` | Add `compute_response_logits` — sibling to `compute_token_log_probs`, returns `[T_resp, V]` float32 logits instead of `[T_resp]` gathered log-probs. ~50 LOC mirror of lines 73-119. | pending (this PR) |
| C.3 | `torchtitan/experiments/rl/actors/opd_trainer.py` | New `OPDTrainer` class — composes `PolicyTrainer._build_model`, replaces `step()` loss path with `opd_loss(student_logits, teacher_logits, labels)`. ~150 LOC. | pending (this PR) |
| C.4 | `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py` (launcher-side) | Wire `--task opd` switch: skip grader / reward / advantage, route to `OPDTrainer.step`, load prompt pool from `llava_v1_5_mix665k.json` COCO entries. | pending (launcher only — not part of upstream PR) |
| D | end-to-end | 1-step OPD smoke → 50-step run → GQA testdev greedy acc lift vs 12.3% baseline. | gates the PR |

## Conflict surface

`PolicyTrainer` was refactored upstream by `627f4a31 [rl] Trainer
refactor (#2985)` on 2026-04-20 (the same commit that obsoleted PR #4).
Our `attention_residual_dev` branch is behind that refactor — fork
rebase tracked in `Raising_PRs/FORK_REBASE_TASK.md`. **Land the
fork rebase first**, then write `OPDTrainer` against the rebased
`PolicyTrainer` shape; that way the upstream PR is rebase-clean from
day one.

`utils.py::compute_token_log_probs` was extended on 2026-05-15
(`581e78b [RL][VLM] vision-aware compute_token_log_probs in
PolicyTrainer`) — that commit *is* on `attention_residual_dev` and is
also upstream-pending. The new `compute_response_logits` follows the
same vision-injection contract; either land both in one PR or sequence
the vision-aware variant first.

## Filing path (after Stages C.2 / C.3 / D complete on fork)

```bash
# 1. Rebase fork onto upstream main if not done (FORK_REBASE_TASK.md).
cd torchtitan
git checkout attention_residual_dev
git fetch upstream
git rebase upstream/main
# (reconcile against 627f4a31 PolicyTrainer refactor)

# 2. Cherry-pick / hand-port the two torchtitan-side commits onto a
#    fresh upstream-tracking branch.
git checkout -b experiments-rl-opd-trainer upstream/main
git cherry-pick <utils.py compute_response_logits commit>
git cherry-pick <opd_trainer.py commit>

# 3. Add CPU smoke tests under torchtitan/experiments/rl/tests/.
#    See PR.md "Test plan" — opd_loss smoke, compute_response_logits
#    shape/numerics test, opd_trainer.step smoke.

# 4. pre-commit run --all-files; pytest experiments/rl/tests/ -x.

# 5. Push + open PR using PR.md as the body. CC PolicyTrainer authors
#    surfaced by `git log --format=%an torchtitan/experiments/rl/actors/trainer.py`.
git push origin experiments-rl-opd-trainer
```

## Why land it (vs keeping it fork-private)

- Pattern is becoming common (DeepSeek-R1-distill, Llama-3 distill
  family). Upstreaming the trainer scaffold once means downstream
  forks don't re-invent the rollout-actor-mesh wiring.
- Removes the temptation in downstream forks to hack distillation into
  `PolicyTrainer` by repurposing `advantages` as a per-token KL
  surrogate (which loses the GKD interpolation between forward and
  reverse KL).
- De-risks RFC PR #12 (engine-agnostic Generator) by demonstrating a
  second consumer of the rollout infra besides `PolicyTrainer`.

## Notes for the PR opener

- The TRL dep stays out of `torchtitan/experiments/rl/actors/`. The
  ~30-LOC `opd_loss` adapter lives in the launcher; if upstream wants
  it inline, copy `trl.experimental.gkd.GKDTrainer.generalized_jsd_loss`
  with attribution and drop the TRL dep entirely.
- Teacher-side `TeacherScorer` is deliberately not part of this PR — it
  uses HF `transformers` (`LlavaNextProcessor`,
  `LlavaNextForConditionalGeneration`) which violates core principle #1
  ("PyTorch-native training techniques"). Documented as an out-of-tree
  reference impl in `PR.md`.
- The student-head vocab slice (163840 → 128256) is hard-coded to the
  Llama-3 family. For the first upstream landing, document this as a
  Llama-3-only assumption with a clear TODO; generalising to other
  vocab pairs is follow-up work.
