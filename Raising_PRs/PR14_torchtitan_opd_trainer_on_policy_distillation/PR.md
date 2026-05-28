# PR #14 — `experiments/rl/OPDTrainer`: on-policy distillation as a sibling to `PolicyTrainer`

**Target repo**: `pytorch/torchtitan`
**Target path**: `torchtitan/experiments/rl/actors/` (new files + one `utils.py` addition)
**Fork reference**: `QIU023/torchtitan @ attention_residual_dev`, HEAD `ba498b1`
**Effort**: ~3 days (trainer + utils helper + CPU smoke + 50-step distillation run + PR description)
**Risk**: low — new file under `experiments/rl/`, plus one additive helper in `utils.py`. No edits to `PolicyTrainer`, no core changes. Sibling to existing GRPO path, gated by a launcher flag.

---

## Suggested PR title

> [experiments/rl] `OPDTrainer` — on-policy distillation (GKD) as a sibling to `PolicyTrainer`, reusing `compute_token_log_probs` infra

---

## Suggested PR body

### Summary

`experiments/rl` today offers exactly one trainer flavour: `PolicyTrainer`
(GRPO, scalar-advantage policy gradient). For settings where the reward
signal is hard to design or noisy — early-stage multimodal VQA, captioning,
small custom architectures distilling from a much larger teacher — GRPO
collapses to flat reward with zero within-group variance, and the
advantage tensor becomes 0 every step.

This PR adds **`OPDTrainer`** (On-Policy Distillation, Agarwal et al. 2024
GKD formulation) as a sibling alongside `PolicyTrainer`, reusing the
existing SGLang generator + monarch actor mesh + FSDP/optimizer/ckpt
setup. It swaps GRPO's reward-scored advantage path for direct
generalized-JSD loss against a frozen teacher's logits at the
student's own rollout positions.

### Motivation

We hit two situations where GRPO has no signal:

1. **From-scratch arch ablation (Kimi-AttnRes 447M)** distilling from
   `llava-hf/llama3-llava-next-8b-hf`. Reward signal on GQA short-form
   answers is brittle (exact-match collapses; BLEU saturates; rule-based
   graders disagree across templates). Distillation needs a teacher
   distribution, not a scalar reward.
2. **Multimodal captioning / open-ended generation** where there is no
   single gold answer. GRPO with rule-based rewards collapses
   within-group variance to ~0 after a few hundred steps.

In both, the natural "right" objective is to match a stronger teacher's
distribution on the student's own sampled tokens — exactly what GKD
generalised-JSD does (Agarwal et al. 2024).

We did the work in our fork
([`QIU023/torchtitan@attention_residual_dev`](https://github.com/QIU023/torchtitan/tree/attention_residual_dev))
and `OPDTrainer` lived in `torchtitan/experiments/rl/actors/opd_trainer.py`
as a sibling to `trainer.py`. After validation we'd like to upstream
because:

- `experiments/rl` already owns the rollout / actor-mesh / FSDP wiring;
  adding distillation here avoids duplicating that infrastructure in a
  separate experiment folder.
- The student-side machinery `compute_token_log_probs` already needed
  (logits at response positions, vision injection for VLMs) is **one
  line away** from what OPD needs (full logits at response positions
  rather than gather-by-token-id). Reusing it is strictly better than
  forking another rollout pipeline.
- The teacher side is intentionally outside torchtitan — a small HF
  process loading `LlavaNextForConditionalGeneration` in bf16 on a
  single device, queried RPC-style by the actor mesh. Keeps torchtitan's
  "PyTorch-native training techniques" core principle intact (no HF
  Transformers dep in `actors/`).

### What lands in this PR

#### File 1 — `torchtitan/experiments/rl/actors/utils.py` (additive)

Add a sibling to `compute_token_log_probs` named
`compute_response_logits` that runs the same forward (incl. vision
injection) but returns the **full-vocab logits** at response positions
instead of gathered-by-token-id log probabilities. The two functions
share the prompt/gen tensor build, the varlen metadata, the explicit
positions, and the vision-tower / projector / image-mask injection
path.

```python
# torchtitan/experiments/rl/actors/utils.py

@torch.no_grad()  # call site disables this when grad is needed
def compute_response_logits(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
    *,
    vision_tower: torch.nn.Module | None = None,
    projector: torch.nn.Module | None = None,
    image_path: str | None = None,
    image_token_id: int = 32000,
) -> torch.Tensor:
    """Forward pass returning full-vocab logits at response positions.

    Mirrors ``compute_token_log_probs`` (positions, varlen metadata,
    vision injection); the only difference is the return value —
    [T_resp, V] float32 logits rather than [T_resp] gathered log-probs.

    Used by ``OPDTrainer`` to score the student at its own rollout
    positions and feed those logits into ``opd_loss`` against a
    teacher's logits at the same positions.
    """
    # ... ~50 lines, mirror of compute_token_log_probs lines 73-119;
    # return logits_f32[0, gen_start_idx:gen_end_idx, :]
```

The two functions remain separate (rather than one with a mode flag)
because the gather-and-cast operation is small and the call-site
contract is clearer per-function. Internal helper
`_encode_image_for_logprob` is shared as-is.

#### File 2 — `torchtitan/experiments/rl/actors/opd_trainer.py` (new)

`OPDTrainer` mirrors `PolicyTrainer` for:

- `_build_model` (FSDP + parallelize_fn + DCP load — call the existing
  `PolicyTrainer._build_model` via composition, not inheritance, to
  avoid pulling in the policy-gradient bits).
- Optimizer / scheduler / checkpoint setup.
- Monarch actor mesh registration.
- SGLang rollout via `SGLangGenerator` (reuse existing weight-sync path).

Diverges from `PolicyTrainer` in `step()`:

- No reward, no grader, no advantages, no ref-model KL.
- For each rollout episode:
  - Build aligned response slice on student via
    `compute_response_logits(model, prompt_ids, gen_ids, ...,
    vision_tower=, projector=, image_path=)`.
  - RPC the teacher process: `teacher_scorer.score(image_path,
    prompt_text, response_text) -> (teacher_logits, response_ids)`.
  - `loss = opd_loss(student_logits, teacher_logits, labels=...)`
    where `labels` masks prompt positions with `-100`.
  - `loss.backward()` and step optimizer.
- The actor-mesh boundary stays the same — `OPDTrainer.step` returns a
  `{loss, lr, throughput, ...}` dict matching the existing actor protocol.

#### What the teacher side looks like (out-of-tree)

Intentionally not in this PR — it's a small HF process that loads
`llava-hf/llama3-llava-next-8b-hf` with `device_map="cuda:0"` and
exposes `score(image, prompt_text, response_text)`. Talks to
`OPDTrainer` over the same actor-mesh primitive as the
generator-side rollouts. Lives in the launcher (or as a separate
optional experiment) so torchtitan's actor code stays HF-free.

A reference implementation (`teacher_scorer.py`) is in our fork
at `phase11_rlhf_grpo_infra/rlhf/teacher_scorer.py` for review.

#### Loss function (third-party, used as-is)

`opd_loss` is a thin adapter around
`trl.experimental.gkd.GKDTrainer.generalized_jsd_loss` — the Agarwal
2024 GKD formula. We do **not** re-derive the loss; we only handle
(a) vocab alignment (student head padded to 163840 for a Kimi arch
default, teacher head 128320 → shared 128256 = Llama-3 base vocab),
and (b) label masking on prompt positions. Living in the launcher
keeps the TRL dep out of torchtitan; if upstream wants TRL as an
optional dep, the file is ~30 LOC and trivial to move.

### Why upstream / why land it

- Distillation is a peer training objective to GRPO in the
  RLHF-adjacent space (cf. Anthropic's "constitutional AI distillation",
  the DeepSeek-R1 distill family, every recent <8B "frontier-from-
  frontier" model). `experiments/rl` is the right home because the
  rollout / actor / FSDP infra is identical — only the loss differs.
- Pattern is genuinely small (~150 LOC of new code, one ~50-LOC
  utility addition). Most of the value is the clean split between
  "score the student at its rollout" (in torchtitan) and "score the
  teacher" (out-of-tree HF process), which sidesteps the recurring
  question of whether to take a transformers dep.
- Removes the need for downstream forks to either (a) duplicate the
  monarch + SGLang + FSDP rollout infrastructure in a parallel
  distillation experiment, or (b) hack distillation into `PolicyTrainer`
  by repurposing `advantages` as a per-token KL surrogate.

### Test plan

**CPU smoke** (no GPU, runnable in CI):

1. `test_opd_loss_smoke.py` — 2 × 8 × 163840 student / 128256 teacher /
   masked labels. Verify finite loss; gradient nonzero on response
   positions in shared vocab; zero on masked prompt positions and on
   student padding dims `[SHARED_VOCAB:]`. (Already implemented in
   `phase11_rlhf_grpo_infra/rlhf/opd_loss.py::_smoke`.)
2. `test_compute_response_logits.py` — toy 2-layer model, prompt+gen
   tensor, verify returned shape `[T_resp, V]` and that float32 logits
   match the bf16 forward to within atol=1e-2 (the float32 conversion
   step is the only numerical change vs `compute_token_log_probs`).

**GPU smoke** (existing fork uses RTX 4090):

3. `test_opd_trainer_step_smoke.py` — 1-step OPD on a tiny student
   (4-layer 256-hidden Llama-3) + an even tinier "teacher" (1-layer
   stub matching vocab). Verify trainer.step() returns finite loss
   and parameters move (param-delta L2 > 0).

**End-to-end** (out-of-tree, will report results in PR description):

4. 50-step distillation run on Kimi-AttnRes 447M student ←
   llama3-llava-next-8b teacher on COCO captioning prompts.
   Acceptance: loss decreasing monotonically over the last 30 steps;
   GQA testdev greedy acc lifts from our SFT-only baseline of 12.3%.
5. Confirms the trainer scaffolding holds end-to-end on a real
   multimodal student.

### Discovered via

GRPO on the same student stalled with flat reward (zero within-group
variance after the captioner failed to answer VQA questions in a
parsable format — SFT recipe used 150K LLaVA-Instruct conversation
only, no academic VQA). Pivoted to OPD per the user's suggestion,
realised torchtitan's `experiments/rl` already had every piece except
the loss + the "return logits instead of gathered log-probs" forward.

### Out of scope (separate follow-up work)

- Teacher-side caching (precompute teacher logits on a fixed prompt
  pool to skip the teacher forward at train time). This is a real
  optimisation but orthogonal — same `OPDTrainer.step` accepts cached
  teacher logits instead of an RPC call.
- Speculative-decoding-style distillation (Eagle, Medusa). Different
  loss + rollout shape; deserves its own trainer.
- The student-head vocab slice (163840 → 128256) is hard-coded to the
  Llama-3 family here. A model-family-agnostic version would parse a
  config field; out of scope for the first landing.
- Engine-agnostic `Generator` abstraction (RFC PR #12) is independent —
  this PR uses `SGLangGenerator` as PolicyTrainer does today.

---

## Filing checklist

- [ ] Fork branch up to date with torchtitan `main` (rebase off the
      `627f4a31` trainer refactor first; this PR sits next to
      `trainer.py` and benefits from the rebased context).
- [ ] `pre-commit run --all-files` passes on touched files.
- [ ] CPU smoke: `pytest experiments/rl/tests/test_opd_loss_smoke.py
      experiments/rl/tests/test_compute_response_logits.py -x`.
- [ ] GPU smoke: `pytest experiments/rl/tests/test_opd_trainer_step_smoke.py
      -x` on a single A100 / 4090.
- [ ] PR body links to (a) the teacher-side reference impl in our fork,
      (b) the 50-step distillation result table (loss curve + final
      GQA acc).
- [ ] CC maintainers active on `experiments/rl/actors/trainer.py` (look in
      `git log --format=%an experiments/rl/actors/trainer.py`).
- [ ] Mention that the TRL dep is launcher-side; if upstream prefers,
      inline the ~30 LOC of `generalized_jsd_loss` instead with attribution.

---

## Why now (vs waiting)

- Foundation (loss + teacher scorer + smoke proofs) is built and
  validated on our fork — see `commits.md`. Trainer assembly remaining
  is ~3 days of focused work and three new files, all under
  `experiments/rl/actors/`.
- Pattern is becoming common downstream (DeepSeek-R1-distill, Llama-3
  distill family, our own 447M Kimi-AttnRes from-scratch ablation).
  Upstreaming the trainer scaffold once means downstream forks don't
  each re-invent the rollout-actor-mesh wiring.
- This PR also de-risks PR #12 (engine-agnostic Generator abstraction)
  by demonstrating a second consumer of the rollout infra besides
  `PolicyTrainer`.
