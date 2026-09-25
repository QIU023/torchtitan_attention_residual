# CLAUDE.md — Kimi K3 / AttnRes reference stack (logbook repo)

Operational context for any Claude instance working in this repo. Full narrative:
[phase13_k3like_48b_posttrain/HANDOFF_2026-07-17.md](phase13_k3like_48b_posttrain/HANDOFF_2026-07-17.md)
(authoritative strategy, from the planning session) and
[phase13_k3like_48b_posttrain/PLAN.md](phase13_k3like_48b_posttrain/PLAN.md)
(repo-side execution detail; §0 reconciles the two).

## Language rule (user, 2026-09-25)

- Replies to the user: always Chinese. Internal reasoning may be English.
- Logbook explanatory files (notes, audits, handoffs, READMEs in this repo): Chinese where possible.
- Anything that goes into a contributed repo (torchtitan, veRL, pytorch, sglang, ...): PR bodies, review replies, commit messages, code comments and docstrings: English, as the PR-text rule below says.

## Commit-message rule: no cross-repo reference forms (both repos, incl. the fork)

Never put `owner/repo#N` or a full `github.com/.../pull|issues/N` URL for a
THIRD-PARTY issue/PR in a commit message (title or body) — on push, GitHub
creates a permanent "referenced this PR" event in THEIR timeline; ~14 such
events already spammed pytorch/torchtitan#4025 (irrevocable; history rewrite
does not remove them and can double-fire). Write `PR-4025` / "the upstream K3
PR" instead. Bare `#N` resolves repo-locally (harmless) but avoid it too for
consistency. Deliberate cross-links belong ONLY in issue/PR comments we
intentionally post. Reference forms in FILE CONTENT (docs) are fine — files
never fire timeline events. Our own repo's issues/PRs are exempt.

## Push rule for published PR branches (user, 2026-08-29)

Never push directly to a branch that backs a published PR (`k3_ep`, `k3_cp`,
`k3_pp`, ...). The flow is: push to the fork's REVIEW branch
(`ep_review1`/`cp_review1`/`pp_review1`), the user verifies locally and
approves, and only then does the PR branch get synced. No exceptions for
"trivial" fixes. Related trap, twice triggered: `pre-commit`'s pyrefly hook
edits files REPO-WIDE (deletes "unused" suppressions with newer pyrefly), so
never `git add -A` after running hooks -- stage only the intended files and
`git checkout` the hook's collateral away.

## Numerics-table rule (user, 2026-09-06)

Reviewers read step-10 gaps as bugs (4312, 4492). A results table pairs only
cells that read the same samples: the loader shards documents by dp rank, so
dp1 and dp2 rows are different data streams (their step-1 forward already
differs) and never sit as a before/after pair. The comparison a PR needs is
the same cell with and without the change (backend, kernel, transport), on
ONE shared inductor cache -- a fresh cache picks other autotuned kernels and
moves step 3/10 of the K3 debug flavor by percents on its own. Every table
carries one noise-floor row (the same cell twice on fresh caches, or the same
data under another reduction order) so the reader sees what the flavor does
by itself. Step 1 (bitwise) and step-1 gradient comparisons are the
correctness bar; step 10 is shown, never argued from.

No step past the point where the reference memorises the dataset is ever
reported (user, 2026-09-12). The K3 debug set repeats within a 100-step run:
the reference loss stops falling and swings between near-zero and normal
values (dp1 at 256 tokens/step: 0.84 at step 60, 0.59 at 90; dp2 at 512:
1.27 at 60, 0.82 at 90), and percentages against it measure memorisation,
not the change. Before a table is drafted, read the reference's loss
trajectory and report only steps before its first non-monotone drop -- for
the debug flavor at 256-512 tokens/step that is steps 1 / 10 / 20. A longer
run may still back an "identical on all N steps" claim; its late values are
not shown. One table per run carries loss and grad norm side by side.

## Numerics-acceptance rule (user, 2026-09-06)

When a maintainer challenges numbers, the reply's first sentence states the
verdict against a stated bar, not the method. The bar we claim a change on:
step-1 loss BITWISE between the paired cells, and step-1 gradients bitwise
(or, when a comm order legitimately differs, every parameter group inside
the stated floor) -- N/N parameters, counted, with sign flips reported. A
change that clears that bar is correct even when step 3/10 move; say so in
one sentence and show the noise-floor row that sizes the move.

If any gap remains, it is LOCATED, never waved at: name the layer, op or
reduction that produces it, the probe that found it (script in
`matrix_scripts/`, tree named), and the magnitude next to the floor. An
unexplained gap blocks the push -- do not file numbers we cannot account
for, and never attribute a gap to "backend arithmetic" or "bf16 rounding"
without a measurement that separates it from a real defect. When the honest
answer is that the cells are not comparable (different data streams,
different caches), say that first and re-pair them per the numerics-table
rule; do not argue the old table.

## Review replies: reconstruct every motive, then take one position (user, 2026-09-25)

When reviewers pull different ways (one asked for a check, another questions it), or one reviewer argues from a leaning, never side with whoever spoke last. Before drafting:
- find each earlier decision on that code: the commit, the comment that caused it, and what it guarded at the time;
- check whether that reason still holds in the current tree;
- work out what each option costs the PR's main model and every other model on the same code path.

The reply states the one position that analysis supports. It gives the reviewer who did not see the history its context, with links. In the same reply, it tells the reviewer whose request is being reversed why. It names who owns the gap. Do not let "Agreed" stand in for the argument, and do not offer to flip back if the other side prefers ("Happy to keep X if you prefer"): that is fence-sitting. Ask for a look at the concrete change instead.

Trigger: the first PR 4312 r4097059930 draft agreed with jinsooihm and offered to keep tianyu's assertion. It did not explain that the refusal had replaced a silent clearing whose one reason (a removed `__post_init__` check) was gone. It also left out what the refusal cost: Kimi K3 could not reach vp4 from the knob at 93 layers, and the other models on that path could not use the knob either.

## Diff-audit rule (user, 2026-09-09)

Before a PR draft is called ready, read the branch's own diff line by line --
`git diff <base> <head>` -- not just the changed-file counts. Three things
that reached a draft this way and must never appear upstream:

- a logbook path in a source file (`See phase13_.../X.md`); the code names no
  file the reader cannot open from the repo it is in;
- experiment records inside a docstring (measured cv numbers, bin sweeps,
  "two findings about that plateau"); findings belong in the PR body or the
  logbook, the docstring says what the function does;
- docstrings that carry the design argument. One K3 file reached 115 docstring
  lines out of 408. Upstream's own components are far leaner; match them.

The audit is part of drafting, not a step after review asks for it. Its
output goes in the kit's notes so the next reader can see what was checked.

## Flavor rule (user, 2026-08-29)

Do NOT add model flavors casually on upstream-bound branches. A text-only
flavor was already rejected in review, and backend-selection flavors
(`kimi_k3_debugmodel_deepep` and the like) must not be added: reviewers who
want another backend change the registry parameter themselves; the debug
flavor stays on `standard`. Capability notes ("these backends run on this
model") go in a 1-2 line code comment in parallelize.py, with no
test-result language. Feature flavors that are the only way to ENABLE a
technique (qb, mx_qat, lora) are currently tolerated -- flux ships mxfp8
flavors upstream -- but each one needs that justification, not convenience.

## PR-text rule (maintainer feedback 2026-08-13, restyled by the user 2026-08-29)

For new bodies the section structure below is superseded by the #4577 format (see "#4577 is the reference for code and body", user 2026-09-14); the rest of this rule (English only, one-line paragraphs, the first sentence answers the question, nothing the branch does not carry) still holds.

Upstream PR bodies and review replies read as terse human engineering notes,
English only (no Chinese, not even a preamble). Structure is the user's
sectioned format, the same across every body in `Raising_PRs/PR_K3_PARALLELISM/`:
`### Summary` (exact op-level before/after stated FIRST), `### Design`
(only when there is a design; two-level bullets), `### Results` (one table
per run, one-line caption, no prose around it, rows per configuration,
steps 1 / 3 / 10; a `torchrun` reproduction block before the main table),
`### Changed files` (an indented block, `file  +a/-b  one clause`), then
`### CI/CD Coverage` and review-round sections as needed. Every paragraph is
ONE line -- never hard-wrap prose, the rendering breaks. Code, identifiers
and `file.py:a-b` in backticks; math as `$...$`; tensor names in italics.
Everything else -- design history, alternatives, pre-empted objections --
goes to a linked logbook doc or a follow-up comment when asked. A body
describes only what its own branch carries. Code comments 1-2 lines.
When a maintainer asks a question, the first sentence of the reply answers
it. Verbatim trigger: "sorry I couldn't really understand the PR summary
which seems to be written by AI."

## No new branches unless asked (user, 2026-09-13)

Do not create a new git branch (or a `*_reviewN+1` review branch) on your own. Changes go onto the branch the work already lives on -- for a PR, its current review branch (`pp_review4` for PR 4312) -- unless the user explicitly asks for a new branch, or the change would clearly conflict with what that branch must keep (then say so and ask first). The fork already carries dozens of review / probe / integration branches and ~250 worktrees; every extra one is state the user has to track. Throwaway probe hacks go into an existing probe branch or an uncommitted patch file in `matrix_scripts/`, not a new branch.

## Cold compile caches: locate and rerun before writing anything (user, 2026-09-13)

Every cell of a comparison runs on ONE inductor / triton cache, warmed by a 1-step run of each configuration before the measured runs. The kit's `cell()` gives each cell its own cache (`ind_<name>`, `tri_<name>`): fine for a smoke, never for a table; a cold cache autotunes flex and picks other kernels, and a copy of a cache that was filled cold does not reproduce the cold run either. Incident: with the fp32 grad norm, dp2 x pp2 read 14.4183 at step 1 on its own cold cache and 14.4192 on a copy of it, while on the cache the reference had warmed it read the reference's 14.4170 and stayed bitwise for 100 steps; before that rerun a stream-race story was proposed, a probe written for it, and an "except dp2 x pp2" sentence put into the reply draft.

When a cell disagrees: do not write it into a draft, note, body or reply, and do not propose a mechanism. First locate it (the same cell again on the same cache, a step-1 gradient dump against the reference on a shared cache), rerun on the shared warm cache, and only then write the result. Tell the user explicitly that the number is held back until that rerun lands.

## Drafts go in files, not in the chat (user, 2026-09-12)

Reply and PR-body drafts are never pasted into the conversation: markdown tables and code blocks do not render there. The chat names the file and the marker to copy from (`--- PASTE BEGIN ---` / `--- PASTE ---`) and says what changed; the text itself lives only in the `.md` file.

## Drafts speak as the author: no "we" (user, 2026-09-13)

Every draft the user will paste to GitHub (review replies, PR bodies, comments) is written as the author: first person singular ("I", "my earlier reply") or impersonal ("the test now ...", "this PR ..."). Never "we" / "our" / "us": reviewers read it as an AI answer pasted by the author, and they dislike that. Before handing a draft over, grep its paste section for we/our/us.

## No dashes in drafts (user, 2026-09-13)

Paste-ready drafts use no dash as punctuation (no em dash, en dash or spaced ` -- `); use a period, comma, colon, semicolon or parentheses. Grep the paste section before handing it over. Details in `Raising_PRs/PR_WRITING_RULES.md`.

## Abstraction rule: titan's seams first, no side structures (user, 2026-09-10)

Triggered by quantile balancing: our #4412 built a side object (`QuantileBalancer` holding histograms in a dict keyed by `id(moe)`, a `register_forward_hook` on the router that redoes the top-k, a flavor-level `post_optimizer_build_fn` swap that needs the sign-rule coefficient set just to get the bias buffer); the maintainers' #4577 does the same maths as a `TokenChoiceTopKRouter` subclass with a `Module.Config`, a non-persistent buffer re-created by `_init_self_buffers`, the persistent `expert_bias_E` owned by the MoE, `model_registry(post_optimizer_build_fn=...)` for every flavor, and a `DTensorTestBase` GPU test. Same pattern in the old tree generally: 7.6k lines in the K3 folder against upstream's 2.25k, with model-local LoRA, key maps, pipeline adapters and a tests dir of their own where core had the seam. Before writing any helper, hook, wrapper or side object, find the titan class that owns that responsibility and extend it (details in `phase13_k3like_48b_posttrain/QB_4577_VS_4412_2026-09-10.md`):

- Model behaviour: subclass the `models/common` module (`TokenChoiceTopKRouter`, `MoE`, `GroupedExperts`, `FeedForward`, `Attention`) and widen its `Config` (`Module.Config`, `build()`); state is a `register_buffer` (persistent when it must checkpoint) plus `_init_self_buffers` for the meta -> device move; never a dict on a helper object, never a forward hook to recover what the module's own forward already computed (the graft port did this right: the plain MoE is core's router + experts + FFN).
- Per-step actions: `OptimizersContainer.register_step_pre_hook` via the spec's `post_optimizer_build_fn` on `model_registry` (all flavors), composing with core's registration rather than replacing it; the sign rule and any balancer are alternatives selected by config, not by which flavor overwrote the slot.
- Meshes: `ParallelDims.get_optional_mesh("loss")`, `get_dense_tp_mesh()`, `get_mesh(...)`; never a hand-built group, and reductions must name every axis that shards the tensor (the router is token-sharded on the dense-tp axis under EP).
- Recompute: `remat.region(..., recompute=False)` + `recompute_needs_tensor` for routing decisions, the way core's router does it, instead of reasoning about hook order under SAC.
- Parallelism: the spmd declarations and `local_map` of #4527, `apply_compile(fullgraph=)`-style parameters on core functions, `parallelize_*` seams; a core change that only serves our model is the upstream issue to raise, not a fork edit (the TP/SP clip-grouping revert, memory `search-the-host-repo-utils-first`).
- Checkpoint formats: `StateDictAdapter` + the HF storage readers (`QuantizedHuggingFaceStorageReader`), not a model-local key map module.
- Tests: `tests/unit_tests/cpu` and `tests/unit_tests/gpu` (`DTensorTestBase` / `with_comms` for anything with a collective), never a tests dir inside the model folder; a GPU test that runs the real collective beats a CPU mock of it.
- Review yardstick: when a maintainer re-implements a feature of ours, the diff between the two is the list of seams we missed; write that list down (as above) before touching the code again.

## Reuse check before any private helper (user, 2026-09-11)

The abstraction rule above, applied at function level, because it keeps recurring and reviewers read each case as not having looked. Three in one week: `_local_head_split` (core already has `local_head_split(t, head_dim)` in `models/common/attention.py`; it survived a handoff that read the GQA closure `local_qkv_head_split` instead and concluded core could not take a head dim), `_set_vision_encoder_sharding` in K3 (a line-for-line copy of Kimi K2.5's in `kimi_k2_7/sharding.py`, differing only in `pre_norm` vs `post_norm`, written right after the reviewer asked to "reuse or generalize the existing MoonViT sharding path"), and quantile balancing (#4412 against #4577). Before an upstream-bound branch keeps or adds any private `def`:

- Search `models/common`, `distributed`, `components` and the sibling models (`kimi_k2_7` for the MoonViT tower, `qwen3_5` for hybrid attention, `deepseek_v3` for MLA and MoE) for the behaviour, not just the name, and read the candidate's signature before deciding it does not fit.
- A sibling model's function with a one-line difference is generalized into `models/common` with that difference as a parameter and called from both models; it is never copied, and never called-then-overridden.
- When core really lacks it, the PR body says so in one line naming what was checked; the docstring does not argue it.
- The diff audit lists every new private `def` with what was searched and what was found. A copy is a finding, not a style note.

## #4577 is the reference for code and body (user, 2026-09-14)

The maintainers' quantile-balancing PR (pytorch/torchtitan#4577, shuhuayu) is the standard our code and bodies are held to; ours (#4412) was closed in its favour on 2026-09-11. Added lines, measured on the two diffs: #4577 has 398 code, 8 comment and 10 docstring lines (under 5%); #4412 had 381 code, 50 comment and 61 docstring lines (29%), and four of Shuhua's eight review comments on it were "nit: remove." Structural comparison: `phase13_k3like_48b_posttrain/QB_4577_VS_4412_2026-09-10.md`.

Order of work, before any code is written:
1. Survey the current tree's module structure (`models/common`, `components`, `distributed`, the sibling models) and decide where each piece belongs. Reusable behaviour goes to the module that owns that responsibility (#4577: the router subclass and the histogram `Module` in `models/common/moe.py`, the optimizer pre-hook in `components/optimizer/optimizer.py`); the model folder keeps only its config and wiring (#4577: 6 lines in `kimi_k3/__init__.py`, 20 in `kimi_k3/moe.py`).
2. Assemble from what exists (a subclass with a `Config`, buffers with `_init_self_buffers`, `model_registry(post_optimizer_build_fn=...)`, `register_step_pre_hook`), then write only the new maths. Never a new file in the model folder that re-implements a common responsibility (#4412's `kimi_k3/quantile_balance.py`, 299 lines).
3. No hooks or patches unless there is no seam: no `register_forward_hook` that recomputes what the module's forward already computed (#4412 ran top-k twice), no monkeypatch, no module global or `_armed` flag, no dict keyed by `id(module)`, no call into another class's private method, no flavor-only wiring of a model-wide feature. titan's own extension points (`post_optimizer_build_fn`, `register_step_pre_hook`, `Module.Config`, `_init_self_buffers`) are the seam and are fine. When a hook really is needed, the PR body says in one line which seam was missing.

Comments and docstrings, the way #4577 writes them:
- A class or public function gets a one-line docstring saying what it does ("Top-k router that uses a biased Top-(k+1) cutoff during training."), plus a short paragraph only for a fact the code cannot show (a value range). No `Args:` block restating the signature, no docstrings on private helpers, no module docstring arguing the design, no report section numbers, memory sizes or measured values in code.
- A comment only for a non-obvious invariant or constraint, stated as a fact ("With EP, the router is token-sharded on the dense TP axis even when model-wide sequence parallelism is disabled."). Never a comment narrating the next line, what the PR changed, or why the design was chosen (#4412's removed lines: "The report balances the router bias by solving it, not by stepping it.", a three-line mechanism summary above a class, a memory-size note in a docstring).
- Tests live in `tests/unit_tests/{cpu,gpu}`; one GPU test that runs the real collective beats many CPU tests of an estimator.

Body format (#4577): `## Summary` is one sentence of what and why, then one bullet per component naming where it lives; `## Design` is short prose, the mechanism first, then one paragraph on why each piece lives where it does; `## Relation to #N` when it replaces or overlaps another PR; `## Test plan` lists the exact commands with their pass counts. No changed-files block (GitHub shows the files). A results table only when the PR's claim is numerical, under the numerics rules above.

## What this project is

IC (Yiqiao / QIU023) **reference implementation** of Kimi K3's training-side
infra — NOT a product. Kimi K3 (2.8T total / 104.2B activated, released 2026-07-16, weights+report
due 2026-07-27) confirmed Block Attention Residuals (AttnRes) + KDA in
production. This repo owns the earliest torchtitan AttnRes implementation +
PP cross-stage adapter (backward-correct, validated to PP8×VP4 on 8 GPUs).

Three adoption surfaces, in priority order:
1. **torchtitan upstream**: `experiments/kimi_k3/` folder to inclusion
   standard, structured to the qwen3_5 template (the hybrid linear-attention
   precedent) so core promotion is a `git mv` — but target experiments first;
   move to `models/` only if the maintainer proactively suggests it in review.
   New SHORT RFC "Kimi K3 support" before 7.27: cite the original AttnRes RFC
   (pytorch/torchtitan#3029, whose adoption gate K3 now satisfies) and offer
   issue consolidation. Maintainer history: Tianyu rejected upstreaming the
   generic cross-stage PP mechanism (~2026-04); the adapter lives as private
   impl inside the model folder's parallelize. Do not re-propose it as a
   generic mechanism.
2. **veRL recipe**: 48B+AttnRes SFT/GRPO one-command (LoRA + full-param configs).
3. **This repo**: integration hub — KD scripts, provisional 2.8T flavor,
   EP@896 scaled smoke, version pins, ≤8-GPU quickstart.

## Honesty rules (non-negotiable, from the user)

- Never claim 2.8T was personally validated. Claim: "validated on 48B real
  weights and K3-faithful topology; scale-out is config-level."
- Never present under-trained model benchmarks as competitive results.
- K3's "<2% overhead" (algorithm FLOPs) ≠ our "+2.7% step-time" (PP-adapter
  comms on PCIe). Never conflate.
- Structure details pending tech report → interfaces hold placeholders; say so.
- Inline comments: one line max; move WHY to PR body/docs.

## Repo map

- Submodule `torchtitan/` (fork QIU023/torchtitan). Branch `main` is a plain
  mirror of upstream `pytorch/torchtitan` main and tracks it (reset there
  2026-09-19; what it used to hold, the pre-rebuild integration alias, is kept
  by the tag `k3_int_20260902_pre_rebuild`). The integration tree is
  **`k3_on_4025`**, and PR branches are cut from upstream main, not from it;
  `attention_residual_dev` is retired. The model lives at
  `torchtitan/models/kimi_k3/` (it moved out of `experiments/` during the
  09-12/09-16 rebuilds): FSDP2/HSDP + TP + EP + CP + PP all run, CP through
  the maintainers' #4639 stack, PP through the model-local adapter.
- **attention-gym is not a submodule.** It was removed 2026-09-19: the checkout
  sat on `pr453`, an unmerged upstream PR branch of drisspg's from 09-02, while
  `torchtitan/pyproject.toml` pins `attn-gym[linear] @
  git+https://github.com/meta-pytorch/attention-gym.git@main`, and the older
  checkout lacked `ContextParallelRouting`, which stopped the tree's K3 GPU
  tests from even collecting. It is now an ordinary venv package installed from
  that pin; reinstall with the same line rather than re-adding a submodule. The
  old branch is preserved as `pr453` on the fork.
- Submodule `sglang/` (fork QIU023/sglang, branch
  `attention_residual_inference`): Block AttnRes two-phase inference overlay +
  VLM serving + PR branches (pr1/pr7/pr8/pr15 pushed).
- `phase2..phase12_*/`: logbook of past phases (pretrain evidence, PP pressure
  tests, VLM SFT, GRPO/OPD attempts, AD/VLA research docs).
- `phase13_k3like_48b_posttrain/`: current phase. PLAN.md + HANDOFF doc.
- `Raising_PRs/`: upstream PR filing kits (sglang/fla/pytorch/torchstore).
- `K3_RELEASE_IMPACT_2026-07-16.md`: verified K3 facts + reconciliation
  checklist for 7.27.

## Current state (2026-07-17)

- Validated: AttnRes+PP adapter numerics (|Δloss| ≤ 0.011 @ PP8×VP4, 48B-shape
  downscale, −11.4% peak mem / +2.7% tps); 447M full pipeline
  (pretrain→SFT→GRPO end-to-end, model too weak to gain — infra correct);
  PR15 loaded official Kimi-Linear-48B in SGLang.
- Upstream torchtitan merge: **DONE on the fork** (2026-07-17, `469577cdf`).
  Dev branch diff vs upstream/main is now exactly `experiments/kimi_k3/`
  (renamed from attention_residual) + 1 registry line. `experiments/rl/` =
  upstream's rebuilt version; our SGLang/Monarch RL is preserved on branch
  `experiments_rl_unmerged` (phase11 replay must check that branch out).
  All torch-2.9 compat shims dropped (fleet is torch 2.11). compileall
  clean; **pytest + debug-flavor smoke still pending on the GPU box.**
- HF↔DCP converters exist as phase11 scripts
  (`hf_to_dcp_kimi_attn_res.py`, 424/424 keys @ meta-49.12B) — the handoff's
  "state_dict_adapter ❌" is really "promote script → titan
  state_dict_adapter.py" ⚠️.
- veRL-torchtitan backend: handoff says veRL native = FSDP/Megatron only
  (titan backend = major integration work); an earlier web search suggested
  titan engine-workers exist. **Unverified conflict — check veRL source on the
  GPU box before planning around either.**

## Near-term order (pre-7.27, from HANDOFF §8)

① titan `kimi_k3/` folder to inclusion standard (debugmodel CI, parity,
smoke curves+MFU, state_dict_adapter) → ② veRL recipe one-command
(LoRA-only weight sync first) → ③ 48B POC (LoRA + full-param small GRPO;
α trainable-vs-frozen; cross-engine logprob parity) → ④ docs/extension
points → ⑤ provisional 2.8T flavor + EP@896 scaled smoke → ⑥ new RFC.
KD/downscale line is deprioritized behind ①②. Flavor configs are a
**parameterized generator** (layer ratio / experts / latent dims as variables).

On 7.27: watch vLLM PR queue (config truth may land before the report) →
artifact-discovery checklist (K3_RELEASE_IMPACT §4; incl. packed-MXFP4
quantized-weight import — never treat packed weights as plain tensors) →
regenerate flavors → official weight mapping → freeze weight-sync tensor
naming against official vLLM K3 class → rerun smoke+parity → K3 model PR.
Competitive context: Megatron-Bridge#4910 tracks KDA/AttnRes/Block-AttnRes-PP
as open/planned; our PP adapter row is already validated (PLAN §0b).

## Key technical positions (details in HANDOFF §5)

- AttnRes is likely position-wise → inference needs NO persistent cross-stage
  cache, only per-token stage payload (vLLM IntermediateTensors); keep the
  interface open until 7.27.
- LoRA does NOT exempt cross-stage backward — skip-edge gradients are real
  autograd paths; adapter must route them or gradients are silently wrong.
  Optimization hook: `first_trainable_stage` partial-backward truncation.
- CP for KDA = state-passing/scan (LASP-family), not ring attention; declared
  non-goal in the RFC (also blank in titan's qwen3_5 — a future opportunity).
- 48B graft anchor: Kimi-Linear-48B + AttnRes(α=0) [+ Gated MLA near-identity]
  is numerically identical to the original checkpoint at step 0 — the cleanest
  adapter-correctness anchor.
