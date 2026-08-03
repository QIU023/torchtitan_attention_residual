# Upstream-PR extraction cleanup checklist (kimi_k3 -> torchtitan)

2026-07-26. Target: the `experiments/kimi_k3/` upstream PR branch. Baseline:
fork `attention_residual_dev` @ `20bd4f3a`, upstream merge-base `fbceec07`.
Every item below says WHAT, WHERE (exact locations at `20bd4f3a`), HOW, and
the VERIFY gate. Items are ordered so each step's verification still holds
after the next step.

Ground rules for the whole exercise: the cleanup is comment/config/branch
surgery only -- **zero functional change**. The final gate (item 10) proves it.

---

## 0. Branch construction (do this first; items 4-5 fall out of it)

**What**: build the PR branch from upstream/main, never from the dev branch,
so the torch-2.12 compat shims and unrelated fork history never enter it.

**How**:

```bash
cd torchtitan
git fetch upstream
git checkout -b kimi_k3_upstream_pr upstream/main
# bring over ONLY the experiment folder and the registry line
git checkout attention_residual_dev -- torchtitan/experiments/kimi_k3
# then hand-edit torchtitan/experiments/__init__.py: add the "kimi_k3" line
```

**Verify**:

```bash
git diff upstream/main --stat
# must list ONLY torchtitan/experiments/kimi_k3/* and
# torchtitan/experiments/__init__.py (+1 line). Anything else = leakage.
```

---

## 1. `parallelize.py` module docstring: stale "CP not supported" claim

**What**: the top-of-file docstring (L9-37) still lists CP under
`**Not supported**` ("blocked on fla-core's chunk_kda triton kernel lacking
ring-recurrence"). CP has been Ulysses-style and composable since the 07-24
landing; commit `7d8acabe8` fixed the *function* docstring (L86-94) but
missed the module docstring.

**How**: rewrite the module docstring's parallelism list to include:

```
* **CP** -- Ulysses-style (all-to-all seq<->head inside each attention
  module); composes with TP/FSDP/PP/EP. Requires
  context_parallel_load_balancer=None (enforced with a ValueError).
```

Delete the `**Not supported**` block entirely. Copy the accurate wording
from the `parallelize_kimi_linear` function docstring (L86-94), which is
already correct. New text must be ASCII-only (fork rule).

**Verify**: `grep -n "blocked on fla-core" torchtitan/experiments/kimi_k3/parallelize.py`
returns nothing; `grep -n "Not supported" parallelize.py` returns nothing.

---

## 2. Internal phase-number scrub (~32 hits, 8 non-test files + tests)

**What**: docstrings/comments reference this project's private phase log
("Phase 6 A3", "Phase 3 cache adapter", "Phase 4e scope", "phase10/phase11
converters"). Meaningless to upstream readers.

**Where** (counts at `20bd4f3a`, excluding tests): parallelize.py (10),
model_configs.py (5), config_registry.py (5), attn_res_model.py (4),
multimodal_model.py (3), model.py (3), state_dict_adapter.py (1),
pipeline_adapter.py (1). Run the same sweep over `tests/`.

**How**: find every hit, replace the phase tag with the thing it names:

```bash
grep -rniE "phase[- ]?[0-9]+" --include="*.py" torchtitan/experiments/kimi_k3/
```

Replacement mapping (apply consistently):

| current | replacement |
|---|---|
| "Phase 3 cache adapter" / "the Phase-3 PP cache adapter" | "the cross-stage cache adapter (pipeline_adapter.py)" |
| "Phase 6 A3" / "(Phase 6 A6)" / "(as of Phase 6)" | drop the tag; keep the sentence |
| "Phase 4 onwards" (compile) | drop the tag |
| "Phase 4e scope" (multimodal) | "Scaffolding scope" |
| "Phase 5 deliverables" / "Phase 5 adds ..." | "a follow-up training recipe" |
| "phase10/phase11 logbook converters" | "the standalone HF<->DCP converters (424/424 keys validated at meta-49.12B)" |

**Verify**: the grep above returns zero hits in the PR branch.

---

## 3. Private planning-doc references scrub (~10 hits)

**What**: comments cite internal planning docs an upstream reader cannot
resolve: `HANDOFF sec 5`, `PLAN 0a #4`, `PLAN 3b`, `PLAN 0b gap (1)`,
`PLAN 1`, `handoff_status_20260420_part3.md`, "logbook".

**Where**: attn_res_model.py:142, lora.py:16, model.py:124, mxfp4_qat.py:15,
quantile_balance.py:24, state_dict_adapter.py:9+38, pipeline_adapter.py:32+321,
config_registry.py:16.

**How**, by kind:

- Reference that *justifies a claim* (e.g. pipeline_adapter's pressure-test
  citations): replace with the public logbook URL
  (`https://github.com/QIU023/torchtitan_attention_residual/...` -- the
  pressure-test reports and verification docs are public). One link max per
  docstring; move the rest to the PR body.
- Reference that is *pure planning cross-talk* ("PLAN 0a #4",
  "HANDOFF sec 5 anchor"): inline the one-sentence WHY and delete the
  pointer. Example: `# Graft gate (HANDOFF sec 5 anchor): ...` becomes
  `# Graft gate: alpha=0 makes the model exactly identical to the plain
  backbone at step 0 (adapter-correctness anchor).`

**Verify**:
`grep -rniE "HANDOFF|PLAN [0-9]|handoff_status|logbook" --include="*.py" torchtitan/experiments/kimi_k3/`
returns zero hits (public URLs containing "attention_residual" are fine).

---

## 4. torch-2.12-stable compat shims: must NOT be on the PR branch

**What**: three fork-only diffs exist because the fleet runs torch 2.12
stable while upstream main tracks nightly:

1. `torchtitan/distributed/utils.py` (+5): `_set_pg_timeout` fallback
   (2.12 has no public `ProcessGroup.set_timeout`).
2. `torchtitan/distributed/pipeline_parallel.py` (+19) +
   `torchtitan/trainer.py` (+15): module-level `_step_loss_kwargs` holder,
   because 2.12's `schedule.step()` lacks the `loss_kwargs=` parameter.

**How**: nothing to do if item 0 was followed -- the branch is built from
upstream/main and these files are never checked out. They stay maintained on
`attention_residual_dev` only.

**If a reviewer asks about 2.12**: answer honestly -- "the fork carries three
small stable-torch shims; not included here because upstream CI targets
nightly."

**Verify**: covered by item 0's `git diff --stat` gate.

---

## 5. `models/common/moe.py` scatter fix: RETIRED as a PR (2026-08-03) -- now a torch-2.12 compat shim

**What (updated)**: the routing-map scatter "fix" (`129e29de`) turned out to
patch a torch-2.12-only gap -- torch >= 2.13 has the `aten.scatter_`
sharding strategy and the bare construction works (evidence chain in
`Raising_PRs/PR16_*/PR.md`; probe in that folder). **PR16 is not filed.**
The fork keeps the change as a stable-torch compat shim, same category as
item 4's shims.

The OTHER core fix, `moe_sharding.py` EP-off `in_grad_placements`
(`Raising_PRs/PR19_*`), is unaffected: it is a declaration bug present in
current upstream source, with a DSv3 reproduction valid on the default
backend. PR19 files on its own; the K3 PR body references it if TP cells
depend on it.

**How**: item 0's branch construction already excludes both moe.py and
moe_sharding.py diffs from the K3 PR branch. Nothing else to do.

**Verify**: `git log --oneline -- torchtitan/models/common/` on the PR
branch shows no fork commits.

---

## 6. Formatting traps: lora.py ufmt + pyrefly version sweep

**What**: two known pre-commit landmines (both bit us on 07-25):

- `kimi_k3/lora.py` is not ufmt-clean at `20bd4f3a`; any `pre-commit run`
  reformats ~25 pre-existing lines.
- A locally installed `pyrefly` (version drift vs the pinned hook) strips
  `# pyrefly: ignore [...]` comments across ~20 unrelated core files.

**How**:

- Commit 1 of the PR branch: `pre-commit run --all-files`, commit ONLY the
  kimi_k3 formatting churn as "kimi_k3: ufmt reformat, no functional change".
- Functional content lands as commit 2+, so review diffs stay readable.
- Do not pip-install pyrefly manually; let pre-commit manage its pinned env.
  If a run ever touches files outside `experiments/kimi_k3/` + the registry
  line, revert the sweep -- never commit it.

**Verify**: `git show --stat <format-commit>` touches only
`experiments/kimi_k3/*`; a second `pre-commit run --all-files` is idempotent
(no new diff).

---

## 7. Flavor trimming (42 registered -> ~10 for upstream)

**What**: `config_registry.py` registers 42 `kimi_linear_*` flavors. The
scaling-law sweep and the 48B downscale matrix are fork-side research
artifacts; upstream wants a reviewable, CI-runnable set.

**How** -- keep on the PR branch:

- `debugmodel` family: `debugmodel`, `debugmodel8h`,
  `debugmodel_gated_lora`, `debugmodel_gated_qlora_mxfp4`,
  `debugmodel_k3faithful` (CI + parity carriers)
- 48B set: `48b_baseline`, `48b_block_attn_res`,
  `48b_block_attn_res_gated`, `48b_block_attn_res_gated_lora` (the real
  checkpoint targets)
- `2p8t_block_attn_res` (provisional, reconciles at the official config)

Drop from the PR branch (stay on the fork): the 194m/241m/296m/436m/447m/
528m scaling-law trios, `436m_block_attn_res_n4`, `447m_aligned_*`, the
`48b_*_e{8,16,32}` / `d{1024,1280}` / `L{24,32}_N8` downscale matrix, and
`528m_l16_*`. Delete their registration functions; prune the corresponding
generator entries in `model_configs.py` only where they become dead code
(`SCALING_LAW_TABLE` rows may stay if `build_kimi_linear_config` references
them -- deleting registrations is enough).

**Verify**: `pytest torchtitan/experiments/kimi_k3/tests/test_flavor_registry_sweep.py -x`
passes on the trimmed set; `grep -c "^def kimi_linear" config_registry.py`
reports the reduced count.

---

## 8. `TODO(kimi-parity)` on MoE score ordering: keep, and disclose

**What**: `model.py:1032-1035` -- upstream removed `score_before_experts`;
Kimi's reference applies router scores BEFORE the experts. Fixed upstream
ordering vs official 48B checkpoint is unverified until the SGLang-side A/B.

**How**: keep the TODO comment (repo rules allow TODOs for known
limitations). Add one line to the PR body's "Known limitations": "MoE router
score ordering follows upstream; parity vs the official checkpoint's
score-before-experts convention is pending an A/B (TODO(kimi-parity) in
model.py)."

**Verify**: TODO present in branch; limitation named in the PR body draft.

---

## 9. ASCII-only pass on everything items 1-3 rewrote

**What**: fork rule -- new/rewritten comments and docstrings must be ASCII
(`->` not arrows, `--` not em dashes). Pre-existing untouched Unicode stays.

**How**: after items 1-3, check only the lines this cleanup touched:

```bash
git diff upstream/main -- torchtitan/experiments/kimi_k3 | \
  grep "^+" | grep -nP "[^\x00-\x7F]"
```

Any hit inside a comment/docstring line that items 1-3 rewrote gets
ASCII-fied. (Hits in pre-existing lines that merely moved are left alone.)

**Verify**: the pipeline above returns no hits attributable to rewritten
comment lines.

---

## 10. Final gates (prove "zero functional change")

Run all four; all must pass before the PR opens:

1. `pre-commit run --all-files` -- clean, idempotent.
2. `pytest torchtitan/experiments/kimi_k3/tests -x` -- 64 passed / 6 skipped
   baseline (minus tests belonging to trimmed flavors, if any assert on
   registry counts).
3. `python -m compileall torchtitan/experiments/kimi_k3` -- clean.
4. GPU box, debug flavor, `--debug.seed 42 --debug.deterministic`:
   step-1 loss AND grad_norm bit-identical between `attention_residual_dev`
   @ `20bd4f3a` and the PR branch (comment/config surgery must not move a
   single bit; `scripts/loss_compare.py` for digits beyond stdout's five).

Item 7 is the only one allowed to change behavior at all, and only by
*removing* registry entries -- gate 4 runs on a kept flavor, so it is
unaffected.

---

## Out of scope for this checklist

- The RFC/issue text itself (RFC_K3_SUPPORT_DRAFT.md tracks that).
- 7.27 reconciliation actions (packed-MXFP4 guard flip, flavor regeneration,
  weight-sync name freeze) -- runbook lives in K3_RELEASE_IMPACT sec 4.
- PR16/17/18 filing decisions (need explicit go-ahead; kits in Raising_PRs/).

---

# Execution record + corrections (2026-07-27)

Worked through on box #2. Fork commit `301eab64`; PR branch
`kimi_k3_upstream_pr` = `daab44f9` (base) + `b5abac26` (trim). Seven things the
checklist got wrong or missed -- recorded here rather than silently patched.

## C1. Ordering: items 1-3 belong on the FORK, not on the PR branch

The checklist frames the whole exercise as PR-branch surgery. But items 1-3 fix
statements that are **wrong in the shipped folder**, so they are defects on
`attention_residual_dev` too, and two practical reasons force fork-first:

- the GPU gate (item 10.4) can only run where the torch-2.12 shims exist -- see
  C7; on the PR branch it dies after step 1;
- doing it twice would let the fork and the PR branch drift textually.

Done in that order: fork `301eab64` (44 replacements, verified) -> PR branch
imports the already-verified folder. **Item 7 stays PR-branch-only** -- the
scaling-law and downscale flavors are fork research assets.

## C2. Item 10 gate 2's baseline is wrong

Not "64 passed / 6 skipped". Measured: **85 passed + 66 subtests** on the fork,
and **85 passed + 35 subtests** on the trimmed PR branch -- the subtest count
tracks the flavor count because the registry sweep is parameterized over it, so
a *dropping* subtest number is expected after item 7, not a regression.
(A separate `tests/unit_tests/test_moe_routing_map_placement.py` adds 2, which
is where an "87" reading comes from; that test belongs to PR16, not here.)

## C3. Two more stale factual claims (same class as item 1)

Both found while verifying item 1, both fixed on the fork:

- `config_registry.py`: "AC off -- parallelize.py Phase 4c doesn't implement
  it". AC **is** implemented (`ac_config.build(dump_folder).apply(model)` in
  `parallelize_kimi_linear`) and verified under CP and TP. Off-by-default is a
  fine choice; only the false reason was rewritten.
- `model_configs.py` docstring: "This module does NOT yet return
  Trainer.Config nor ModelSpec. ModelSpec integration is Phase 4c ... use the
  Llama3-backed attn_res/config_registry.py flavors". Wrong three ways -- the
  module has a "Trainer.Config factories" section, the `BaseModel.Config` shim
  exists (`KimiLinearSpec` in model.py), and `attn_res/` no longer exists.

Lesson for the next pass: item 1 was not a one-off. Grep for *claims* ("not
supported", "doesn't implement", "does NOT yet", "blocked on"), not just for
phase tags.

## C4. Item 3's verify grep has a false positive

`grep -niE "HANDOFF|..."` also matches the ordinary English word in
`attn_res_model.py` ("the adapter's P2P handoff"). Use a case-sensitive
`HANDOFF` or anchor on `HANDOFF sec`.

## C5. Item 7's gate is incomplete, and its keep-list drops an ablation arm

- The named gate (`test_flavor_registry_sweep.py`) is dynamic and passes either
  way. But `test_kimi_model_spec.py` and `test_state_dict_adapter.py`
  **hard-import** trimmed flavors -- the suite fails at *collection*. Both were
  repointed at kept flavors of the same kind, including two LR assertions and
  two test names that carried the old sizes (Table-2 rows 2.99e-3 / 2.02e-3 ->
  the 48B row's 1.0e-3).
- The keep-list omits **`48b_full_attn_res`**, which is the ablation arm the
  Block-AttnRes claim is measured against. Kept it: 11 flavors, not 10.
- Bonus: the trim removes all 3 **N802** findings for free (the offenders were
  `d1280_e32_L24_N8`-style names whose capital L/N tripped the check).

## C6. NEW item 11 -- the folder is not flake8-clean, which blocks gate 1

**28 F401 + 2 F811 + 1 NU002** (plus E301/E241 fixed by ufmt) exist in the
folder. Identical count at `20bd4f3a` and `301eab64`, so the doc cleanup
introduced none, and no `# noqa` was removed. Two traps make this NOT a
mechanical sweep:

- the `F401`s in `__init__.py` / `config_registry.py` are largely
  **re-exports** (`build_kimi_linear_config`, `SCALING_LAW_TABLE`, ...).
  Deleting them changes the public surface -- use `__all__` or
  `# noqa: F401`.
- on the FORK, `N802` cannot be "fixed" by renaming: those function names ARE
  the `--config` names the ConfigManager resolves, so renaming is an API
  change. Suppress instead.

Needs its own commit and its own bit-identical gate, since unlike items 1-3 it
touches code lines. Left undone deliberately.

## C7. Gate 10.4 result, and the limit of what the PR branch can prove here

Gate 4 **passes on its own terms**: PR-branch step-1 on a single GPU is
`loss 7.63564 / grad_norm 3.3177`, bit-identical to the fork's cp1 baseline.

It then dies with `AttributeError: module 'torch.distributed' has no attribute
'set_timeout'` -- exactly compat shim #1 from item 4. So on torch 2.12 stable
the PR branch cannot be exercised past step 1, and fuller GPU validation of the
PR branch needs a nightly torch. The fork-side gate is the load-bearing one;
it was run across every touched path (seed 42, deterministic, all bit-identical
to the values recorded at `20bd4f3a`):

| cell | step-1 loss / grad_norm |
|---|---|
| tp2cp2pp2 (3D) | 7.62559 / 3.3709 |
| fsdp8 | 7.58611 / 1.8625 |
| 4D fsdp2tp2pp2ep2 | 7.64436 / 2.8285 |
| QLoRA packed tp2 | 7.58910 / 0.1672 |

## C8. The pyrefly hook cannot be satisfied locally at all

It is `language: system`. Absent -> the hook fails ("Executable pyrefly not
found"); installed from PyPI -> a version drifting from the pinned one strips
`# pyrefly: ignore [...]` across ~20 unrelated core files (item 6 warns about
the second half only). Both runs here used `SKIP=pyrefly-check`; it has to be
left to CI, and gate 1 should say so.
