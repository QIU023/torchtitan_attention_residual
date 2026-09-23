# 2026-09-23 overnight: the four draft branches rebased onto latest main + 4312 / 4639

Goal: rebase the PP-stacked drafts (balance #4764, offload #4765, DEP #4381) onto latest main and
4312's latest head, and dynamic CP (#4380) onto main and 4639's latest head; push the drafts
(they are draft PRs); do not touch the published PR branches (#4312's `k3_pp_text`, pytorch #194033).

## Bases

- upstream main moved 2 commits past `7349a2282` (4312's base): `892ecfed0` (DistMuon BlockShard) and
  `b64103072` (#4596, the K3 debug recipe switched to per-head DistMuon). #4312 is `mergeable_state: dirty`
  against it: both add a cell to the B200 lists (`tests/integration_tests/b200.py`,
  `tests/unit_tests/cpu/test_integration_test_definitions.py`, `torchtitan_recipes/tests/b200.py`).
- `pp4312_on_main_0923` (scratch, local only): 4312's 7 commits replayed on `b64103072` = `99894da64`,
  plus one fixup commit `3b2746ce0` "kimi_k3: the pipeline cells train with AdamW under the per-head DistMuon
  recipe". The 4312 tree is otherwise byte-identical to `3f9201ea0` outside the three cell lists
  (checked with a diff of diffs). Patch of the fixup: `Raising_PRs/PR_K3_PARALLELISM/0001-kimi_k3-the-pipeline-cells-train-with-AdamW-under-th.patch`.
- `cp4639_on_main_0923` (scratch): 4639's stack (`010b9015a` refactor cp shard + `77c7f0185` support CP for
  KimiK3, itself re-based by fegin onto `7349a2282` on 09-22) replayed on `b64103072` = `7d5ddb602`, no conflicts.

## What #4596 breaks (for the user's own 4312 rebase, and for fegin's 4639)

`kimi_k3_debugmodel()` now builds a DistMuon optimizer whose param groups are regex patterns and
`OptimizersContainer._build_param_groups` raises "pattern ... matched no parameters" per model part.
- 4312's `kimi_k3_debugmodel_pp4_vp4` puts the head alone on the last stage: the Muon pattern matches nothing
  there. `fsdp2_tp2_ep2_pp2` also runs TP, which DistMuon refuses (K2.5's PP cell docstring says so).
  Fix used on the scratch base: both cells set `config.optimizer = default_adamw(lr=8e-4)`. The alternative is a
  core change (skip an empty group under PP), which is upstream's call.
- DEP's `kimi_k3_debugmodel_pp4_vp2_vit_dep` (tower + embedding alone on the first stage): same, fixed in the
  cell commit of the DEP draft.
- 4639's `kimi_k3_debugmodel_mm_allgather_kv_cp2` fails with "Muon compute layout for parameter
  'layers.0.delta_attention.q_proj.weight' declares no axis in storage mesh ['dp_shard_cp']; declared axes:
  ['dp_shard']": the recipe builds the optimizer for a default ParallelismConfig and the cell sets cp=2
  afterwards. Not ours; the smoke of the CP draft runs that cell on AdamW.

## The drafts (new heads; old heads pinned on the fork as backup/<branch>_pre_20260923)

| draft | review branch | new head | on | conflicts |
|---|---|---|---|---|
| balance #4764 | pp_balance_review1 | ff4ca4044 | pp4312_on_main_0923 | none |
| offload #4765 | pp_offload_review1 | 949bc6c1c | pp4312_on_main_0923 | none |
| DEP #4381 | dep_review1 | e4025f559 | pp4312_on_main_0923 | pipeline_parallel/__init__.py (round 4's map guard next to the stage-class choice); the cell lists |
| dynamic CP #4380 | cpmm_review1 | 923f8bd47 | cp4639_on_main_0923 | a port, not a merge: pre-#4810 `parallelize.py` is gone, so the sub-CP group build moved into `KimiK3Model.parallelize` (`_build_cp_subgroups` now in model.py); the class is `MultimodalModel`-based and 4639's vision-bank gather takes our `encode_images` output as the bank |

Each draft's diff against its base is unchanged by the rebase except in the files above (diff of diffs).

## Checks on the 5060 box (venv_bfx9, torch 2.15.0.dev20260906; KDA guard lifted locally in the smoke worktrees, never committed)

- CPU pytest (test_kimi_k3_*, test_pipeline_parallel, test_integration_test_definitions, test_config_manager):
  balance 109 passed, offload 104, DEP 127, CP 70 (CP needs `PYTHONPATH=/workspace/pylib/attn_gym_main`
  for `attn_gym.linear.context_parallel`; `transformers` was installed into venv_bfx9 for the flux import).
- pyrefly on each draft's changed files: the same error count as on the old heads (7 / 14 / 1), nothing new.
- GPU: base `kimi_k3_debugmodel_pp4_vp4` 3 steps on 4 GPUs (after the AdamW fixup);
  DEP cell 3 steps on 4 GPUs (bubble plan: 2/2 planned encodes in a bubble, backward at planned slots).
- GPU, pp2 x 1F1B x 4 microbatches, `--debug.seed 42 --debug.deterministic`, scratch recipes
  (`scratchpad/goal0923/probe_recipes.py`, never committed): plain, balance (source rank 0 parks one stage's saved
  tensors on rank 1, 2048 MiB span, first tensor parked 1024 KiB) and offload (`attn_res_cache_offload=True`) give
  the same loss and grad norm on every step: 7.97090 / 7.92552 / 7.51493, grad norm 2.2500 / 2.4062 / 2.8438.
  These three ran the recipe's DistMuon: with matrices on both stages the param groups are fine.
- GPU, CP: 4639's `kimi_k3_debugmodel_mm_allgather_kv_cp2` does not run on the attn-gym 0.0.10 it pins
  (`ContextParallelRouting.from_fragments` is missing; 0.0.10 has `ContextParallelPlan.from_fragments(...).routing(device, conv_history=)`).
  With that one-call shim applied locally in `cp_kda.py` (uncommitted) and AdamW, the cell runs 3 steps on 2 GPUs
  (loss 12.494 / 11.656 / 10.649); the same cell with `dynamic_cp_min_patches=1` partitions the image
  ("Dynamic CP: 1 large image(s) of 1 over 1 sub-CP group(s) of 2 rank(s)") and runs 3 steps (12.508 / 11.148 / 9.930;
  unseeded, so no identity claim). The CP probe is not blocking, per the user.

## Pushed (force with lease, old heads pinned)

| PR | branch | head | backup |
|---|---|---|---|
| #4764 | k3_pp_balance | ff4ca4044 | backup/k3_pp_balance_pre_20260923 = 76f7d90df |
| #4765 | k3_pp_offload | 949bc6c1c | backup/k3_pp_offload_pre_20260923 = f87e74c7b |
| #4381 | k3_pp_mm | e4025f559 | backup/k3_pp_mm_pre_20260923 = 191b31bc0 |
| #4380 | k3_cp_mm | 923f8bd47 | backup/k3_cp_mm_pre_20260923 = e0f1b8569 |

`k3_pp_text` stays `3f9201ea0`; pytorch `get-total-norm-dtype` stays unpushed (the user checks it first).

## Left uncommitted on this box, on purpose

- `torchtitan/models/kimi_k3/kda.py` guard lift `(12, 0)` in /tmp/wt_ppbal, /tmp/wt_ppoff, /tmp/wt_depnew, /tmp/wt_cpnew and the scratch base worktree.
- `torchtitan/models/kimi_k3/cp_kda.py` attn-gym 0.0.10 shim in /tmp/wt_cpnew.
- venv_bfx9 gained `transformers` (flux import in the definitions test); attn-gym 0.0.10 sits in `scratchpad/goal0923/attn_gym_010` for PYTHONPATH.
