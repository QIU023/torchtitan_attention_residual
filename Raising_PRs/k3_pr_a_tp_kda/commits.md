# PR-A backing work and extraction recipe

**Base**: PR-4025's head, NOT `upstream/main`. 4025 adds the model and rejects TP
explicitly; this PR is what answers that rejection, so it has nothing to apply to until
4025 lands. Local tracking branch: `pr4025_latest`.

## Why there is no cherry-pick recipe

The other kits in this folder give backing commit hashes because their changes are
file-scoped. These four are not. All four features live in the same handful of files, and
of the last 60 commits touching `torchtitan/models/kimi_k3/`, **43 touch two or more of
them** (one touches 14). No commit is scoped to TP alone, so a cherry-pick sequence cannot
produce a reviewable single-feature branch. The split is by FUNCTION.

The four `k3_pr_*` branches are bookmarks at `a146d1bf2` (2026-08-05), created from
`attention_residual_dev` and periodically reset to its HEAD. They have never carried
isolated per-PR content and are not a starting point for filing -- treat them as markers
for "the dev state the PR body describes".

## Extraction list

From `torchtitan/models/kimi_k3/parallelize.py`:

| symbol | lines | note |
|---|---|---|
| `apply_tp_kimi_k3` | 577 | the plan itself |
| `_apply_tp_moonvit_mlp` | 92 | vision tower MLP |
| `_patch_fla_for_dtensor` | 117 | kernel-boundary shims |
| `_bind_fla_dtensor_shims` | 24 | per-instance binding |
| the `tp_enabled` branch of `parallelize_kimi_k3` | ~30 | call site |

Plus, in `moonvit.py`, the replicated-attention branch of `_attend` (94 lines total; the
TP-relevant part is the `grad_placements=[Replicate()]` drop-to-local path).

## Must NOT come along

* Anything reading `_cp_group` or `cp_context` (PR-D).
* `apply_ep_kimi_k3` and the EP prefetch in `apply_fsdp` (PR-B).
* `pipeline_adapter.py` in any form (PR-C).
* The `Partial()` -> `Replicate()` correction on the AttnRes pseudo-query gradient IS
  in scope and is one of the two defects the PR body cites; keep it.

## Verification

Matrix cells that must pass, multimodal `kimi_k3_debugmodel_report_arch` and the LoRA twin:
`tp2`, `tp4`, `fsdp2_tp2_cp2` (proves TP composes with CP), `ep2_fsdp2_tp2_cp2`.
Command: `matrix_scripts/run13_flav.sh` + `run_maxdeg.sh`. Reference tables:
`MATRIX_DEP_DYNCP_2026-08-10.md`.

## Status

Ready to extract; blocked only on 4025 landing.
