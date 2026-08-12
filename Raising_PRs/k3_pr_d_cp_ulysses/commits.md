# PR-D backing work and extraction recipe

**Base**: PR-4025's head (`pr4025_latest`). See PR-A's `commits.md` for why the four
`k3_pr_*` branches are bookmarks rather than content.

Read `DYNAMIC_CP_SCOPE_NOTE.md` in this folder first: it decides whether dynamic CP for
the vision tower ships with this PR or follows it.

## Extraction list

| file | symbol | lines |
|---|---|---|
| `parallelize.py` | `_build_cp_subgroups` | 55 |
| `model.py` | `_cp_all_to_all_headseq` | 41 |
| `model.py` | `KimiMLAAttention._forward_cp` (Ulysses head-sharding) | 107 |
| `model.py` | `KimiDeltaAttention._forward_kcp` (fla merged KCP) | 72 |
| `model.py` | `KimiDeltaAttention._forward_cp` (KDA dispatch + conv halo) | 116 |
| `moonvit.py` | `_attend` + `_attend_gather_kv` + `set_cp_patch_plan` | 176 |
| `multimodal_model.py` | `_encode_images_cp`, `_encode_images_dynamic_cp`, `_select_cp_shard`, `_cp_world_size` | 301 |
| `vit_cp_plan.py` | whole file | 301 |

~1180 lines outside `parallelize.py`. The vision half (`moonvit`, `multimodal_model`,
`vit_cp_plan`, ~780 of it) is separable from the text half and probably should be.

## The KDA claim to keep intact

We do NOT carry our own recurrence. Plain state summation is wrong for KDA because the
delta rule applies a token-dependent transition to the incoming state; the PR uses fla's
merged KCP, which is the implementation the K3 report cites. The evidence that the chunked
kernel is exact under sequence splitting is `matrix_scripts/kda_shape_independent_probe.py`
-- one rank, no process group, `fla`'s own `naive_recurrent_kda` as reference, with
realistic gates and l2-normalised q/k (the first version drove `g ~ 0` and unnormalised
q/k, blew the state to 1e17, and both implementations "agreed" on garbage).

## Must NOT come along

* `_dep_reject_cp` (8 lines) belongs with DEP, i.e. with PR-C's DEP split.
* The TP-side `grad_placements` correction in `_attend` (PR-A), though the two touch the
  same function and the extraction order matters: take PR-A's first.

## Verification

Cells: `cp2`, `cp4`, `fsdp2_tp2_cp2`, `ep2_fsdp2_tp2_cp2`, `fsdp2_pp2_cp2`,
`ep2_fsdp2_pp2_cp2`, with `KIMI_VIT_DYNAMIC_CP=1`. The three TP+CP cells were the recorded
defect until 2026-08-10 (`F54_CP_HANG_2026-08-11.md` for the FSDP unit hazard that a
wrapper-level alias fixed, and the replicated-attention fix for the defect itself).

## Status

Ready to extract; blocked only on 4025 landing. Decide the vision/text split first.
