# PR title: [DRAFT] Dynamic context parallelism for the Kimi K3 vision encoder

Draft stacked on the text-side CP PR #4639. Review branch `cpmm_review1` = `e0f1b8569`, one commit on 4639's head `0088c9b70`, which is 4639's own two commits on main `1e4b1f686`. The fork branch `k3_cp_mm` was force-pushed to this head on 2026-09-22; its previous head `774e0b9b5` sat on 4639's older head `e06dcbee3`, which that PR force-replaced.

The old first commit is dropped. It added `exclude_fqn_prefixes` to `ContextParallelTransform` so the K3 CP recipes could keep the tower's local attention; 4639's new head already skips an inner attention whose parent is not a `BaseAttention.Config`, and the tower's `VisionAttention.Config` is not one, so the transform no longer reaches the tower and there is nothing left to exclude.

What changed in the remaining commit, from the 2026-09-22 review of the draft branches:

- The CP group comes from the current SPMD mesh, `spmd_mesh_group(MeshAxisName.CP)`, not from an attribute written onto the model at parallelize time with a `noqa`. The old shape failed silently: a missing attribute fell back to the replicated tower. A two-rank gloo check confirms the group resolves inside `spmd_local_context("dp")`, which is the context the tower encodes in and the reason the attribute was there.
- The sub-CP groups, which must still be built up front because building a group is collective, are handed to the model through `set_vision_cp_subgroups` and built with `dist.new_subgroups_by_enumeration` instead of nested `new_group` loops over every CP group in the world. On eight ranks over a 2 x 4 mesh each rank gets the whole CP group for one sub-group, its pair for two, and itself for four.
- The partitioned attention reuses the base class's `_qkv` and keeps its recompute regions, which it previously dropped, so the path that exists to save memory no longer ignores the model's recompute policy. The gather sits outside every region, because replaying one would issue the collective a second time.
- The type-checking rule for torch's differentiable all-gather is registered when context parallelism is on, through `enable_vision_cp_typecheck_rules()`, rather than at import; the rule is process-wide and the function it names is torch's private one.
- Comment plus docstring is 157 of 597 added production non-blank lines, 26.3 percent, from 274 of 706, 38.8 percent.

CPU on this head: 8 passed (`pytest tests/unit_tests/cpu -k "kimi_k3 or vit_cp"`), pyrefly clean on `torchtitan/models/kimi_k3`.

GPU (2026-09-22, 2 x RTX 5060 Ti, torch 2.15.0.dev20260906+cu130, attn-gym 0.0.9.dev73, seed 42; the local SM120 guard lift on `kda.py` is not in the diff): `kimi_k3_debugmodel_mm_allgather_kv_cp2` runs two steps, and with the threshold lowered so the debug batch's image qualifies the partition engages, `Dynamic CP: 1 large image(s) of 1 over 1 sub-CP group(s) of 2 rank(s)`, and the step completes. The stock recipe's images are below the 256-patch default, so the default cell does not exercise the partition on the debug data. Both runs need `--debug.no-spmd-typechecking`: with type checking on, 4639's own head raises `SpmdTypeError` in `kda.py`'s gated norm on this box, with none of this branch's changes present, which is checked against a clean `0088c9b70` worktree.

The partitioned-versus-replicated comparison below has NOT been re-measured on this head: the 09-10 scratch probe that produced it is gone, and the rewrite (`matrix_scripts/vitcp_probe.py`) hangs at the partitioned call. The computation it measures is unchanged by the re-home, whose diff is the group source, the sub-group API, the recompute regions and the docstrings, but the numbers are owed on this head together with the hundred-step H100 table before the draft leaves DRAFT.

--- PASTE BEGIN ---

Draft, stacked on the text-side CP PR (#4639): the diff tab shows that PR's content too, so review the last commit. It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3, both halves: a large image is partitioned along its patch dimension across the ranks of a sub-CP group and the tower's attention gathers keys and values inside that group; the CP group is divided into sub-groups over which several large images are distributed longest-first, so the key exchange stays local instead of growing with the group.

### Design

- `vit_cp_plan.py` is pure planning (`row_partition`, `subgroup_layout`, `balance_images`, `classify`), so the decisions are tested without ranks. Cuts land on merge-row blocks, the unit the projector's spatial merge needs whole; a video's band is the same rows of every frame, because the projector's temporal pooling spans frames.
- `KimiK3VisionCPAttention`, a Kimi K3 subclass of the shared vision attention, gathers keys and values through the differentiable collective, masks padded key positions (not as a prefix: the padding is interleaved per frame), and builds both position tables for the whole image and slices them to the rank's band, since both index from row 0. It keeps the base class's recompute regions, with the gather outside them. The shared attention and block only pass `cp_plan` through: it travels as an argument because activation checkpointing recomputes the forward from its saved arguments.
- `encode_images` classifies the batch's images, lays out the sub-groups, cuts the pixels, calls the tower with a `CPPatchPlan`, all-gathers the merged tokens, and encodes the small images replicated. Sub-groups with no image run a placeholder pass so the collectives stay matched, and a rank whose batch holds no image ties its placeholder in with a zero-valued dependency so the FSDP collectives stay matched too.
- The sub-CP process groups are built once at parallelize time, one layout per divisor of the CP size, and handed to the model; the CP group itself comes from the current SPMD mesh at use time.
- No new flavor and no new default: the path is live whenever `context_parallel_degree > 1` and the batch holds an image at or above `dynamic_cp_min_patches` (256).

### Results

The partitioned tower against the replicated tower, two ranks, one image over both. Measured 2026-09-10 on 8 x RTX 5060 Ti on the integration tree `k3_int_20260910`, which carried the same partition code; the debug tower is 8 layers at width 512, the same seed on both ranks, and the summed parameter gradients are compared against the replicated gradient. Not yet re-measured on this head.

| case | dtype | forward max diff / scale | grad rel, median | grad rel, worst |
| --- | --- | ---: | ---: | ---: |
| one image 32x32 | fp32 | 1.0e-6 | 1.6e-6 | 4.3e-6 |
| one image 30x32 (padded band) | fp32 | 1.1e-6 | 1.7e-6 | 5.2e-6 |
| video t=2, 32x32 | fp32 | 1.2e-6 | 1.5e-6 | 1.2e-5 |
| large + small (one partitioned, one replicated) | fp32 | 1.6e-6 | 1.8e-6 | 4.6e-6 |
| one image 32x32 | bf16 params | 8.5e-3 | 1.2e-2 | 2.4e-2 |
| one image 30x32 (padded band) | bf16 params | 1.3e-2 | 1.4e-2 | 2.1e-2 |
| video t=2, 32x32 | bf16 params | 1.0e-2 | 9.8e-3 | 1.8e-2 |
| large + small | bf16 params | 1.2e-2 | 1.4e-2 | 2.1e-2 |

In fp32 the partition is the replicated computation to rounding, forward and backward, on every layout the planner produces; the bf16 rows are the dtype's own noise, with the fp32 rows as the control.

On this head (2026-09-22, 2 x RTX 5060 Ti, seed 42, type checking off, see above): `kimi_k3_debugmodel_mm_allgather_kv_cp2` completes two steps, and with the threshold lowered so the debug batch's image qualifies, the partition engages over both ranks and the step completes. The training table, on H100 and over a hundred steps, is owed.

### Test plan

- `pytest tests/unit_tests/cpu -k "kimi_k3 or vit_cp" -q` (8 passed): the planner's row partition, the merge-kernel constraint, the sub-group layout, the longest-first balance and the threshold.
- `kimi_k3_debugmodel_mm_allgather_kv_cp2` and `kimi_k3_debugmodel_mm_ulysses_cp2`, which run the partitioned tower whenever the batch holds an image above the threshold.
- The partitioned tower against the replicated one on two ranks, fp32 as the control and bf16 alongside: the table above, to be re-measured on this head.


### Changed files

    torchtitan/models/kimi_k3/
      vit_cp_plan.py        the pure planners
      vision_encoder.py     CPPatchPlan, the position-table slicing, the padded-key mask, gather-KV attention
      model.py              encode_images, _encode_images_partitioned, the sub-group seam, the plain-grad boundary
      parallelize.py        the sub-CP group layouts, built once
      __init__.py           the tower's attention builds the CP-capable class
    torchtitan/models/common/vision_encoder.py   the cp_plan pass-through
    torchtitan/distributed/fsdp.py               add_zero_valued_dependency
    tests/unit_tests/cpu/test_kimi_k3_vit_cp_plan.py (7 tests)

--- PASTE END ---
