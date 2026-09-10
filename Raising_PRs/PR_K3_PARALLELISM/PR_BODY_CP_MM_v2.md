# PR title: [DRAFT] Dynamic context parallelism for the Kimi K3 vision encoder

Branch `k3_cp_mm_v2` (= the published `k3_cp_text` head 61a73ca6c + one commit `121303718`). Replaces the old-tree head `k3_cp_mm` (a5339256e) of PR 4380 once the GPU cells on this branch are in (see `phase13_k3like_48b_posttrain/CPMM_NEW_TREE_2026-09-10.md`). File as DRAFT stacked on the text CP PR.

--- PASTE BEGIN ---

Draft, stacked on the text-side CP PR: the diff tab shows that PR's content too, so review the last commit. It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3, both halves: a large image is partitioned along its patch dimension across the ranks of a sub-CP group and the tower's attention gathers keys and values inside that group; the CP group is divided into sub-groups over which several large images are distributed longest-first, so the key exchange stays local instead of growing with the group.

### Design

- `vit_cp_plan.py` is pure planning (`row_partition`, `subgroup_layout`, `balance_images`, `classify`), so the decisions are tested without ranks. Cuts land on merge-row blocks, the unit the projector's spatial merge needs whole; a video's band is the same rows of every frame because the projector's temporal pooling spans frames.
- `KimiK3VisionCPAttention` (a Kimi K3 subclass of the shared vision attention) gathers keys and values through the differentiable collective, masks padded key positions (not a prefix: padding is interleaved per frame), and both position tables (the learned absolute embedding and the 2-D RoPE) are built for the whole image and sliced to the rank's band, since both index from row 0. The shared attention and block only pass `cp_plan` through: it travels as an argument because activation checkpointing recomputes the forward from its saved arguments.
- `encode_images` classifies the batch's images, lays out the sub-groups, cuts the pixels, calls the tower with a `CPPatchPlan`, all-gathers the merged tokens, and encodes the small images replicated; sub-groups with no image run a placeholder pass so the collectives stay matched. `_PlainGradBoundary` keeps the gather's transpose (a reduce-scatter with no DTensor sharding strategy) on plain tensors.
- Sub-CP process groups are built once at parallelize time for every divisor of the CP size, rank lists all-gathered so every rank walks the same list.
- No new flavor and no new default: the path is live whenever `context_parallel_degree > 1` and the batch holds an image at or above `dynamic_cp_min_patches` (256).

### Results

The partitioned tower against the replicated tower, two ranks, one image over both, fp32 (8-layer debug tower, the same seed on both ranks; the summed parameter gradients against the replicated gradient):

| case | forward max diff / scale | grad rel, median | grad rel, worst |
| --- | ---: | ---: | ---: |
| one image 32x32 | 1.0e-6 | 1.6e-6 | 4.3e-6 |
| one image 30x32 (padded band) | 1.1e-6 | 1.7e-6 | 5.2e-6 |
| video t=2, 32x32 | 1.2e-6 | 1.5e-6 | 1.2e-5 |
| large + small (one partitioned, one replicated) | 1.6e-6 | 1.8e-6 | 4.6e-6 |

With bf16 parameters the same cases read 0.9-1.3e-2 forward and 1-2e-2 in the gradients, the dtype's own noise (the fp32 rows are the control).

Integration tree (`k3_int_20260910`, 33-layer debug model, 4096 tokens per step, spmd_types, seeded; the debug batch holds one image above the threshold, split over the pair):

| cell | loss 1 | loss 2 | loss 3 |
| --- | ---: | ---: | ---: |
| dp1 | 12.40087 | 10.55432 | 7.71281 |
| cp2 Ulysses MLA + KCP, replicated tower (before) | 12.39174 | | |
| cp2 Ulysses MLA + KCP, partitioned tower | 12.41139 | 10.48488 | 7.39567 |
| cp2 all-gather MLA + KCP, partitioned tower | 12.41139 | 10.46471 | 7.43285 |

On this branch (`k3_cp_mm_v2`): dp1 and the two cp2 recipes -- (to be filled from `heads_verify`).

### Changed files

    torchtitan/models/kimi_k3/
      vit_cp_plan.py        the pure planners
      vision_encoder.py     CPPatchPlan, the position-table slicing, the padded-key mask, gather-KV attention
      model.py              encode_images, _encode_images_partitioned, the plain-grad boundary
      parallelize.py        the pre-built sub-CP group layouts
      __init__.py           the tower's attention builds the CP-capable class
    torchtitan/models/common/vision_encoder.py   the cp_plan pass-through
    torchtitan/distributed/fsdp.py               add_zero_valued_dependency
    tests/unit_tests/cpu/test_kimi_k3_vit_cp_plan.py (7 tests)

--- PASTE END ---
