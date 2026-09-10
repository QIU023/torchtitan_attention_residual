# Dynamic vision CP on the new tree (2026-09-10)

Port of the multimodal CP draft (PR 4380, `k3_cp_mm`, the report's section 5.2.3 "dynamic CP": one large image split along its patch dimension across a sub-CP group, keys and values gathered inside it) onto `k3_int_20260910`, on top of the DEP port (`DEP_NEW_TREE_2026-09-10.md`).

## What moved, and where it landed on the new tree

- Planning stays pure: `torchtitan/models/kimi_k3/vit_cp_plan.py` (`row_partition` on merge-row blocks, `subgroup_layout`, `balance_images` LPT, `classify` at 256 patches), tested on CPU (`tests/unit_tests/cpu/test_kimi_k3_vit_cp_plan.py`, 7 tests).
- The partitioned attention is a Kimi K3 subclass of the shared tower attention: `KimiK3VisionCPAttention` in `torchtitan/models/kimi_k3/vision_encoder.py`, gather-KV through `torch.distributed.nn.functional.all_gather` (differentiable), a non-prefix padded-key mask, position tables (learned absolute embedding and 2-D RoPE) built for the whole image and sliced to the rank's row band. `KimiK3VisionEncoder.forward` takes `cp_plan`; the shared `VisionAttention`/`VisionTransformerBlock` in `models/common` only pass it through (k2.5 ignores it). The registry builds the K3 tower with the CP attention.
- The model side lives in the CP-aware `encode_images` that DEP already routes every encode through: `_encode_images_partitioned` classifies, lays out the sub-groups, cuts the pixels into row bands (every frame of a video), pads, calls the tower with a `CPPatchPlan`, all-gathers the merged tokens, and encodes the small images replicated. Sub-groups that have no image run a placeholder pass so the collectives stay matched. A rank whose batch has no image runs the tower on a placeholder and ties it in with `add_zero_valued_dependency` (`torchtitan/distributed/fsdp.py`, the old tree's FSDP dependency guard) so the FSDP all-gathers stay matched too.
- Sub-CP groups are process groups and cannot be created per batch, so `parallelize_kimi_k3` builds one per divisor of the CP size up front (`_build_cp_subgroups`, rank lists all-gathered so every rank walks the same list) and stores the dict on the model.
- Knob: `dynamic_cp_min_patches` on the K3 model config (256).

## Results (33-layer debug model, 4096 tokens per step in 256-token micro-batches, spmd_types, one seed checkpoint `.mx3_seeds_main33`; the debug batch holds one image above the threshold, so at cp2 it is split over the pair)

| cell | loss 1 | loss 2 | loss 3 | tower |
| --- | ---: | ---: | ---: | --- |
| dp1 | 12.40087 | 10.55432 | 7.71281 | whole |
| cp2 Ulysses MLA + KCP, before the port (`K3_INT_20260910.md`) | 12.39174 | | | replicated |
| cp2 Ulysses MLA + KCP, after the port | 12.41139 | 10.48488 | 7.39567 | one image over 2 ranks (log: "Dynamic CP: 1 large image(s) of 1 over 1 sub-CP group(s) of 2 rank(s)") |
| cp2 all-gather MLA + KCP, after the port | 12.41139 | 10.46471 | 7.43285 | same |

The two CP flavours read the same step-1 loss because the tower path is the same in both and the text-side flavour only diverges from step 2. The step-1 move against the replicated tower (12.39174 to 12.41139, 0.16%) is the question the next section answers directly, since a loss at init cannot tell rounding from a wrong slice.

## The partitioned tower against the replicated tower (two ranks, one image over both)

`matrix_scripts/../vitcp_probe.py` (scratch): the debug tower (8 layers, width 512), the same seed on both ranks, the replicated forward and backward on the whole image against the partitioned path through `_encode_images_partitioned` with a fixed random weighting of the merged tokens as the loss. The sum over ranks of the partitioned parameter gradients is compared with the replicated gradient (every rank scores the whole gathered output, so the gather's transpose sums the ranks' contributions; the loss is divided by the rank count to make the two comparable).

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

In fp32 the partition is the replicated computation to rounding, forward and backward, on every layout the planner produces; the bf16 rows are the dtype's own noise (the fp32 rows are the control), which is where the 0.16% step-1 move of the cp2 cells comes from. Commit on `k3_int_20260910`: see the branch log ("dynamic context parallelism for the vision tower").
