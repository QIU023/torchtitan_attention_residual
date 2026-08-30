# PR title: [DRAFT] Dynamic context parallelism for the Kimi K3 vision encoder

Branch `k3_cp_mm` (base = the `k3_cp_text` review head + one commit). File as DRAFT stacked on the text CP PR; do not undraft before that PR lands and the tables are re-measured on this branch. Paste between the markers.

--- PASTE BEGIN ---

Draft, stacked on the text-side CP PR: the diff tab shows that PR's content too, so review only the last commit (`kimi_k3: dynamic CP for the vision encoder`). It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3, both halves: a single large image is partitioned along the PATCH dimension across the ranks of a sub-CP group, with attention gathering k/v across the group (gather-KV), and each CP group is divided into sub-CP groups with the large images distributed across them so the communication fraction does not grow with the group. Images below the partition threshold, or whose grid height does not divide the merge kernel, stay whole and are encoded replicated, which is exactly the text PR's behavior for every image.

### Design

- `vit_cp_plan.py` is pure planning -- no collectives, no tensors in signatures -- so the scheduling decisions (`row_partition`, `subgroup_layout`, `balance_images`, `classify`) are testable without spawning ranks.
  - The merge kernel constrains where a partition may cut: the safe unit is a merge-row block (`kh` grid rows), and a video is cut as "rows r0..r1 of EVERY frame" because the projector's temporal mean spans all frames.
- Position tables (the learned absolute embedding AND the 2-D RoPE cache) are built for the WHOLE image and sliced to each rank's band; building them from the shard's own grid gives every rank rank 0's positions -- measured at 2.3e-3 step-1 loss drift before this was carried, far too large for a reduction-order effect.
- Padded key positions are masked out of attention, and the mask is NOT a prefix: padding is interleaved per frame, so a prefix mask would admit frame 0's padding and mask frame 1's real keys whenever t > 1 and some rank runs short.
- Sub-CP process groups are pre-built at wiring time for every divisor layout of `cp_size`: `new_group` must be called by every rank with the same lists in the same order, so a per-batch call is exactly the mismatch that hangs.
- `_PlainGradBoundary` keeps the tower plain in both directions: the gather's transpose is a reduce-scatter with no DTensor sharding strategy, and neither `to_local()` nor `grad_placements` can say "do not re-wrap".
- No new flavor and no new config default: the path is live whenever `context_parallel_degree > 1` and the batch holds an image at or above `dynamic_cp_min_patches` (default 256); the debug dataset's images reach that threshold, so the debug matrix exercises it as-is.
- One common touch: `VisionAttention.forward` and `VisionTransformerBlock.forward` gain a pass-through `cp_plan` argument (ignored in common, consumed by the subclass), because activation checkpointing recomputes forwards from saved arguments and module state set around the call is gone by recompute time.

### Results

Draft placeholder: the tables below were measured on the integration tree this commit is extracted from (trees `2f9dd3098` and `77a298ac5`, 8x RTX 5060 Ti, seed 42, `--debug.deterministic`, steps 1/3/10 protocol), not on this branch; they will be re-measured on this branch before the draft is undrafted.

Sequence 512, one large image per stream: cp2 differs from dp1 at step 2 by 2.62e-4 and cp4 by 3.52e-3; sequence 1024 taken to cp8: 9.47e-3 / 9.46e-3 / 2.22e-3 for cp2/cp4/cp8, no trend with degree. Every CP cell logs its actual partition ("N large image(s) over M sub-CP group(s)"), so an inert path fails loudly rather than producing a plausible table.

### Changed files

    torchtitan/models/kimi_k3/
      vit_cp_plan.py        +170  the pure planners: row bands, sub-group layout,
                                  image balancing, the partition threshold
      vision_encoder.py     +260  CPPatchPlan, the position-table slicing, the
                                  padded-key mask, gather-KV attention
      model.py              +258  _encode_images: sub-group dispatch, empty-pass
                                  padding, the plain-grad boundary; the image-free
                                  placeholder keeps FSDP collectives matched
      parallelize.py         +59  the pre-built sub-CP group layouts
      __init__.py            +14  the tower's attention builds the CP-capable class
    torchtitan/models/common/vision_encoder.py  +8  the cp_plan pass-through

### CI/CD Coverage

None added in this draft; planned before undrafting: CPU unit tests for the pure planners (`row_partition` band invariants, `subgroup_layout`, the non-prefix key mask), which need no ranks.

--- PASTE END ---
