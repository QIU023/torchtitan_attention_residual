# PR title: [Kimi K3] Pipeline parallelism: the rank store parks its blocks on pinned host memory

Follow-up to PR 4312 (it stacks on `pp_review3`). Branch `pp_offload_review1` on the fork (`240c3205b`, one commit on `pp_review3` `a3be242bf`, the 33-layer debug model and the pp8 x vp4 recipe, upstream/main `6e2ac3dcd`; `20d83a8cb` was the head on `fe34932ee`); the old `attn_res_cache_offload` commit `e8897274d` of the pre-review adapter, ported onto the stage class. 25 PP CPU tests pass (one added), pyrefly 0 on the touched files. The two flag-on cells reran on this head on 2026-09-05, bitwise the rows below. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Adds `attn_res_cache_offload` to `pipeline_kimi_k3`. Before this change the blocks a rank keeps for its later stages (`RankStore`) stay on the device from the stage that commits them until the last stage on the rank has read them; after it, with the switch on, `RankStore.put` copies each block to pinned host memory on the current stream and `RankStore.blocks` copies it back when a later stage assembles its stack. Every stored block is detached (its gradient travels through the store's deposits), so the round trip is value-identical and the loss is bitwise the on-device store. A recipe turns it on with `functools.partial(pipeline_kimi_k3, attn_res_cache_offload=True)`, the way the transport switch is set.

### Design

- The store keeps the device it saw and hands back device copies from `blocks()`; both copies are `non_blocking` on the current stream, so stream order alone serializes the D2H of the commit and the H2D of the read.
- Nothing changes in the stage: `_assemble` still checks the held set against the routing tables and stacks what `blocks()` returns.
- A CPU tensor is stored as it is (a store on CPU has nowhere to park), which is what the unit test on a CPU-only runner exercises; the CUDA half of the test checks the parked copy is pinned and the read-back equal.

### Results

pp2 x vp4 and pp8 x vp4 on the 33-layer debug flavor with the switch on, against the same cells with it off (the PR 4312 matrix, same seed and batch, the same Attention Gym checkout `7c83f6c`); both are bitwise, and the log line "block transport delta with rank store on pinned host memory" is in every switched-on run. Peak memory is rank 0's; at this scale a block is 256 tokens x 1024 x 2 bytes, so the store is a few MB against activations of GiB and the saving does not show.

| cell | store | step 1 | step 3 | step 10 | peak memory |
|---|---|---|---|---|---|
| pp2 x vp4 | device | 12.41967 | 7.47862 | 3.42131 | 13.98 GiB |
| pp2 x vp4 | pinned host | 12.41967 | 7.47862 | 3.42131 | 13.98 GiB |
| pp8 x vp4 | device | 12.41967 | 7.51825 | 3.37366 | 8.67 GiB |
| pp8 x vp4 | pinned host | 12.41967 | 7.51825 | 3.37366 | 8.67 GiB |

### Changed files

    torchtitan/models/kimi_k3/
      pipeline_stage.py     +25/-3  RankStore(offload=...): park on put, copy back on blocks
      parallelize.py        +12/-3  the attn_res_cache_offload switch on pipeline_kimi_k3
    tests/unit_tests/cpu/
      test_kimi_k3_pp_stage.py  +19/-0  the store's offload path (CPU pass-through; pinned parking under CUDA)

### CI/CD Coverage

The store test runs in the CPU suite; its CUDA half runs wherever the GPU unit tests do. No integration cell: the switch is a recipe override.

--- PASTE END ---
