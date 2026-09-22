# PR title: [DO NOT review, stack on PR 4312] [Kimi K3] Pipeline parallelism: the rank store parks its blocks on pinned host memory

Draft stacked on PR 4312. Review branch `pp_offload_review1` = `f87e74c7b`, one commit on 4312's round-3 head `78be13c96`, which is itself five typed commits on upstream main `7349a2282` (re-homed 2026-09-22 after #4810 removed the per-model `parallelize.py` and moved the pipeline into `kimi_k3/pipeline_parallel/`). The fork branch `k3_pp_offload` was force-pushed to this head on 2026-09-22; its previous head `d49bb388b` sat on the old 4312 head `de6f29514` and all four of its hunks landed on files that round 3 had moved.

What changed since `d49bb388b`, from the 2026-09-22 review of the draft branches: the store takes its device explicitly instead of inferring it from the first block it is handed, so a store that has already parked one does not start moving every later one; the switch is a field on `KimiK3Model.Config`, because after #4810 the model owns its pipelining and there is no `pipelining_fn` for a recipe to wrap, which also answers the finding that nothing in the branch could turn the switch on; pairing it with the whole-stack transport now raises instead of being silently ignored; and the CPU test skips with `@unittest.skipUnless(torch.cuda.is_available())` rather than returning early, so a CPU run no longer reports a pass for a path it did not execute.

Diff: 4 files, 62 added lines, 38 of them production non-blank. CPU on this head: 51 passed (`pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline"`), pyrefly clean on `torchtitan/models/kimi_k3`.

GPU (2026-09-22, 4 x RTX 5060 Ti, torch 2.15.0.dev20260906+cu130, seed 42, five steps, one warm inductor and Triton cache lineage shared by both cells): `kimi_k3_debugmodel_pp4_vp4` with the switch off and on read `8.20651 / 7.39539 / 5.77873 / 5.39160 / 4.95098` on steps 1-5, identical on every step, peak memory 0.68 GiB on both. The control's first, cold-cache run read `5.77892 / 5.38883 / 4.95183` from step 3; that is the autotune confound, not the switch, and the pair above is the rerun on the shared warm cache. This is a smoke for exactness and rc, not a memory measurement: a block is 256 x 1024 x 2 bytes at this scale. The hundred-step table on H100 comes before the draft leaves DO NOT review.

--- PASTE BEGIN ---

## Summary

Add `attn_res_cache_offload` to `KimiK3Model.Config`: with it on, the blocks a rank keeps for its later pipeline stages are parked on pinned host memory from the stage that commits them until a later stage on the rank reads them, instead of holding device memory for that whole span.

- `PPRankLocalCache(device=...)` (`kimi_k3/pipeline_parallel/stage.py`): `put` copies each block to a pinned host tensor on the current stream, `blocks` copies it back to the given device when a later stage assembles its stack.
- `attn_res_cache_offload` (`kimi_k3/model.py`): the switch, read by `pipeline_kimi_k3` from the model config. Paired with `attn_res_cache=False` it raises, because the whole-stack transport keeps nothing between hops to park.
- A CPU test of the store's two paths: pass-through when no device is given, and under CUDA the parked copy pinned on the host and the read-back equal to what was put.

## Design

Every stored block is detached, since its gradient travels through the store's deposits, so the round trip changes no value. Both copies are `non_blocking` on the current stream, so stream order alone serializes the device-to-host copy of the commit against the host-to-device copy of the read. Nothing changes in the stage: `_assemble` still checks the held set against the routing tables and stacks what `blocks()` returns.

The device is a constructor argument rather than something inferred from the first block, so a store handed a host tensor does not decide it should be moving them.

## Test plan

- `pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline" -q` (51 passed)
- `kimi_k3_debugmodel_pp4_vp4` with the switch off and on, same seed and batch on one warm compile cache: the loss is identical on every step, as the parked copy is exact. Measured on 4 x RTX 5060 Ti; the hundred-step table on H100 goes here.

--- PASTE END ---

Record, not for pasting: the 09-05 measurement on `pp_offload_review1` `240c3205b` (the previous adapter, 33-layer flavor, 5060 Ti): pp2 x vp4 and pp8 x vp4 bitwise with the switch off and on over 10 steps (`12.41967` / `7.47862` / `3.42131` and `12.41967` / `7.51825` / `3.37366`); peak memory unchanged at that scale (a block is 256 x 1024 x 2 bytes).
