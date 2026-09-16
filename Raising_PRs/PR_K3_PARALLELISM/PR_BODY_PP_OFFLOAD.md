# PR title: [DO NOT review, stack on PR 4312] [Kimi K3] Pipeline parallelism: the rank store parks its blocks on pinned host memory

Draft stacked on PR 4312 (user, 2026-09-16: 4312 has been pending, the follow-ups go up as drafts on it). Review branch `pp_offload_review1` = `d49bb388b`: one commit on 4312's head `de6f29514` (= `k3_pp_text`), the integration tree's `c429df487` cherry-picked clean; the only edit since is the docstring line that claimed a result ("bitwise the on-device cache") now stating the mechanism ("the blocks are detached, so the copies are exact"). Diff audit (2026-09-16): 39 added lines, 30 code, 8 docstring, 0 comment; one new parameter on `pipeline_kimi_k3` and one on `RankStore`; no logbook paths, no measured values in source. CPU: `test_kimi_k3_pp_stage.py` (with the store's offload test), `test_kimi_k3_pp_layout.py`, `test_kimi_k3_stage_swap.py`, `test_kimi_k3_pp_exact_block_grads.py`: 17 passed on this head. GPU: the seeded pp2 x vp2 pair (switch off and on) on the 5060 is a smoke for rc and same-box bitwise; the body's table is measured on H100 before the draft leaves DO NOT review. The old table below the paste section came from the 09-05 stack on the previous adapter and is kept for the record only.

To file: create the fork branch `k3_pp_offload` from `pp_offload_review1` and open the PR against `main` with base commits showing 4312's diff too (as 4380 and 4381 do), title as above, body from the paste section.

--- PASTE BEGIN ---

## Summary

Add `attn_res_cache_offload` to `pipeline_kimi_k3`: with it on, the blocks a rank keeps for its later stages (`RankStore`) are parked on pinned host memory from the stage that commits them until a later stage on the rank reads them, instead of staying on the device for that whole span.

- `RankStore(offload=...)` (`kimi_k3/pipeline_stage.py`): `put` copies each block to a pinned host tensor on the current stream, `blocks` copies it back to the device it saw when a later stage assembles its stack.
- `pipeline_kimi_k3(attn_res_cache_offload=...)` (`kimi_k3/parallelize.py`): the switch, passed the way the transport switch is, `functools.partial(pipeline_kimi_k3, attn_res_cache_offload=True)` as the `pipelining_fn` of a recipe.
- A CPU test of the store's offload path (pass-through on a CPU store; the parked copy pinned and the read-back equal under CUDA).

## Design

Every stored block is detached (its gradient travels through the store's deposits), so the round trip changes values nowhere. Both copies are `non_blocking` on the current stream, so stream order alone serializes the D2H of the commit and the H2D of the read; nothing changes in the stage, `_assemble` still checks the held set against the routing tables and stacks what `blocks()` returns. A store on CPU has nowhere to park and stores the tensor as it is.

The switch lives on `pipeline_kimi_k3` next to `attn_res_cache` because it is a property of the rank store the pipelining entry builds, not of the model.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_kimi_k3_stage_swap.py tests/unit_tests/cpu/test_kimi_k3_pp_exact_block_grads.py -q` (17 passed)
- pp2 x vp2 and pp8 x vp4 on the debug model with the switch off and on, same seed checkpoint and batch, one warm inductor cache: the loss is expected identical on every step, since the parked copy is exact; the table goes here once measured on H100.

--- PASTE END ---

Record, not for pasting: the 09-05 measurement on `pp_offload_review1` `240c3205b` (the previous adapter, 33-layer flavor, 5060 Ti): pp2 x vp4 and pp8 x vp4 bitwise with the switch off and on over 10 steps (`12.41967` / `7.47862` / `3.42131` and `12.41967` / `7.51825` / `3.37366`); peak memory unchanged at that scale (a block is 256 x 1024 x 2 bytes).
