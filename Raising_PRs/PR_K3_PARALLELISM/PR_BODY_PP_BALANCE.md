# PR title: [DO NOT review, stack on PR 4312] [Kimi K3] Pipeline parallelism: PP ranks park saved activations on a peer through the Mooncake Transfer Engine

Draft stacked on PR 4312 (user, 2026-09-16). Review branch `pp_balance_review1` = `977bb6be4`: one commit on 4312's head `de6f29514`, the integration tree's `a15e60514` cherry-picked (one positional conflict in `parallelize.py`, the `pipeline_kimi_k3` signature and docstring; this stack carries only `pp_balance`, not the offload switch of the sibling draft). Edits since the integration version, from the diff audit: the two environment switches are gone (`K3_PPBAL_DEBUG`, a per-tensor debug print, and `K3_PPBAL_KEEP_LOCAL`, a numerics-isolation mode that ran every transfer but kept the local storage; its comment narrated a measurement), and the contiguity comment states the constraint without the finding. Audit: 380 added lines, 268 code, 43 docstring, 25 comment before the trims; new file `pp_balance.py` (the engine, the pool allocator, the hooks, the knob record); no logbook paths in source. CPU: `test_kimi_k3_pp_balance_pool.py` and the PP tests, 21 passed on this head. GPU: the two-rank pp2 x vp2 pair (rank 0 parks on rank 1 over TCP) on the 5060 is a smoke; the H100 table comes before the draft leaves DO NOT review. The Mooncake wheel (`mooncake-transfer-engine` 0.3.13) is a run-time import, not a torchtitan dependency; a box without it fails at `install_pp_balance` with the package named.

To file: fork branch `k3_pp_balance` from `pp_balance_review1`, PR against `main`, title as above, body from the paste section. Independent of the offload draft; both stack on 4312.

--- PASTE BEGIN ---

## Summary

Add `pp_balance` to `pipeline_kimi_k3`: the PP ranks named in `PPBalanceKnobs.pp_balance_source_ranks` wrap their stages' forwards in `saved_tensors_hooks` whose `pack` copies each tensor autograd saves into a pool on `pp_balance_dest_rank`'s GPU through the Mooncake Transfer Engine and frees the local storage, and whose `unpack` reads it back when backward needs it. Under interleaved 1F1B the resident activation load is uneven across PP ranks; this moves it from the heavy ranks to a light one. Copies are exact.

- `PPBalanceEngine`, `_PoolAllocator`, `PPBalanceKnobs`, `install_pp_balance` (`kimi_k3/pp_balance.py`, new): the engine over the Transfer Engine, the pool's allocator, the knob record, and the installer that wires the hooks onto the schedule's stages.
- `pipeline_kimi_k3(pp_balance=...)` (`kimi_k3/parallelize.py`): the record a recipe passes through `functools.partial`, like the transport switch; the engine is installed after the split and hangs off the schedule for the schedule's lifetime, since it owns the registered buffers and sessions.
- A CPU test of the pool allocator.

## Design

Every PP rank constructs the engine, because the address-book exchange is a collective: the destination allocates and registers the pool, the sources register a staging buffer they funnel transfers through, so per-tensor register and unregister calls stay off the hot path. Tensors below `pp_balance_min_tensor_mib`, non-contiguous tensors and tensors larger than the staging buffer stay local. The engine picks RDMA where an HCA exists and TCP where one does not, so the same wiring runs on a workstation and on a cluster.

The pool's allocator coalesces its free list: parked tensors are freed in whatever order backward reaches them, not in allocation order, and a bump pointer would run a long step out of pool while most of it is free. Only contiguous tensors are parked: a normalized layout keeps the values but changes the strides the backward kernels tile by.

The transport stays in the model folder, as the AttnRes block transport does; the package is imported at `install_pp_balance`, so a run without `pp_balance` never needs it.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_balance_pool.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_kimi_k3_stage_swap.py tests/unit_tests/cpu/test_kimi_k3_pp_exact_block_grads.py -q` (21 passed)
- pp2 x vp2 on the debug model, rank 0 parking on rank 1, against the same cell without `pp_balance`, same seed checkpoint and batch, one warm inductor cache; the engine's counters (tensors parked and fetched back, bytes, tensors kept local) and rank 0's peak memory; the table goes here once measured on H100.

--- PASTE END ---

Record, not for pasting: the 09-05 measurement on `pp_balance_review1` `3bb3fc6cf` (previous adapter, 33-layer flavor, 5060 Ti, TCP): pp2 x vp4 bitwise off and on over 10 steps (`12.41967` / `7.47862` / `3.42131`), 1,440 tensors and 1,760 MiB parked and fetched back, 11,840 tensors below the 1 MiB floor kept local, rank 0's peak unchanged at that scale (about 176 MiB a step against 14 GiB). The keep-local mode that isolated the transfer from the allocator effects was a local diagnostic and is not in this branch.
