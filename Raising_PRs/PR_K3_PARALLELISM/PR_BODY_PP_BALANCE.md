# PR title: [Kimi K3] Pipeline parallelism: PP ranks park saved activations on a peer through the Mooncake Transfer Engine

Follow-up to PR 4312 (it stacks on `pp_review3`). Branch `pp_balance_review1` on the fork (`54a9e81ee`, one commit on `pp_review3` `0e7cc5ea1`, upstream/main `6e2ac3dcd`); the old `pp_balance` commit `1c0c1416c` of the pre-review adapter, ported: `pp_balance.py` and its allocator test are the old files, the knobs became a record on `pipeline_kimi_k3`. Needs `mooncake-transfer-engine` (optional dependency, fla's standing) and, with a CUDA 13 torch, the `nvidia-cuda-runtime-cu12` wheel the engine's wheel links against, which the module preloads. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Adds `pp_balance` to `pipeline_kimi_k3`: the PP ranks named in `PPBalanceKnobs.pp_balance_source_ranks` wrap their stages' forwards in `saved_tensors_hooks` whose `pack` copies each tensor autograd saves into a pool on `pp_balance_dest_rank`'s GPU through the Mooncake Transfer Engine and frees the local storage, and whose `unpack` reads it back when backward needs it. Under interleaved 1F1B the resident activation load is uneven across PP ranks; this moves it from the heavy ranks to a light one. Copies are exact. The engine picks RDMA where an HCA exists and TCP where one does not, so the same wiring runs on a workstation and on a cluster.

### Design

- Every PP rank constructs the engine (the address-book exchange is a collective): the destination allocates and registers the pool, the sources register a staging buffer they funnel transfers through, so per-tensor register/unregister churn stays off the hot path. Tensors below `pp_balance_min_tensor_mib` stay local.
- The pool's allocator coalesces its free list: parked tensors are freed in whatever order backward reaches them, not in allocation order, and a bump pointer would run a long step out of pool while most of it is free (the CPU test).
- `K3_PPBAL_KEEP_LOCAL=1` runs every transfer but keeps the local storage alive: bitwise against the unbalanced run, which proves the transfer machinery value-exact. The balanced run itself moves in the last digits because freeing storage early changes the intra-step allocator layout, and the KDA backward's atomic reductions sum in address order.
- The knobs are a frozen record (`PPBalanceKnobs`) a recipe passes through `functools.partial(pipeline_kimi_k3, pp_balance=...)`, like the transport switch; the engine hangs off the schedule for the schedule's lifetime because it owns the registered buffers and sessions.

### Results

Running locally: pp2 x vp4 on the 30-layer debug flavor, rank 0 parking on rank 1 over TCP (no HCA on the box), once as designed and once with `K3_PPBAL_KEEP_LOCAL=1`, against the PR 4312 row; the rows follow with peak memory per rank.

| cell | balance | step 1 | step 3 | step 10 | peak memory rank 0 / rank 1 |
|---|---|---|---|---|---|
| pp2 x vp4 | off | | | | |
| pp2 x vp4 | rank 0 parks on rank 1 | | | | |
| pp2 x vp4 | same, `K3_PPBAL_KEEP_LOCAL=1` | | | | |

### Changed files

    torchtitan/models/kimi_k3/
      pp_balance.py         +360/-0  the engine, the pool allocator, the saved-tensors hooks, the knob record (new)
      parallelize.py        +14/-1   the pp_balance record on pipeline_kimi_k3; the engine installed after the split
    tests/unit_tests/cpu/
      test_kimi_k3_pp_balance_pool.py  +64/-0  the pool allocator (new)

### CI/CD Coverage

The allocator test runs in the CPU suite. The engine needs two ranks and the Mooncake wheel, so no integration cell; the switch is a recipe override.

--- PASTE END ---
