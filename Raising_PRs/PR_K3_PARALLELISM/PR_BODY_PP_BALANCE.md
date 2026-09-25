# PR 4764 body (draft PR, `k3_pp_balance`), rewritten 2026-09-25

Title, updated for the new stack: `[DO NOT review, stack on #4765] [Kimi K3] cross PP ranks activation memory LoadBalancing` (was `[DO NOT review, stack on K3 text PP PR #4312] ...`).

## Status (not for pasting)

- **Branch.** On 2026-09-25 the user said to sync the draft PRs directly. `k3_pp_balance` was force-pushed from `005cf4aee` to `pp_balance_review1` `46692171b`; the old head is pinned as `backup/k3_pp_balance_pre_20260925`. GitHub shows 18 commits, 26 files: everything in #4765 plus the balance commit.
- **CPU on this head:** the command in the test plan, 104 passed, no skips. The remote test ran, not skipped; mooncake is installed in `/workspace/venv_bfx9`.
- **GPU smoke, 8 x RTX 5060 Ti, identity only.**
  - pairs 3 to 7, 1 to 6 and 2 to 5, 8 stage micro-batches each, seq 512 without AC;
  - loss and grad norm bitwise with the same group's baseline over 3 steps (`PP_ACTIVATION_STORAGE_REPORT_2026-09-25.md` §3).
  - The box has no RDMA NIC, so the pools sat in the destinations' pinned host memory behind mooncake's TCP transport.
- **Open before an H100 run: the RDMA path is unverified.**
  - `RemoteBackend` initializes the engine with protocol `"tcp"` in every case.
  - It places the pool and staging buffer on the GPU when `get_local_topology()` lists HCAs, and its own comment says the TCP transport serves host memory only.
  - On an RDMA box those two choices disagree. Either the protocol follows the placement, or the placement stays on host until an RDMA run shows device memory works.

--- PR 4764 body: PASTE BEGIN ---

## Summary

Heavy Kimi K3 pipeline ranks can park the saved activations they hold longest in a pool on a lighter PP rank through mooncake's transfer engine, as the Kimi K3 report balances activations across PP ranks.

- `RemoteBackend` (`torchtitan/distributed/activation_storage.py`): a second backend of #4765's activation storage. Each destination registers one pool for the run, split evenly among its sources; transfers run in order on the storage stream through a registered staging buffer (`transfer_write_on_cuda`, `transfer_read_on_cuda`).
- `PPBalanceKnobs` (`kimi_k3/pipeline_parallel/activations.py`) and `KimiK3Model.Config.pp_balance` (`kimi_k3/model.py`): the (source, destination) pairs, the stage micro-batches each source moves, the lead, and the pool and staging sizes; off by default.

## Design

Balancing is a storage policy of the same activation storage, not a separate system: the pack hook routes a save to the remote backend, and the layer-ahead prefetch reads it back. With both switches on, the plan gives the remote backend the longest-held stage micro-batches and host offload the next ones.

Under interleaved 1F1B the resident activations decrease as the PP rank increases, so the pairs name which ranks give and which take. Every rank of the pipeline group builds the backend, because the segment exchange is collective. A destination keeps its backend for the run, since the transfer engine unregisters the pool when its owner goes away. A full pool or staging buffer keeps the tensor on the device.

The backend lives next to the host backend in `torchtitan/distributed`. The pairs are K3 pipeline knobs, since only the schedule knows which ranks are heavy.

## Relation to #4765

Stacked on #4765, which is stacked on #4312.

## Test plan

- `pytest tests/unit_tests/cpu/test_activation_storage.py tests/unit_tests/cpu/test_activation_storage_pool.py tests/unit_tests/cpu/test_kimi_k3_pp_offload.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_config_manager.py tests/unit_tests/cpu/test_no_new_cli_options.py -q` (104 passed)
  - `test_activation_storage_pool.py`: the pool allocator's first fit and coalescing.
  - `test_kimi_k3_pp_offload.py`: rank 0 parks on rank 1 through mooncake's TCP transport under the real Interleaved1F1B schedule; loss and every gradient bitwise.
- 8 x RTX 5060 Ti smoke, pp8 x vp2 on a local debug flavor: ranks 3, 1 and 2 park 8 stage micro-batches each on ranks 7, 6 and 5; loss and grad norm bitwise with `pp_balance` off over 3 steps. The box has no RDMA NIC, so the pools sit in pinned host memory; peak memory per PP rank on H100 goes here once measured.

--- PASTE END ---
