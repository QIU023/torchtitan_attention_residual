# PR title: [Kimi K3] An FSDP2 x EP2 integration cell

Branch `k3_ep_cell` on the fork (`5d53688b8`, one commit on upstream/main `9b5f60c40`). Paste between the markers.

--- PASTE BEGIN ---

### Summary

The expert-parallel support merged without a cell that shards the experts: the B200 list runs the multimodal debug model on two ranks under FSDP2 only. This adds the same configuration with `expert_parallel_degree=2`, the smallest cell that exercises the routed experts' own mesh, next to the existing one.

### Results

The multimodal debug model, `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256, on main's arithmetic; step 1 differs from dp2 by 9e-5 because the expert kernels round differently.

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp2 (the existing cell) | 2 | 12.53137 | 7.31248 | 3.15823 |
| dp2 x ep2 (this cell) | 2 | 12.53146 | 7.20212 | 3.10296 |

### Changed files

    torchtitan_recipes/tests/b200.py       +7/-0   kimi_k3_debugmodel_mm_fsdp2_ep2
    tests/integration_tests/b200.py         +6/-0   the kimi_k3_mm_fsdp2_ep2 cell

### CI/CD Coverage

One B200 cell on two GPUs.

--- PASTE END ---
