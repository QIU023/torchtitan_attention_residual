# PR title: [kimi_k3] quantile balancing for the MoE router bias

Branch `k3_qb` (`160812fa1`, on upstream/main `13da2d77`). Evidence: overnight ledger + `LORA_QLORA_QAT_EVIDENCE`. Paste between the markers into the PR body.

--- PASTE BEGIN ---

### Summary

Adds quantile balancing for the MoE router bias. Before this change the bias update adds a fixed sign-step nudge from the current step's load counts (core's default load-balancing hook); after it the bias is solved at each optimizer step by inverting an accumulated per-expert load histogram at the balanced quantile -- same bias tensor, same hook point, no model change. Kimi K3 (report sec 2.3) runs sparsity beyond where the nudge still balances.

### Design

- One preallocated `(num_experts, num_bins)` histogram per MoE layer on the expert-bias device; the forward hook is a single `add_` and `step()` zeroes in place.
  - Branch-free and allocation-free on purpose: selective activation checkpointing replays the forward, and a lazily allocated buffer changes the op sequence between forward and recompute -- that failure was observed, and with this shape SAC on/off prints identical losses.
- The solver registers as a `post_optimizer_build_fn` replacing the default hook; `kimi_k3_debugmodel_qb` enables it on the debug flavor.
- 14 CPU unit tests: histogram accumulation, CDF inversion, the bias solve against a brute-force reference, in-place zeroing, the SAC-identical op sequence.

### Results

Measured at the pre-rebase tip d5393c240; the rebase onto current main is conflict-free, but main's KDA kernel now requires SM100-class hardware the measuring box does not have.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_qb \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 2
```

Training loss on `kimi_k3_debugmodel_qb`, one seed, warmed compile cache:

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.58962 | 8.46094 | 4.01340 |
| dp2 | 12.58743 | 8.19710 | 3.62665 |

### Changed files

    torchtitan/components/
      quantile_balance.py           +374  the histogram hook and the solver (new)
    torchtitan/models/kimi_k3/
      config_registry.py             +15  the qb flavor
    tests/unit_tests/cpu/
      test_quantile_balance.py      +273  (new)

### CI/CD Coverage

The 14 unit tests are CPU and run in the default suite; no GPU cell is added.

--- PASTE END ---
