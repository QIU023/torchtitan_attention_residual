# PR title: [Draft] [Kimi K3] quantile balancing for the MoE router bias

Branch `k3_qb` (`3c9cef31a`, base `30eb5e502`). Merges clean onto upstream/main `1dcb14a0c`; the 14 CPU tests pass on the merged tree. Results are a placeholder: the draft goes up without numbers, the measured rows get pasted in later. Paste between the markers into the PR body.

--- PASTE BEGIN ---

### Summary

Adds quantile balancing for the MoE router bias. Before this change the bias update adds a fixed sign-step nudge from the current step's load counts (core's default load-balancing hook); after it the bias is solved at each optimizer step by inverting an accumulated per-expert load histogram at the balanced quantile -- same bias tensor, same hook point, no model change. Kimi K3 (report sec 2.3) runs sparsity beyond where the nudge still balances.

### Design

- One preallocated `(num_experts, num_bins)` histogram per MoE layer on the expert-bias device; the router forward hook is a single `add_` and `step()` zeroes in place.
  - Branch-free and allocation-free on purpose: selective activation checkpointing replays the forward, and a lazily allocated buffer changes the op sequence between forward and recompute -- that failure was observed, and with this shape SAC on/off prints identical losses.
- The solver registers as a `post_optimizer_build_fn` replacing core's sign-step hook at the same optimizer-step pre-hook, and writes the same `expert_bias_E`; `kimi_k3_debugmodel_qb` enables it on the debug flavor.
  - Expert parallelism is not a prerequisite: the hook reads the router's scores and the bias, both held in full on every rank with or without EP, so the change is independent of how the experts are sharded.
- Histograms are summed once over the `loss` mesh (dp_replicate x dp_shard x cp), stacked across layers into one collective, and not over tp: the router's scores are Replicate under TP and a second sum would scale the histogram by tp. This is the group core's hook reduces over for the same case.
- Full activation checkpointing replays the router forward, so the hook counts every token twice; the solve inverts `cumsum / total`, and a uniformly doubled histogram gives the same bias. Core's hook needs a `// 2` for the same replay.
- 14 CPU unit tests: histogram accumulation, CDF inversion, the bias solve against a brute-force reference, in-place zeroing, the SAC-identical op sequence.

### Results

<placeholder: dp1 / dp2 rows on the branch merged with current main, steps 1 / 3 / 10, one seed, warmed compile cache>

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_qb \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 2
```

### Changed files

    torchtitan/components/
      quantile_balance.py           +393/-0  the histogram hook and the solver (new)
    torchtitan/models/kimi_k3/
      config_registry.py            +15/-0  the qb flavor
    tests/unit_tests/cpu/
      test_quantile_balance.py      +273/-0  (new)

### CI/CD Coverage

The 14 unit tests are CPU and run in the default suite; no GPU cell is added.

--- PASTE END ---
