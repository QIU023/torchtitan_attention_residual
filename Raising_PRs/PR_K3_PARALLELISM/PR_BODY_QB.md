# PR title: [Kimi K3] Quantile balancing for the MoE router bias

PR 4412. Branch `qb_release` on the fork (`47ec648b4`: the `k3_qb` content rebased onto upstream/main `6e2ac3dcd` on 2026-09-04, plus a typing commit for pyrefly; `0902c7a24` on post-expert-parallel main was the previous head); clean rebase, no conflicts. The PR branch `k3_qb` is synced to it only on the user's approval. Paste between the markers into the PR body; drop the old "[DO NOT review, pending EP PR merging]" prefix from the title.

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

Training loss on `0902c7a24` (the rerun on `47ec648b4` is running locally and replaces these rows), one seed (`--debug.seed 42 --debug.deterministic`, one seed checkpoint per flavor, each cell run twice on an idle box and the second run read); the control rows are the same tree and seed with core's sign-step hook (`kimi_k3_debugmodel`). Step 1 is identical to the digit: the bias is only rewritten at the optimizer step, so the first forward cannot differ; the runs separate from step 2 on. The `dp2 x ep2` rows are where the balancing is exercised across expert shards.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_qb \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2
```

| config | hook | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | sign-step (main) | 12.52977 | 7.27107 | 2.98077 |
| dp1 | quantile balancing | 12.52977 | 7.30620 | 3.11376 |
| dp2 | sign-step (main) | 12.53137 | 7.31248 | 3.15823 |
| dp2 | quantile balancing | 12.53137 | 7.19897 | 3.24552 |
| dp2 x ep2 | sign-step (main) | 12.53146 | 7.20212 | 3.10296 |
| dp2 x ep2 | quantile balancing | 12.53146 | 7.67547 | 3.17317 |

The dp1 and dp2 rows reproduce the pre-rebase measurement to the digit; the ep2 rows are new. Step 1 differs from dp2 by 9e-5 under EP because the expert kernels round differently, on both hooks alike.

### Changed files

    torchtitan/components/
      quantile_balance.py           +393/-0  the histogram hook and the solver (new)
    torchtitan/models/kimi_k3/
      config_registry.py            +15/-0   the qb flavor
    tests/unit_tests/cpu/
      test_quantile_balance.py      +273/-0  (new)

### CI/CD Coverage

The 14 unit tests are CPU and run in the default suite; no GPU cell is added.

--- PASTE END ---
