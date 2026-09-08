# PR title: [Kimi K3] Quantile balancing for the MoE router bias

PR 4412 (draft, head `k3_qb` = `3c9cef31a`, the pre-rebase content; GitHub title still carries the "[DO NOT review, pending EP PR merging]" prefix from before the EP merge of 09-03). The content to raise is the review branch `qb_review2` = `d0d75fa8b`: the two QB commits (`00599e8db` + `d0d75fa8b`) on upstream/main `aecbb8199`, the last commit before #4484's BFX9 gate; +687/-0 over three files; QB unit tests 14/14 on that tree (2026-09-08); merges cleanly onto upstream/main `f6b9152e9` (#4505, 2026-09-08). `qb_release` (`a4658eefe`, the 09-05 rebase) is superseded. Raising is the user's: force-push `qb_review2` to `k3_qb`, drop the title prefix, undraft, paste the body below.

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

Training loss on `d0d75fa8b` (the two commits on upstream/main `aecbb8199`, the last commit before #4484's Blackwell BFX9 gate, which this box's PyTorch does not pass; on `0902c7a24`, `47ec648b4` and `a4658eefe` the same rows reproduced to the digit), rerun on 2026-09-07 with Attention Gym at upstream/main `b19162e` and its SM100/SM103 guard in `kda.py` lifted locally, every row of the table on one compile cache (the quantile cells start from the control cells' inductor cache), under `partial_dtensor`: on main the multimodal debug flavor has no input layout for `spmd_types`, which the declarations PR supplies; the control rows are bitwise the declarations PR's dp rows on the same gym), one seed (`--debug.seed 42 --debug.deterministic`, one seed checkpoint per flavor, each cell run twice on an idle box and the second run read); the control rows are the same tree and seed with core's sign-step hook (`kimi_k3_debugmodel`). Step 1 is identical to the digit: the bias is only rewritten at the optimizer step, so the first forward cannot differ; the runs separate from step 2 on. The `dp2 x ep2` rows are where the balancing is exercised across expert shards.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_qb \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
  --parallelism.spmd_backend partial_dtensor
```

The backend flag is required on this base: under the default `spmd_types` the multimodal debug flavor stops at `preprocess_inputs` (`pixel_values` and `grid_thw` have no input layout on main; the declarations PR supplies one). The table's cells start from one seed checkpoint shared by every configuration (`--checkpoint.enable`, written once by the dp1 cell), so the rows differ by the parallelism and the hook alone.

| config | hook | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | sign-step (main) | 12.52977 | 7.36833 | 2.91045 |
| dp1 | quantile balancing | 12.52977 | 7.38270 | 3.00769 |
| dp2 | sign-step (main) | 12.53137 | 7.25082 | 3.15411 |
| dp2 | quantile balancing | 12.53137 | 7.30862 | 3.20259 |
| dp2 x ep2 | sign-step (main) | 12.53146 | 7.13441 | 3.09174 |
| dp2 x ep2 | quantile balancing | 12.53146 | 7.57599 | 3.16632 |
| dp1, other cache | sign-step (main) | 12.52977 | 7.27107 | 2.98077 |
| dp1, other cache | quantile balancing | 12.52977 | 7.30620 | 3.11376 |

Each pair (both hooks of one config) runs on one inductor cache: the sign-step control compiles fresh and the quantile-balancing cell starts from that cache, so the two rows differ by the hook alone. The last two rows are the same two dp1 cells on `0902c7a24` (2026-09-03), each compiled on its own fresh cache: step 1 is bitwise across caches and rebases, steps 3 and 10 move by 1.3 and 2.4 percent (control) and 1.0 and 3.5 percent (quantile balancing) from the autotune picks alone -- the floor the in-table hook differences (0.2 to 6 percent at step 3, 1.6 to 3.3 percent at step 10) are read against; the pairs on `a4658eefe` (09-05) and `d0d75fa8b` (09-07) agree to the digit. The bar claimed is step 1 bitwise on every pair: the hook first writes the bias at the end of step 1, so step 1 must and does agree. Step 1 differs from dp2 by 9e-5 under EP because the expert kernels round differently, on both hooks alike.

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

--- RAISE CHECKLIST (user) ---

1. `git push origin qb_review2:k3_qb --force-with-lease` (the PR head moves from `3c9cef31a` to `d0d75fa8b`; a rebase, so a force push).
2. Retitle the PR to the line at the top of this file and mark it ready for review.
3. Paste the body between the PASTE markers (one line per paragraph, as written).
4. CI: the 14 CPU tests run in the default suite; the GPU matrix above is not a CI cell.
