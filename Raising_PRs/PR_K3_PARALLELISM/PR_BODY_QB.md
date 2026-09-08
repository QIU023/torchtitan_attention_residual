# PR title: [Kimi K3] Quantile balancing for the MoE router bias

PR 4412 (draft, head `k3_qb` = `3c9cef31a`, the pre-rebase content; the GitHub title still carries the "[DO NOT review, pending EP PR merging]" prefix). The content to raise is `qb_review3` = `db65fc9f2`: the two QB commits plus the registration-comment/counter fix, on upstream/main `f6b9152e9` (#4505, 2026-09-08); clean cherry-pick; QB unit tests 14/14 in venv_bfx9 (the tip needs `torch_remat`, installed). `qb_review2` (`d0d75fa8b` on `aecbb8199`) and `qb_release` (`a4658eefe`) are superseded. Evidence and the probe scripts: `phase13_k3like_48b_posttrain/QB_EVIDENCE_2026-09-08.md`, `matrix_scripts/qb_probe/`. Raising is the user's: force-push `qb_review3` to `k3_qb`, drop the title prefix, undraft, paste the body below.

--- PASTE BEGIN ---

### Summary

Adds quantile balancing for the MoE router bias. Before this change the bias update adds a fixed sign-step nudge from the current step's load counts (core's default load-balancing hook); after it the bias is solved at each optimizer step by inverting an accumulated per-expert load histogram at the balanced quantile -- same bias tensor, same hook point, no model change. Kimi K3 (report sec 2.3) runs sparsity beyond where the nudge still balances.

### Design

- One preallocated `(num_experts, num_bins)` histogram per MoE layer on the expert-bias device; the router forward hook is a single `add_` and `step()` zeroes in place.
  - Branch-free and allocation-free on purpose: selective activation checkpointing replays the forward, and a lazily allocated buffer changes the op sequence between forward and recompute -- that failure was observed, and with this shape SAC on/off prints identical losses.
- `kimi_k3_debugmodel_qb` installs the solver as the model spec's `post_optimizer_build_fn`, the slot core's sign-rule registration occupies by default; the trainer calls one such function, so under the flavor the sign rule is not registered and the solver is the only writer of `expert_bias_E` (it also resets the MoE's per-step token counter in core's place).
  - Expert parallelism is not a prerequisite: the hook reads the router's scores and the bias, both held in full on every rank with or without EP, so the change is independent of how the experts are sharded.
- Histograms are summed once over the `loss` mesh (dp_replicate x dp_shard x cp), stacked across layers into one collective, and not over tp: the router's scores are Replicate under TP and a second sum would scale the histogram by tp. This is the group core's hook reduces over for the same case.
- Full activation checkpointing replays the router forward, so the hook counts every token twice; the solve inverts `cumsum / total`, and a uniformly doubled histogram gives the same bias. Core's hook needs a `// 2` for the same replay.
- 14 CPU unit tests: histogram accumulation, CDF inversion, the bias solve against a brute-force reference, in-place zeroing, the SAC-identical op sequence.

### Results

Debug flavor on `f6b9152e9` plus the two commits, seed 42, one seed checkpoint shared by every configuration, 30 measured steps, each hook pair on one inductor cache (the sign-step control compiles fresh, the quantile-balancing cell starts from that cache), `partial_dtensor` backend: under the default `spmd_types` the multimodal debug flavor stops at `preprocess_inputs` on main (`pixel_values` and `grid_thw` have no input layout; the declarations PR supplies one).

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_qb \
  --debug.seed 42 --debug.deterministic --training.steps 30 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
  --parallelism.spmd_backend partial_dtensor --checkpoint.enable
```

Expert load per MoE layer (tokens routed to each expert, summed over the loss mesh), as the coefficient of variation over the 32 experts and the largest and smallest expert's load relative to the mean, averaged over the 23 MoE layers and over ten-step windows; sign step is core's rule at `load_balance_coeff` 1e-3, the flavor's default:

| config | hook | cv steps 1-10 | 11-20 | 21-30 | max/mean 21-30 | min/mean 21-30 | bias range at step 30 |
|---|---|---|---|---|---|---|---|
| dp1 | sign-step (main) | 1.37 | 1.46 | 1.33 | 5.0 | 0.03 | [-0.046, 0.014] |
| dp1 | quantile balancing | 1.00 | 0.83 | 0.59 | 2.6 | 0.18 | [-0.290, 0.277] |
| dp2 | sign-step (main) | 1.46 | 1.64 | 1.48 | 5.5 | 0.03 | [-0.046, 0.013] |
| dp2 | quantile balancing | 1.06 | 0.88 | 0.62 | 2.7 | 0.13 | [-0.202, 0.279] |
| dp2 x ep2 | sign-step (main) | 1.42 | 1.48 | 1.27 | 4.6 | 0.03 | [-0.042, 0.016] |
| dp2 x ep2 | quantile balancing | 1.14 | 0.97 | 0.74 | 3.1 | 0.08 | [-0.183, 0.273] |

Every rank holds the same solved bias after every step (all-gathered and compared, ep2 included). With the router frozen (`lr` 0, dp1, 15 steps, the same data through unchanging weights) the sign step moves the load cv from 1.07 to 0.84 and quantile balancing to 0.42 (0.93, 0.64, 0.58 after steps 2-4); an offline solve on two dumped batches of the real scores puts the exact-quantile update at cv 0.53 / 0.40 / 0.34 / 0.28 / 0.21 after 1 / 2 / 3 / 5 / 10 solves on one batch and 0.31 on the next batch, with the histogram within 0.02 of the exact quantile at 512 bins, so the rate is that of the per-step coordinate update on this router, not of the estimator.

Training loss, the same runs (the LR schedule spans the 30 steps):

| config | hook | step 1 | step 3 | step 10 | step 30 |
|---|---|---|---|---|---|
| dp1 | sign-step (main) | 12.51887 | 7.22217 | 3.05235 | 1.67791 |
| dp1 | quantile balancing | 12.51887 | 7.33664 | 3.03395 | 1.39994 |
| dp2 | sign-step (main) | 12.52560 | 7.29986 | 3.02123 | 0.61437 |
| dp2 | quantile balancing | 12.52560 | 7.35559 | 3.05933 | 1.31018 |
| dp2 x ep2 | sign-step (main) | 12.52372 | 7.50218 | 3.00328 | 1.05916 |
| dp2 x ep2 | quantile balancing | 12.52372 | 7.28418 | 2.94781 | 0.61779 |
| dp1, other cache | sign-step (main) | 12.51887 | 7.11252 | 3.02868 | 1.31970 |
| dp1, other cache | quantile balancing | 12.51887 | 7.35326 | 2.86286 | 1.19053 |

The last two rows are the dp1 pair on a second fresh cache: step 1 is bitwise across caches, steps 3 / 10 / 30 move by 1.5 / 0.8 / 21 percent (sign step) and 0.2 / 5.6 / 15 percent (quantile balancing) from the autotune picks alone, while the load windows reproduce within 0.1 (sign step 1.32 / 1.41 / 1.39, quantile balancing 0.94 / 0.91 / 0.63) -- the load table is the reproducible quantity, the late loss is not. Step 1 is bitwise on every pair (the bias is first written at the end of step 1). From step 2 the two hooks route the same tokens differently -- 96 percent of the tokens' routed expert sets differ at step 2 and 99.7 percent at step 10, on every configuration -- so the later loss values are two different runs and are shown only as "the flavor still trains", not compared. Step 1 differs between dp2 and dp2 x ep2 by 2e-3 because the expert kernels round differently, on both hooks alike.

Loss and total gradient norm, maximum relative difference over steps 1-5, from the same runs (the format of the DSV3 MTP pipeline table); the expert-parallel pair moves as much under either hook and no more than the same cell on two caches, and its step-1 loss pair is the same with and without the hook (the bias is 0 at step 1), so it is expert parallelism's, not the hook's:

| pair | hook | step-1 loss A / B | max rel loss diff, steps 1-5 | max rel grad-norm diff, steps 1-5 |
|---|---|---|---|---|
| dp2 vs dp2 x ep2 | quantile balancing | 12.52560 / 12.52372 | 3.0e-2 | 2.4e-1 |
| dp2 vs dp2 x ep2 | sign-step (main) | 12.52560 / 12.52372 | 3.2e-2 | 1.0e-1 |
| dp1 vs dp1, other cache | quantile balancing | 12.51887 / 12.51887 | 1.2e-2 | 4.8e-2 |
| dp1 vs dp1, other cache | sign-step (main) | 12.51887 / 12.51887 | 7.4e-2 | 2.4e-1 |


### Changed files

    torchtitan/components/
      quantile_balance.py           +407/-0  the histogram hook and the solver (new)
    torchtitan/models/kimi_k3/
      config_registry.py            +16/-0   the qb flavor
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
