# Quantile balancing: what the PR can and cannot claim (2026-09-08)

Prompted by the review of `PR_BODY_QB.md` against the tech report (arXiv 2607.24653v2, sec 2.3.3 and
App. C/D) and the maintainer's standard: a loss table cannot validate a load-balancing rule, and the
step-3/10 gaps in the old table were unexplained. Tree: `qb_review3` = the two QB commits plus a
comment/counter fix on upstream/main `f6b9152e9` (#4505, 2026-09-08), run in `venv_bfx9` (torch
2.15.0.dev20260906, the main tip's `torch_remat` installed), debug flavor, `partial_dtensor` backend
(the multimodal debug flavor has no `spmd_types` input layout on main; the declarations PR supplies it),
seed 42, one seed checkpoint, 30 measured steps, one inductor cache per hook pair. Probe code in
`matrix_scripts/qb_probe/` (a router forward hook counting routed ids, an optimizer pre-hook reducing
them over the loss mesh; run-worktree flavors, not committed).

## 1. Implementation against the report

| report | implementation (`torchtitan/components/quantile_balance.py`) | verdict |
| --- | --- | --- |
| Eq. 13: Top-k on `s + b`, mixture weights from the raw `s` | core's router does both; QB touches only `expert_bias_E` | same |
| Eq. 14: cutoff = (k+1)-th biased score, margins `s - alpha`, `b_hat = -quantile_{1-k/n}`, mean-centred, applied next step | `topk_with_cutoff`, `quantile_balance_bias` (exact), `quantile_balance_bias_histogram`; the solve runs at the optimizer pre-hook, so the batch that produced it is never routed with it | same |
| App. D: per-expert histogram of the required bias `r = alpha - s`, one integer all-reduce, first bin whose cumulative count reaches `ceil(q)`, linear interpolation inside it, mean-centred | histogram of the margin `s - alpha` (the mirror image), pooled over the loss mesh with one all-reduce, `(1-k/n)` quantile read from the CDF with in-bin interpolation | same up to the mirror |
| App. D: bins over `[b_min-1, b_max+1]`, recomputed every step from the current bias; B = 1000 | fixed `[-1, 1]`, 512 bins; margins outside are clamped into the end bins | deviation; harmless while the quantile sits far from the edges (bias within +-0.3 here), a scale concern if the bias grows past ~0.7 |
| App. D: optional EMA of the estimated quantiles across steps | not implemented | omission, optional in the report |
| hook registration | the flavor installs the solver as the model spec's `post_optimizer_build_fn`, the slot core's sign-rule registration occupies; the trainer calls one such function, so the sign rule is NOT registered under the flavor. The module's earlier comments said core's hook still ran first; corrected in `db65fc9f2`, which also resets the MoE's `tokens_per_expert_E` in core's place | text was wrong, math unaffected |

Synthetic check of the range deviation (`qb_range_synthetic.py`, the module's skewed n=16/k=2/m=4096 router,
iterated to a fixed point): exact quantile cv 0.005; fixed [-1, 1] 512 bins 0.026, 1000 bins 0.021; the
report's adaptive range at 1000 bins 0.018, at 512 bins 0.046 (the adaptive range is wider, so its bins are
coarser). The exact fixed point's bias spans [-0.28, 0.33]. Nothing at this bias magnitude separates the two
range conventions.

## 2. What the old loss table showed and why the gaps looked unexplained

Every pair agrees at step 1 (the bias is first written at the end of step 1) and differs by percents from
step 3 on. Routing census (the routed expert set of every token, same data, both hooks, one cache): at
step 2 the two hooks route 96 percent of the tokens differently (min 83, max 100 over the 23 MoE layers),
at step 10 99.7 percent. The hooks are training almost entirely different assignments from the second
step, so the loss curves are two different runs, and comparing them at step 3 or 10 measures nothing
about balancing. The loss rows stay in the PR as "still trains", with the same-cell-other-cache floor.

## 3. The evidence that belongs in the PR: load balance

Per-step expert load per MoE layer (tokens routed to each expert, reduced over the loss mesh), reported
as the coefficient of variation over experts, the largest and the smallest expert's load relative to the
mean, averaged over the 23 MoE layers and over ten-step windows (single steps move by +-0.1 from batch to
batch). Sign step is core's rule with `load_balance_coeff` 1e-3, the flavor's default.

| cell | hook | cv steps 1-10 | 11-20 | 21-30 | max/mean 21-30 | min/mean 21-30 | bias range at step 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dp1 | sign step | 1.37 | 1.46 | 1.33 | 5.0 | 0.03 | [-0.046, 0.014] |
| dp1 | quantile balancing | 1.00 | 0.83 | 0.59 | 2.6 | 0.18 | [-0.290, 0.277] |
| dp2 | sign step | 1.46 | 1.64 | 1.48 | 5.5 | 0.03 | [-0.046, 0.013] |
| dp2 | quantile balancing | 1.06 | 0.88 | 0.62 | 2.7 | 0.13 | [-0.202, 0.279] |
| dp2 x ep2 | sign step | 1.42 | 1.48 | 1.27 | 4.6 | 0.03 | [-0.042, 0.016] |
| dp2 x ep2 | quantile balancing | 1.14 | 0.97 | 0.74 | 3.1 | 0.08 | [-0.183, 0.273] |

Second fresh cache, dp1 pair (`mx3_qbtip2_*`): sign step cv 1.32 / 1.41 / 1.39, quantile balancing 0.94 / 0.91 / 0.63
over the three windows -- the load statistics reproduce across caches within 0.1, while the step-30 losses of
the same cells move by 21 and 15 percent (1.678 vs 1.320, 1.400 vs 1.191).

Under the sign step the imbalance grows from cv 1.07 at step 1 to about 1.4 and stays there (the largest
expert takes five times the mean, the smallest a few percent of it: the rule moves the bias by 1e-3 per
step against score gaps of order 1e-1). Quantile balancing brings it to 0.6-0.7 by steps 21-30 and is still
falling. Every rank holds the same bias after every solve (all-gathered and compared each step, zero
disagreements, ep2 included: the histogram is reduced over dp_shard x cp, which is where the router's
tokens are sharded; EP shards the experts, not the router's inputs).

Frozen router (`lr` 0, dp1, tip, 15 steps): the same data stream through unchanging weights, so only the
bias moves the routing. Sign step: cv 1.07 -> 0.84 at step 15 (bias within +-0.02). Quantile balancing:
1.07 -> 0.93 -> 0.64 -> 0.58 -> ... -> 0.42 at step 15, max/mean 2.3, min/mean 0.28, bias within +-0.11.
Not the few-step convergence to the histogram floor that the frozen synthetic router shows; the offline
solve on the dumped real scores (sec 4) says why.

## 4. Why 30 steps do not reach balance on this model

Two consecutive batches of raw router scores were dumped from the frozen model (23 MoE layers, 8192 tokens
each, 32 experts; `qb_probe/load_probe.py` with `QB_PROBE_SCORE_DIR`) and the update was solved offline
(`qb_score_dump_and_solve.sh`). The scores are not near-tied: mean 0.49, std 0.15, per-token spread 0.58, and
on average only the (k+1)-th score itself lies within 1e-3 of the cutoff. On one batch, iterating Eq. 14 with
the EXACT quantile (no histogram) takes the load cv from 0.93 to 0.53 after one solve, 0.40 after two, 0.34,
0.28 after five, 0.21 after ten; the histogram estimator tracks the exact solve within 0.02 at every count
(512 fixed bins 0.23 at ten solves, 4096 bins 0.21, the report's adaptive 1000 bins 0.22). The bias solved ten
times on batch 1 gives cv 0.31 on batch 2. The live run does one solve per step on a new batch each time; a
replica of its second step (one solve from zero on batch 1, applied to batch 2) gives 0.93 -> 0.53 on batch 1
and 0.58 on batch 2, which is what the frozen run shows (1.07 -> 0.93 -> 0.64).

So the slow approach to balance is a property of the per-step coordinate update of Eq. 14 on this router's
score structure (each solve demotes the popular experts, which moves every token's cutoff, which moves every
other expert's quantile: App. C's alternating solver, one expert-side sweep per training step), not of the
histogram estimator, the bin range, ties, or the reduction. The report's "equilibrates within a few update
steps" is stated for its trained 896-expert routers; on a random-init 32-expert debug router one sweep per
step reduces the imbalance by about 40 percent at the first step and by a few percent per step afterwards.
The sign step at 1e-3 does not move it at all within 30 steps. Neither the module nor the PR claims a
convergence rate; the PR's claim is the mechanism and the measured direction.

## 5. Consequences for the PR

- Base: `qb_review3` (`0e52d7b46`) on `f6b9152e9`; the "last runnable commit before the BFX9 gate" framing
  is obsolete (venv_bfx9 runs the tip, torch upgrade bitwise neutral on the same tree).
- Results section: step 1 bitwise per pair; the load table above as the evidence; the routing census as the
  explanation of the step-3/10 loss gaps; the loss rows with a second-cache floor row; the frozen-router rows.
- Design section: the registration statement corrected (`aff7abb` in the logbook body).
