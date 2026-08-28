### Summary

Adds quantile balancing for the MoE router bias. K3 (report sec 2.3) runs sparsity beyond where the auxiliary-loss-free bias NUDGE still balances; quantile balancing SOLVES for the bias at each optimizer step from accumulated per-expert load histograms, instead of nudging it by a fixed step.

### Design

- One histogram tensor of shape (num_experts, num_bins) per MoE layer, preallocated at registration on the expert-bias device. The forward hook is a single `add_` and `step()` zeroes in place: no allocation, no branch -- which is what lets selective activation checkpointing recompute the forward identically (a lazily allocated buffer changes the op sequence between forward and recompute; that failure was observed and is the reason for the branch-free shape).
- The bias solve inverts the accumulated load CDF at the balanced quantile per expert. Registered as a `post_optimizer_build_fn`, replacing the default load-balancing hook; no model change.
- `kimi_k3_debugmodel_qb` enables it on the debug flavor.

### Evidence

CPU: 14 unit tests (histogram accumulation, CDF inversion, bias solve against a brute-force reference, in-place zeroing, SAC-identical op sequence).

GPU, this branch's worktree (one seed, warm cache, steps 1/3/10; the branch bases on upstream main, where the K3 EP gate is still on, so its cells are data-parallel -- the EP rows below come from the integration tree, where expert parallel is wired):

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.58962 | 8.46094 | 4.01340 |
| dp2 | 2 | 12.58743 | 8.19710 | 3.62665 |

Integration tree, same recipe over its own seed -- quantile balancing composes with expert parallel, which is where a bias-balancing method actually earns its keep:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.45537 | 7.75924 | 3.97453 |
| dp2 | 2 | 12.49584 | 7.08316 | 3.41460 |
| dp2 x ep2 | 2 | 12.49474 | 7.13975 | 3.42744 |
| dp4 x ep4 | 4 | 12.41280 | 6.53547 | 3.49066 |

SAC on/off with QB active prints identical losses (12.50913 / 11.66127 / 10.51695) on the integration tree -- the branch-free histogram hook is what makes the recompute identical.

### Changed files

    torchtitan/components/quantile_balance.py     +374
    torchtitan/models/kimi_k3/config_registry.py   +15  the qb flavor
    tests/unit_tests/cpu/test_quantile_balance.py +273
