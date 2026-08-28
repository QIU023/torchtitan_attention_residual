### Summary

Adds quantile balancing for the MoE router bias. K3 (report sec 2.3) runs sparsity beyond where the auxiliary-loss-free bias NUDGE still balances; quantile balancing SOLVES for the bias at each optimizer step from accumulated per-expert load histograms, instead of nudging it by a fixed step.

### Design

- One histogram tensor of shape (num_experts, num_bins) per MoE layer, preallocated at registration on the expert-bias device. The forward hook is a single `add_` and `step()` zeroes in place: no allocation, no branch -- which is what lets selective activation checkpointing recompute the forward identically (a lazily allocated buffer changes the op sequence between forward and recompute; that failure was observed and is the reason for the branch-free shape).
- The bias solve inverts the accumulated load CDF at the balanced quantile per expert. Registered as a `post_optimizer_build_fn`, replacing the default load-balancing hook; no model change.
- `kimi_k3_debugmodel_qb` enables it on the debug flavor.

### Evidence

CPU: 14 unit tests (histogram accumulation, CDF inversion, bias solve against a brute-force reference, in-place zeroing, SAC-identical op sequence).

GPU (this branch's worktree, 8x consumer GPUs; one seed, warm cache, steps 1/3/10):

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| TBD-QB-MATRIX | | | | |

SAC on/off with QB active prints identical losses (12.50913 / 11.66127 / 10.51695) on the integration tree; the branch re-run lands in the table above.

### Changed files

    torchtitan/components/quantile_balance.py     +374
    torchtitan/models/kimi_k3/config_registry.py   +15  the qb flavor
    tests/unit_tests/cpu/test_quantile_balance.py +273
