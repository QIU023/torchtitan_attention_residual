Kimi K3 (report sec 2.3) runs MoE sparsity beyond where the auxiliary-loss-free
bias nudge still balances. Before this change the router bias update adds a
fixed sign-step nudge from the current step's load counts (core's default
load-balancing hook). After it the bias is solved at each optimizer step by
inverting an accumulated per-expert load histogram at the balanced quantile --
same bias tensor, same hook point, no model change.

Each MoE layer gets one preallocated (num_experts, num_bins) histogram on the
expert-bias device. The forward hook is a single add_ and step() zeroes the
buffer in place. It is branch-free and allocation-free on purpose: selective
activation checkpointing replays the forward, and a lazily allocated buffer
changes the op sequence between forward and recompute -- that failure was
observed, and with the branch-free hook SAC on/off prints identical losses.
The solver registers as a post_optimizer_build_fn replacing the default hook,
and kimi_k3_debugmodel_qb enables it on the debug flavor. 14 CPU unit tests
cover histogram accumulation, CDF inversion, the bias solve against a
brute-force reference, in-place zeroing, and the SAC-identical op sequence;
they pass on this branch as it sits on current main.

Training loss on kimi_k3_debugmodel_qb, one seed, warmed compile cache,
measured at this branch's pre-rebase tip d5393c240 (the rebase onto current main is
conflict-free, but main's KDA kernel now requires SM100-class hardware the
measuring box does not have):

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.58962 | 8.46094 | 4.01340 |
| dp2 | 12.58743 | 8.19710 | 3.62665 |

Changed files: torchtitan/components/quantile_balance.py (new),
tests/unit_tests/cpu/test_quantile_balance.py (new), the qb flavor in
torchtitan/models/kimi_k3/config_registry.py.
