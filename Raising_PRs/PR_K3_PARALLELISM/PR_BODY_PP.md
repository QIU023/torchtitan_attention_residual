Adds pipeline parallelism to the Kimi K3 text decoder. Everything except Block Attention Residuals is mechanical; the residual is not. A block residual is defined over the whole layer stack, so under PP it must travel between stages as a second payload alongside the hidden states, and the final aggregation (output_res_proj, then output_res_norm) must run only on the stage that owns lm_head.

The adapter's design: model.py returns (hidden, block_residual) from a non-head stage and takes the pair back on the next. Each hop ships only the blocks the receiver does not already hold -- layout.py precomputes, per (pp, vp, num_blocks, n_layers), which blocks every stage commits and which subset its outgoing P2P carries, so both ends agree without any metadata on the wire, and the receiver rebuilds the stack from its cached prefix plus the delta. The stack is a live autograd path, not a cache: a block cached on the same rank is stored detached and re-wrapped at read time so its gradient reaches the producer exactly once, and a block from another rank drains its gradient through PP's built-in backward P2P. The carry is configured by a config field, not an environment variable: a launcher exporting a variable non-uniformly gives ranks different topologies and hangs in a collective with nothing pointing at the cause.

Splitting the model in two by hand and running the halves in sequence -- no schedule, no loss, no microbatches -- reproduces the unsplit forward at max_abs 0.000e+00.

| cell | stages | world | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.48548 | 7.92534 | 3.41439 |
| pp2 | 2 | 2 | fallback | 12.48548 | 7.91227 | 3.35806 |
| pp4 | 4 | 4 | fallback | 12.48548 | 7.91227 | 3.35881 |
| pp8 | 8 | 8 | fallback | 12.48548 | 7.90930 | 3.40345 |
| pp2 x vp2 | 4 | 2 | delta | 12.48548 | 7.89923 | 3.42837 |
| pp2 x vp4 | 8 | 2 | delta | 12.48548 | 7.94984 | 3.39687 |
| pp4 x vp2 | 8 | 4 | delta | 12.48548 | 7.93775 | 3.28493 |
| pp4 x vp4 | 16 | 4 | delta | 12.48548 | 7.89573 | 3.33794 |
| pp8 x vp2 | 16 | 8 | delta | 12.48548 | 7.93965 | 3.25196 |
| pp8 x vp4 | 32 | 8 | delta | 12.48548 | 7.91517 | 3.38148 |

Step 1 is bit-identical to dp1 in all nine, across two to thirty-two stages and
two to eight ranks. The transport is the delta one wherever the schedule can
carry it: plain 1F1B gives a rank one stage, so there is no rank-shared stack to
reuse and it falls back to shipping the whole stack.

The same six virtual-stage cells with the transport turned off, against the rows
above:

| cell | step 1 | step 2 | step 3 | step 10 |
|---|---|---|---|---|
| pp2 x vp2 | identical | 3.5e-4 | 1.6e-3 | 2.1e-2 |
| pp2 x vp4 | identical | 3.5e-4 | 5.1e-3 | 1.7e-3 |
| pp4 x vp2 | identical | 1.5e-4 | 3.2e-3 | 2.2e-2 |
| pp4 x vp4 | identical | 4.9e-4 | 1.7e-3 | 1.9e-2 |
| pp8 x vp2 | identical | 5.2e-4 | 3.8e-3 | 4.5e-2 |
| pp8 x vp4 | identical | 6.4e-4 | 7.4e-4 | 6.6e-3 |

Same forward, different gradients: routing the same blocks a different way, and
summing them in a different order.

Peak memory per rank, same topology and schedule, transport against fallback:

| pp8 x vp4 | per-rank peak (GiB) | max | spread |
|---|---|---|---|
| delta | 2.62 x6, 6.57, 6.60 | 6.60 | 3.98 |
| fallback | 2.66 x6, 7.83, 8.50 | 8.50 | 5.84 |

The six ranks that hold little are unchanged; the saving is on the two that
would otherwise accumulate the stack. pp8 x vp2 is the same shape, 7.92 down to
6.03.

Each cell runs twice and the first run is discarded. A cold compile cache moves
this model's step-1 loss by 6.8e-3 (12.59459 against 12.58783, both
reproducible), which is larger than most of the differences above; the discard
applies to the baseline row too.

Not in this PR: the vision tower (its stage assignment and DEP) -- next, on the multimodal path. Without pipeline_parallel_degree > 1 none of this executes.

Files:

    torchtitan/models/kimi_k3/
      pipeline_adapter.py   +1205  the pipelining_fn: FQN split, the block-residual
                                   carry across stages, the rank-shared stack, and
                                   the topology record that used to be an env var
      layout.py              +293  offline algebra over (pp, vp, num_blocks,
                                   n_layers, layers_per_block): which blocks each
                                   stage commits, which subset its P2P ships
      __init__.py           +47/-5 registers pipelining_fn and the 32-layer text
                                   flavor; zero-init on the AttnRes projections
      model.py              +25/-3 returns (hidden, block_residual) off a non-head
                                   stage, takes the pair back on the next, and
                                   guards the head-only aggregation
      config_registry.py       +24 the 32-layer trainer flavor
      parallelize.py         +2/-3 pipeline parallel off the unsupported list
    tests/
      unit_tests/cpu/test_kimi_k3_pp_fqn_injection.py  +105  the FQN split, on CPU
      integration_tests/features.py                     +14  pp2 and pp8 x vp4 cells
    torchtitan_recipes/tests/features.py                +33  their configurations

The same matrix with one patch on top: the grad-norm reduction carried in float32
rather than in the gradients' dtype. That patch is a separate upstream change, still open at https://github.com/pytorch/torchtitan/pull/4135, and is not on this branch.

| cell | stages | world | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.48548 | 7.90621 | 3.40567 |
| pp2 | 2 | 2 | fallback | 12.48548 | 7.87346 | 3.43178 |
| pp4 | 4 | 4 | fallback | 12.48548 | 7.87346 | 3.43178 |
| pp8 | 8 | 8 | fallback | 12.48548 | 7.91590 | 3.41976 |
| pp2 x vp2 | 4 | 2 | delta | 12.48548 | 7.90505 | 3.34575 |
| pp2 x vp4 | 8 | 2 | delta | 12.48548 | 7.94826 | 3.32969 |
| pp4 x vp2 | 8 | 4 | delta | 12.48548 | 7.95831 | 3.38758 |
| pp4 x vp4 | 16 | 4 | delta | 12.48548 | 7.90776 | 3.31534 |
| pp8 x vp2 | 16 | 8 | delta | 12.48548 | 7.92740 | 3.31839 |
| pp8 x vp4 | 32 | 8 | delta | 12.48548 | 7.91857 | 3.37609 |

Step 1 is unchanged. With the reduction in float32 the six virtual-stage cells run
with the transport off collapse to a single value (7.87346, all six); with it on
they stay apart, because the delta a hop carries depends on the cut and summing it in
a different order is arithmetic the patch does not touch.

Tested: a CPU unit test for the FQN split; a pp2 integration cell, and a pp8 x vp4 one on the 32-layer flavor -- one layer per stage over 32 stages, so the residual crosses every boundary the schedule has.
