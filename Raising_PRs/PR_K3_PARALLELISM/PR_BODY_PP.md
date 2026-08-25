Adds pipeline parallelism to the Kimi K3 text decoder. Everything except Block Attention Residuals is mechanical; the residual is not. A block residual is defined over the whole layer stack, so under PP it must travel between stages as a second payload alongside the hidden states, and the final aggregation (output_res_proj, then output_res_norm) must run only on the stage that owns lm_head.

The adapter's design: model.py returns (hidden, block_residual) from a non-head stage and takes the pair back on the next. Each hop ships only the blocks the receiver does not already hold -- layout.py precomputes, per (pp, vp, num_blocks, n_layers), which blocks every stage commits and which subset its outgoing P2P carries, so both ends agree without any metadata on the wire, and the receiver rebuilds the stack from its cached prefix plus the delta. The stack is a live autograd path, not a cache: a block cached on the same rank is stored detached and re-wrapped at read time so its gradient reaches the producer exactly once, and a block from another rank drains its gradient through PP's built-in backward P2P. The carry is configured by a config field, not an environment variable: a launcher exporting a variable non-uniformly gives ranks different topologies and hangs in a collective with nothing pointing at the cause.

Splitting the model in two by hand and running the halves in sequence -- no schedule, no loss, no microbatches -- reproduces the unsplit forward at max_abs 0.000e+00.

kimi_k3_debugmodel_text_32l, seed 42, --debug.deterministic, every cell loading the same seed checkpoint. dp2 is in the table because changing the data-parallel degree alone moves the loss more than pipelining does, so a dp2-mesh cell measured against dp1 would mostly be measuring that:

| cell | stages | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | - | 1 | 12.48548 | 7.92534 | 3.41439 |
| pp2 | 2 | 2 | 12.48548 | 7.91227 | 3.35806 |
| pp4 | 4 | 4 | 12.48548 | 7.91227 | 3.35881 |
| pp8 | 8 | 8 | 12.48548 | 7.90930 | 3.40345 |
| pp2 x vp2 | 4 | 2 | 12.48548 | 7.91227 | 3.35876 |
| pp2 x vp4 | 8 | 2 | 12.48548 | 7.90930 | 3.40269 |
| pp4 x vp2 | 8 | 4 | 12.48548 | 7.91227 | 3.35819 |
| pp4 x vp4 | 16 | 4 | 12.48548 | 7.90930 | 3.40257 |
| pp8 x vp2 | 16 | 8 | 12.48548 | 7.90930 | 3.40356 |
| pp8 x vp4 | 32 | 8 | 12.48548 | 7.90930 | 3.40396 |
| dp2 | - | 2 | 12.47951 | 7.60609 | 3.22524 |
| dp2 x pp2 | 2 | 4 | 12.47951 | 7.65625 | 3.46469 |
| dp2 x pp4 | 4 | 8 | 12.47951 | 7.63091 | 3.46144 |

Step 1 is bit-identical to the baseline in all twelve pipelined cells -- to dp1
for the nine dp1 cells, to dp2 for the two mesh cells -- across two to thirty-two
stages, 1F1B and Interleaved1F1B, two to eight ranks. At step 2 the nine dp1
cells are 3.6e-4 and 6.9e-4 from dp1, and the two mesh cells 1.3e-3 and 6.3e-4
from dp2, against 1.6e-2 for the dp2 row measured the same way against dp1.

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

A second measurement, with one patch on top: the grad-norm reduction carried in
float32 rather than in the gradients' dtype. That patch is not part of this PR
and is not on this branch -- it is a separate upstream change, still open. The
table above is what this branch produces today; this one is what it produces
with that change applied, and the difference is the reason the step-3 column
above has two values instead of one.

| cell | stages | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | - | 1 | 12.48548 | 7.90621 | 3.40567 |
| pp2 | 2 | 2 | 12.48548 | 7.87346 | 3.43178 |
| pp4 | 4 | 4 | 12.48548 | 7.87346 | 3.43178 |
| pp8 | 8 | 8 | 12.48548 | 7.87346 | 3.43178 |
| pp2 x vp2 | 4 | 2 | 12.48548 | 7.87346 | 3.43178 |
| pp2 x vp4 | 8 | 2 | 12.48548 | 7.87346 | 3.43178 |
| pp4 x vp2 | 8 | 4 | 12.48548 | 7.87346 | 3.43178 |
| pp4 x vp4 | 16 | 4 | 12.48548 | 7.87346 | 3.43178 |
| pp8 x vp2 | 16 | 8 | 12.48548 | 7.87346 | 3.43178 |
| pp8 x vp4 | 32 | 8 | 12.48548 | 7.87346 | 3.43178 |
| dp2 | - | 2 | 12.47951 | 7.60609 | 3.22524 |
| dp2 x pp2 | 2 | 4 | 12.47951 | 7.63091 | 3.47191 |
| dp2 x pp4 | 4 | 8 | 12.47951 | 7.63091 | 3.47191 |

Nine pipelined cells, one curve, every digit: two to thirty-two stages, both
schedules, two to eight ranks. The two mesh cells likewise agree with each
other. Where a run cuts the model stops being visible in its numbers.

Tested: a CPU unit test for the FQN split; a pp2 integration cell, and a pp8 x vp4 one on the 32-layer flavor -- one layer per stage over 32 stages, so the residual crosses every boundary the schedule has.
