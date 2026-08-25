Adds pipeline parallelism to the Kimi K3 text decoder. Everything except Block Attention Residuals is mechanical; the residual is not. A block residual is defined over the whole layer stack, so under PP it must travel between stages as a second payload alongside the hidden states, and the final aggregation (output_res_proj, then output_res_norm) must run only on the stage that owns lm_head.

The adapter's design: model.py returns (hidden, block_residual) from a non-head stage and takes the pair back on the next. Each hop ships only the blocks the receiver does not already hold -- layout.py precomputes, per (pp, vp, num_blocks, n_layers), which blocks every stage commits and which subset its outgoing P2P carries, so both ends agree without any metadata on the wire, and the receiver rebuilds the stack from its cached prefix plus the delta. The stack is a live autograd path, not a cache: a block cached on the same rank is stored detached and re-wrapped at read time so its gradient reaches the producer exactly once, and a block from another rank drains its gradient through PP's built-in backward P2P. The carry is configured by a config field, not an environment variable: a launcher exporting a variable non-uniformly gives ranks different topologies and hangs in a collective with nothing pointing at the cause.

Splitting the model in two by hand and running the halves in sequence -- no schedule, no loss, no microbatches -- reproduces the unsplit forward at max_abs 0.000e+00.

kimi_k3_debugmodel_text_32l, seed 42, --debug.deterministic, every cell loading the same seed checkpoint. dp2 is in the table because changing the data-parallel degree alone moves the loss more than pipelining does, so a dp2-mesh cell measured against dp1 would mostly be measuring that:

<<TABLE_PP>>

Not in this PR: the vision tower (its stage assignment and DEP) -- next, on the multimodal path. Without pipeline_parallel_degree > 1 none of this executes.

Tested: a CPU unit test for the FQN split; a pp2 integration cell, and a pp8 x vp4 one on the 32-layer flavor -- one layer per stage over 32 stages, so the residual crosses every boundary the schedule has.
