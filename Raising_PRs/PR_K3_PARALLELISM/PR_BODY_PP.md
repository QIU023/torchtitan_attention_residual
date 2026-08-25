Adds pipeline parallelism to the Kimi K3 text decoder. Everything except Block
Attention Residuals is mechanical; the residual is not. A block residual is
defined over the whole layer stack, so under PP it has to travel between stages
as a second stage payload alongside the hidden states, and the final aggregation
-- output_res_proj then output_res_norm -- has to run on the stage that owns
lm_head and nowhere else.

Three things fail silently today rather than raising. The block residual does
not cross a stage boundary at all, so every stage past the first accumulates
onto a zero it was handed. output_res_proj / output_res_norm run on every stage
instead of the last. And the FQN injection returns early on a Config-tree model
without logging, so the split falls back to core's generic one and the last
stage loses those two modules entirely.

model.py returns (hidden, block_residual) from a non-head stage and takes the
pair back on the next. pipeline_adapter.py holds the pipelining_fn, the FQN
split (tok_embeddings + layers.N + norm/lm_head/output_res_proj/output_res_norm)
and the cross-stage adapter that carries the residual under Interleaved1F1B.
layout.py precomputes, per (P, V, num_blocks, n_layers, layers_per_block), which
blocks a stage commits and which subset its outgoing P2P must ship, so no
metadata travels on the wire. knobs.py moves the carry's topology off the
TORCHTITAN_ATTNRES_CACHE environment variable onto a config field: a launcher
that exports it non-uniformly gives different ranks different topologies and
hangs in a collective with nothing pointing at the cause.

Splitting the model in two by hand and running the halves in sequence -- no
schedule, no loss, no microbatches -- reproduces the unsplit forward at max_abs
0.000e+00. Dropping the carry and keeping everything else moves step 3 at pp2
from 7.44679 to 9.30017.

kimi_k3_debugmodel_text_32l, seed 42, --debug.deterministic, every cell loading
the same seed checkpoint:

<<TABLE_PPVP>>

Same, on a 2D mesh with data parallel, kimi_k3_debugmodel_text:

<<TABLE_PPDP>>

Not in this PR: the vision tower's stage assignment, and the tower on a pipeline
stage of its own. Those come next, on the multimodal path.

Without pipeline_parallel_degree > 1 none of this executes.

Tested: a CPU unit test for the FQN split; pp2 and pp2 x vp2 integration cells.
