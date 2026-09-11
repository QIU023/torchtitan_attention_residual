# Step-1 localisation for PR 4312: setup and partial results, superseded by the peer session's run

This session set up the two step-1 measurements (localise the dp1-vs-pp2 ulp at the stage
boundary; the per-parameter gradient profile at the published 24-layer depth) and was stood
down mid-run: a peer session owns the box and runs the same two as part of a superset. The
runs here are **not** a result — neither comparison completed. What is below is the setup, so
the peer does not rediscover it, and three facts the setup turned up.

## What exists

- **The probe patch**: `matrix_scripts/pploc_probe_hacks.patch` (also applied, uncommitted, in
  the worktree `/tmp/wt_pploc`, a detached checkout of `pp_review4` at `66601a7fb`). It carries
  four uncommitted edits: the SM120 KDA capability-guard lift; `FWD_DUMP`, which saves at every
  `_apply_attention_residual` call the fp32 `values_float` entering the matmul, the softmax
  weights `probs_T1N`, the output, and the stride/contiguity/shape/dtype of both `values_TND`
  and the incoming `block_residual_TND`; `GRAD_DUMP`, which saves `{fqn: grad.float()}` for
  every parameter at step 1 before the optimizer; and a `kimi_k3_debugmodel_deep` alias that
  selects the 33-layer registry flavour with the cached transport.
- **The comparison scripts**: `matrix_scripts/cmp_fwd.py` and `matrix_scripts/cmp_grad.py`. Both
  measure differences in **units in the last place** (int32 view, negatives mapped to a monotone
  ordering) rather than as relative error, which is the unit the claim is stated in. `cmp_fwd.py`
  pairs the two runs by `(layer, call)`, prints per-tensor differing-element counts and max ulps,
  reports which layers came from rank 1 (so the stage boundary is read off the data, not assumed),
  and dumps the stride/contiguity of the first differing pair. `cmp_grad.py` merges the pipeline
  run's per-rank dumps, compares against the single-GPU dump, and prints the per-layer progression
  plus the first non-identical tensor in backward order.
- **A dp1-only dump**, `/workspace/fwd_dump`, 524 files, 2.1 GB, 33-layer shape. Delete it when
  the peer's run supersedes it; it cannot answer anything on its own (see below).

## Why the runs answer nothing

The forward run's `pp2` cell never wrote: 524 files are `ws1` (single GPU) and **zero** are `ws2`,
and the mx3 run directory for the tag is gone, so the cell logs are not available either. A
comparison needs both sides. The 24-layer gradient run produced one file before it was stopped.

## Three facts worth carrying into the peer's run

1. **A dump keyed only by layer is wrong.** `_apply_attention_residual` runs many times per layer
   per step: two reads per layer per micro-batch (before the attention and before the FFN), four
   micro-batches, and then again under activation-checkpoint recomputation. The surviving files
   reach call index 22 for a single layer in one process, so a probe that writes one file per
   layer records only the last call, and "the last call" need not be the same call on the two
   paths. The probe here keys on `(rank, layer, call)` with a per-process counter.
2. **The seed build dumps too.** With `FWD_DUMP` exported, mx3's seed-checkpoint pass writes
   files before any cell runs, including one with `layer=None` (a call that does not pass through
   the patched block path). Files from earlier processes survive alongside the cell's own, so the
   dump directory must be cleared per cell, or the comparison filtered by mtime.
3. **Layers 1 to 32 dump, layer 0 does not** — layer 0 has no `attention_res_proj`, so it takes
   the `h_TD = x_TD` branch and never calls the residual. That is the expected placement and
   confirms the probe sits where it was intended to.

## The hypothesis this was built to test, unresolved

That the single-GPU path's `torch.cat`-grown stack and the pipeline path's
`torch.stack`-reassembled stack (`pipeline_stage.py` `assemble_stack`) differ in layout and so
change the reduction order of the fp32 matmul in `_apply_attention_residual`. Reading the code,
both `cat` and `stack` return contiguous tensors of the same shape, so the stride half of the
hypothesis looks unlikely and the likelier candidate is the open block's partial sum, which the
pipeline path re-materialises. Nothing here measures either way.


## Reading rules drafted for the cancelled 24-layer profile rerun (2026-09-11)

Found as an uncommitted edit to the numerics reply in the shared checkout, written before the
rerun it describes. The rerun was cancelled by the user; the reply cites the existing 33-layer
profile instead. Kept here so the rules are not lost if the rerun is ever done.


Fixing the reading rules first, so the shape of the result cannot be chosen after seeing it.

- **Metric.** Relative L2 distance per parameter tensor, `||g_pp2 - g_dp1|| / ||g_dp1||`, with each
  tensor's own gradient norm printed beside it. Elementwise detail is support only and is
  restricted to elements above the tensor's median absolute value: near-zero elements dominate any
  elementwise metric through sign flips and tiny denominators while carrying no weight in the
  update. Three earlier metrics were discarded for exactly that reason, each returning a number
  about the floating-point representation rather than about the model.
- **Gate.** The profile is reported only if two identical `dp1` runs are bit-identical to each
  other on this box, and only if `dp1` reproduces the table's reference and `pp2` sits within one
  float32 unit in the last place of it at step 1, with the loss read from whichever rank did not
  print the `-1.0` sentinel that non-last pipeline stages emit.
- **What is being looked for.** The per-layer relative L2 from the last layer down to layer 0. If
  the growth is smooth, it is smooth; if it steps at the stage boundary, it steps. At 24 layers the
  boundary is at layer 12 and `12 % 12 == 0`, so no partial block crosses it and the
  partial-sum mechanism predicts nothing special there. A step at layer 12 would weaken that
  mechanism, and the six boundary cells -- which hold the model and its size fixed and vary only
  where the boundary falls -- become the arbiter. Either outcome is usable; the prediction cells
  were built for it.
- **Layer 0 discriminates nothing on its own.** It is the deepest point of the backward pass, so
  the accumulated difference is expected to be largest there under any mechanism, including one
  with no block stack at all. A large value at layer 0 is not evidence for or against anything and
  will not be presented as such.

