# Verifying parallelism at the gradient level: three wrong answers and the control that settled it

Refs: pytorch/torchtitan#3029

The 13-leg and 12-leg matrices verify that every parallelism combination runs
and that the losses agree. That is weaker than it looks. This records an attempt
to verify at the gradient level instead, which produced three successive
conclusions, the first two of which were wrong and the third of which was also
wrong, before a control settled it.

Keeping the failed reasoning because the failure mode is reusable: **a metric
without its control is not evidence.**

## Round 1: norm ratio says KDA's A_log is broken

Instrument: shared warm checkpoint, one varied dimension (tp), per-parameter
gradient norm, ratio `|g_tp1| / |g_tpN|`.

On the full-param flavor, the worst offenders were all `self_attn.A_log`, with
ratios from 0.36 to 3.13 and no consistent direction. Reported as a TP defect in
KDA's replicated parameters.

**Wrong.** Two checks killed it:

* `replicated_grad_rank_probe.py` compared the 503 tp-replicated gradients
  across ranks directly: all agree. A missing all-reduce would leave each rank
  holding its own partial contribution. It does not.
* The deviation is inversely correlated with gradient magnitude
  (`corr(log10 relative magnitude, |ratio-1|) = -0.242`):

  | relative magnitude | n | median \|ratio-1\| | max |
  |---|---|---|---|
  | <1e-4 | 61 | 0.0401 | 2.1345 |
  | 1e-4 .. 1e-2 | 317 | 0.0113 | 0.7609 |
  | >1e-2 | 212 | 0.0053 | 0.1532 |

  `A_log`'s gradients are 4.8e-6 to 1.1e-3 against a model median of 2.5e-3 --
  two to three orders below. Within that one family, the largest gradient was
  off by 3.5% and the smallest by 64%. That is cancellation in bf16, not a
  missing reduction, which would be magnitude-independent and scale with tp.

A first version of the rank probe reported `checked 0 replicated-gradient
parameters` and then printed "all agree". Under FSDP every parameter is Shard on
the dp axis, so requiring Replicate on *all* axes excluded everything. **A check
that examined nothing is indistinguishable from a check that passed.**

## Round 2: "there is no TP defect"

Given round 1, the conclusion was that TP is fine. Also wrong -- the norm ratio
cannot see direction. Two gradients of equal length pointing different ways
score 1.0000.

## Round 3: cosine says AttnRes is broken

Switching to cosine similarity between the full gradient tensors changed the
ranking completely. On the LoRA flavor, tp1 vs tp2:

    1-cos      ratio    |g|ref       parameter
    7.71e-01   1.1979   7.17e-07     layers.8.attn_res_norm.weight
    7.60e-01   1.1192   1.92e-03     layers.8.attn_res_proj.weight
    7.27e-01   0.9430   1.07e-02     layers.16.attn_res_proj.weight

Control (tp1 vs tp1): `1-cos = 1.8e-7`. So the 0.7 is real, not noise. And
`layers.16.attn_res_proj.weight` carries 1.07e-2, above the model median, so the
magnitude explanation from round 1 does not apply. Reported as a real defect in
the AttnRes TP gradient path.

**Also wrong.** The missing control was the model, not the metric.

## The control that settled it: dense vs MoE

`kimi_k3_mini_diag_4l_mla` is dense -- AttnRes, no MoE -- and otherwise the same
shape. Same instrument, same varied dimension:

| model | worst 1-cos, tp1 vs tp2 |
|---|---|
| tp1 vs tp1 (control) | 1.8e-7 |
| **dense + AttnRes** | **1.06e-4** |
| MoE | 7.7e-1 |

Four orders of magnitude apart, and on the dense model the AttnRes parameters do
not appear near the top at all.

MoE top-k routing is a discrete choice. A numerical difference far below any
tolerance flips which expert a token goes to, and that token's gradient then
takes an entirely different path. Near-orthogonal gradients on an MoE model
across parallelism configurations are a property of the architecture, not a
defect. `attn_res_proj` / `attn_res_norm` topped the list because they aggregate
across every layer, so a flip anywhere reaches them -- they are the downstream
victim, not the cause.

## What is actually established

Dense model, shared warm checkpoint, one varied dimension, per-parameter
gradient direction:

| varied | worst 1-cos |
|---|---|
| tp2 | 1.06e-4 |
| cp2 | 1.87e-5 |
| pp2 | not yet measured -- see below |

This is a much stronger statement than the loss matrices: the gradients agree in
direction to 1e-4, not merely the losses to 2%.

## Does this need running for every combination?

**Yes for dense, and it is the only place it can be run.** The check requires a
reference whose gradient is deterministic under a changed parallelism layout,
and MoE routing destroys that. So:

* **TP, CP, PP, and their combinations: test on dense.** TP and CP are done. PP
  is pending: the probe wrote every rank to one path and produced a corrupted
  archive; now fixed to per-rank filenames, but PP also needs the diff to
  compare only the parameters a stage owns.
* **EP cannot be verified this way at all.** EP requires experts, experts
  require routing, routing flips. Any EP gradient comparison across layouts
  measures route divergence, not correctness. EP's evidence has to come from
  somewhere else -- a routing-frozen comparison, or per-expert gradient checks
  with the assignment pinned. Stating this rather than quietly reporting a
  number that cannot mean what it looks like.

## LoRA

The recorded LoRA TP defect (max |ratio-1| 1.27 at tp2, 2.26 at tp4) is now
0.35 and 0.40 on the same instrument, and the signature changed: it no longer
grows with tp degree, which was the missing-reduction tell. The intervening
fixes -- the AttnRes `Partial()` removal, the `moe_sharding`
`in_grad_placements` drop, the non-Shard placement probe -- account for that.

**The residual is not yet characterized**, because every LoRA measurement so far
has been on `kimi_k3_mini_qlora`, which has MoE. It has to be repeated on a
dense LoRA flavor before anything is claimed. That is the next measurement, not
a conclusion.

## Instruments

* `grad_attrib_probe.py` -- per-parameter gradient dump and diff. Reports cosine
  and norm ratio; ranks by cosine. Warns loudly when a reference gradient is
  zero, which is what a cold-start comparison against a zero-initialized adapter
  produces.
* `replicated_grad_rank_probe.py` -- do tp-replicated gradients agree across
  ranks. Checks the tp axis only.

Both require: a shared warm checkpoint, exactly one varied dimension, and a
same-vs-same control run before any result is read.
