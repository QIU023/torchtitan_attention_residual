# PP adapter: per-parameter gradient verification

Until now PP had only ever been checked at the loss / grad_norm level. Those are
aggregates -- a wrong gradient on one parameter can hide inside a global norm,
and the AttnRes cross-stage adapter is exactly the kind of hand-routed backward
path that would show up per-parameter first. The TP investigation had just shown
that a parameter-level defect (block_attn_res, off by exactly tp) sat underneath
grad_norm numbers that looked unremarkable, so the same instrument was pointed at
PP before trusting it further.

## Method

Same instrument as the TP work: one model (`kimi_linear_k3mini_block_attn_res`,
21 layers, KDA + MLA + MoE + AttnRes, 80.9M params), one shared seed checkpoint,
vary only the parallelism. `tp_trainer_grad_probe.py` now dumps per-rank
(`<path>.r<rank>`) because under PP each rank owns a different stage and a
rank-0-only dump would have seen only the first stage.

`dp_shard` is held at 2 on every leg. It cannot be dropped to 1: without FSDP's
mixed-precision cast the KDA params stay fp32 and fla's kernel asks for 108,160 B
of dynamic shared memory against this GPU's 101,376 B limit, at any seq_len or
batch size. That also caps this study at pp4 on 8 GPUs -- **pp8 is not covered
here**, since it would need dp_shard=1.

Every leg accumulates exactly 4 partial sums of batch 1, so the bf16 accumulation
structure is identical and only the parallelism differs:

    pp1  dp2 local1 global8 -> 4 accumulation steps x 1 microbatch
    pp2  dp2 local2 global8 -> 2 accumulation steps x 2 microbatches
    pp4  dp2 local4 global8 -> 1 accumulation step  x 4 microbatches

Without this the accumulation order alone moves the numbers and the comparison
measures bf16 summation, not the adapter.

## Result: clean

Ratio = pp1 / leg, per parameter, over 548 parameters with nonzero gradients
(592 total; 44 are zero in both):

    pp2       max |ratio-1| = 0.00000
    pp4       max |ratio-1| = 0.00000
    pp2 x vp2 max |ratio-1| = 0.00000
    pp4 x vp2 max |ratio-1| = 0.00000

Loss 7.71304 and grad_norm 8.4957 on every leg, matching the no-PP baseline.

The stage split is real, not a merge artifact -- per-rank dump sizes are
312/280 for pp2 and 138/174/168/112 for pp4 (each repeated across the two dp
replicas), unioning to exactly 592. So this covers the whole model, including
every parameter whose gradient crosses a stage boundary.

VP changes nothing, consistent with the earlier finding at the loss level.

## Scope

Verified: the AttnRes cross-stage adapter routes gradients correctly at pp2 and
pp4, with and without VP, on the full K3 topology.

Not covered: pp8 (blocked by the KDA shared-memory limit above), and PP combined
with TP -- TP still carries an open MoE defect (TP_GRAD_FINDING_2026-07-29), so
a PP x TP leg would mix a known bug into the measurement.
