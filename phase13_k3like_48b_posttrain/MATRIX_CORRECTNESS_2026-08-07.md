# What the parallelism matrix cannot see, and the check that can

Two real defects were found on 2026-08-07 that eighteen parallelism cells at 100
steps had passed. Neither was a step-count problem. The matrix's judgement is
**loss agreement across cells**, and that judgement has a floor which is wider than
a defect needs to be.

## Why 100 steps does not help

* The eighteen-cell step-1 spread is **0.009** full-parameter and **0.0146** under
  LoRA. Anything smaller than that is indistinguishable from the band.
* The LoRA `o_proj` defect was worth **2.9e-02** in step-1 loss at tp2 and
  **14%** in grad_norm at tp4. The loss figure sits inside the band.
* The MLA CP defect was worth **1-6% relative on one parameter's gradient** in four
  layers. It moved the loss by far less than the band.

More steps do not narrow the band; they extend it, because divergence accumulates.
The problem is the KIND of judgement, not its duration.

## Cross-run comparison has a floor. Cross-rank comparison does not.

The per-parameter cross-layout comparison (`tp_trainer_grad_probe.py`) looked like
the answer, and it has its own floor, measured rather than assumed:

**Change only the gradient-accumulation order, on ONE GPU, with no parallelism at
all** -- `local_batch_size 2` against `4`, same global batch, same seed:

| comparison | A_log median | A_log max | max over params >1% of norm |
|---|---|---|---|
| **dp1 lbs4 (no parallelism)** | **0.26676** | **0.60516** | **0.09750** |
| pp2 | 0.35683 | 2.36570 | 0.23002 |
| cp2 | 0.32216 | 2.45181 | 0.12343 |
| tp2 | 0.29855 | 0.74178 | 0.11327 |

So a cross-layout number near 0.1 norm-weighted is the instrument, not a finding.
Three properties of that band identify it as reduction-order sensitivity rather
than a defect: the direction is mixed (at pp2, 244 parameters larger and 346
smaller -- a missing reduction can only go one way), the deviation scales inversely
with gradient share (median 0.057 in the smallest bucket down to 0.0028 in the
largest), and `A_log` is 7.6x worse than other parameters of the same magnitude,
which is what a sum of cancelling terms looks like.

A gradient whose placement on some mesh axis is `Replicate` must be **bit-identical
across that axis's ranks**. There is no tolerance to interpret: the answer is
exactly 0.0 or there is a bug. It needs no reference run, so it is one assertion
per cell rather than a comparison between cells.

## The check: `matrix_scripts/replicate_axis_check.py`

Both defects were found with this shape. Each of its properties exists because
getting it wrong cost something on 2026-08-07:

1. **Per-axis subgroups, not the world group.** A gradient can be Replicate on tp
   and Shard on the fsdp axis; gathering over the world then compares different
   shards and reports a false disagreement.
2. **The verdict is world-reduced before it is printed.** At `fsdp2 x tp2 x cp2`
   ranks 6 and 7 each held 12 disagreements while ranks 0-5 held none, and a
   rank-0-only verdict printed **PASS**. A per-rank verdict can hide another rank's
   failure.
3. **The reduction happens inside the step, not at the end.** torchtitan destroys
   the process group before `main()` returns, so a collective after training is a
   silent no-op -- it printed PASS on the run above.
4. **An explicit verdict line, which the collector must REQUIRE.** Exit status
   cannot carry it: an import error, an OOM, a port collision and a real
   disagreement all exit 1. A missing PYTHONPATH once read as "four cells found
   problems".
5. **`NOASSERT` is distinct from `FAIL`.** CP-only and FSDP-only cells have no
   Replicate axis, so the check covers nothing there; calling that a pass is how a
   cell looks verified without being measured, and calling it a failure buries the
   real ones.
6. **Agreements are recorded with their magnitude.** Absence must not mean both
   "agreed" and "never compared", and a zero-initialized parameter agrees with
   itself trivially -- `testable` counts only non-zero agreements. Run >= 3 steps.
7. **Skips are recorded with a reason**, including empty local shards, which occur
   at higher degrees and make `.max()` raise on a zero-element tensor.

## Coverage, stated rather than implied

| cell type | covered floor-free? |
|---|---|
| any cell containing TP | **yes** |
| CP-only (`cp2`, `cp4`) | **no** -- 1182 of 1182 records are `no_replicate_axis` |
| FSDP-only | **no** -- gradients are Shard on the dp axis |

The general CP invariant -- CP shards the sequence, not the parameters, so every CP
rank must hold the same gradient -- **does not hold in this stack**: the `"fsdp"`
mesh here is `dp_shard x cp`, so parameters are FSDP-sharded across the cp axis too.
That was discovered because the check records `no_replicate_axis` rather than
skipping silently. Had it treated "nothing to compare" as a pass, cp2 would have
shown a green PASS.

For the uncovered cells the assertion has to be the cross-run comparison, and its
verdict must be "does not exceed the floor above", not "equals zero". The
accumulation-order control belongs in the matrix as a permanent cell for exactly
that reason: it costs one single-GPU run and it calibrates every other number.

## Results on the main model, after both fixes

`kimi_k3_debugmodel_report_arch`, full-parameter, multimodal, 3 steps:

| cell | verdict | comparisons | testable |
|---|---|---|---|
| `tp2` | PASS | 1956 | 1892 |
| `tp4` | PASS | 3912 | 3784 |
| `tp2 x cp2` | PASS (was FAIL: 24) | 3786 | 3664 |
| `fsdp2 x tp2 x cp2` | PASS | 7410 | 7172 |
| `tp2 x pp2 x cp2` | PASS | 3786 | 3664 |
| `ep2 x fsdp2 x tp2 x cp2` | PASS | 7410 | 7172 |
| `cp2`, `cp4`, `fsdp2` | NOASSERT | 0 | 0 |

## What this means for the published matrices

The four cells carrying both TP and CP -- `tp2 x cp2`, `fsdp2 x tp2 x cp2`,
`tp2 x pp2 x cp2`, `ep2 x fsdp2 x tp2 x cp2` -- produced their published numbers
with the MLA CP defect present, and passed. They need rerunning on the fixed head
before being cited. The remaining fourteen are unaffected by that fix.
