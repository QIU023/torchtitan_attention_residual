# The eager reference's architecture, unchanged: 18/18 eager, 15/18 compiled

The primary matrix is `REPORT_ARCH_MATRIX_2026-08-04.md` (the same cells with
the report's trailing Gated MLA). This file is the same cells on the eager
reference's architecture as published, and it is also the defect history: the
CP multimodal deadlock, the pp8 P2P failure, the grouped_mm operator
limitation, and four claims that were published and then retracted.

Read the sections in order. The early ones say 13/13 because the max-degree
cells did not exist yet; the number grows to 18/18 further down. Nothing above
is edited to match what came later -- the sequence is the point.

Refs: pytorch/torchtitan#3029, pytorch/torchtitan#4025

Flavor `kimi_k3_debugmodel_pr_4025`, which is **multimodal** (MoonViT +
cc12m-test), so this is the multimodal matrix -- not a separate text one.
3 steps, seed 42, `--debug.deterministic`, global batch 8.

Run at the flavor's own settings (seq_len 256, `local_batch_size` 1) rather
than CLI overrides, so what runs is #4025's configuration. `local_batch_size`
1 against global batch 8 means **gradient accumulation is on by default**,
which is the configuration that exposed the zero-sentinel CP defect.

## Eager: 13/13, reproduced

| leg | step 1 | step 2 | step 3 |
|---|---|---|---|
| dp1 | 12.05575 | 11.98981 | 11.80739 |
| fsdp2 | 12.07003 | 12.00412 | 11.81542 |
| pp2 | 12.04905 | 12.00033 | 11.79082 |
| cp2 | 12.06936 | 12.01476 | 11.80412 |
| tp2 | 12.05239 | 12.01448 | 11.83360 |
| fsdp2_tp2_pp2 | 12.05729 | 11.96100 | 11.73580 |
| fsdp2_tp2_cp2 | 12.04102 | 11.97096 | 11.80344 |
| tp2_pp2_cp2 | 12.04419 | 11.93646 | 11.72113 |
| fsdp2_pp2_cp2 | 12.07969 | 11.98816 | 11.75726 |
| ep2_fsdp2 | 12.07003 | 12.00365 | 11.83881 |
| ep2_fsdp2_tp2_pp2 | 12.03851 | 11.97321 | 11.74813 |
| ep2_fsdp2_tp2_cp2 | 12.05770 | 11.96636 | 11.80828 |
| ep2_fsdp2_pp2_cp2 | 12.06318 | 11.98685 | 11.77011 |

Step-1 spread 12.03851 to 12.07969, i.e. 0.041 across every parallelism
combination. Cold-start cross entropy at vocab 163840 is `ln(163840) =
12.006`, and all thirteen sit just above it.

**Two independent full matrix runs agree bit-for-bit on all thirteen legs.**
That is a stronger claim than "it passed": the numbers are reproducible, not a
single sample.

`tp2_pp2_cp2` reported 0/3 rows in one earlier matrix run and passed in three
subsequent runs -- twice standalone with the identical invocation, once in the
matrix -- with the same three losses every time. Recorded as a transient
launch failure rather than quietly dropped, and rather than called flaky
without the retries to back it.

## Compiled: 10/13

Same fixture plus `--compile.enable`.

| leg | step 1 | step 2 | step 3 |
|---|---|---|---|
| dp1 | 12.05531 | 11.99585 | 11.78329 |
| fsdp2 | 12.07168 | 11.98734 | 11.80438 |
| pp2 | 12.04727 | 11.97000 | 11.79151 |
| cp2 | 12.07200 | 12.00927 | 11.79936 |
| tp2 | 12.05092 | 12.00773 | 11.81889 |
| fsdp2_tp2_pp2 | 12.04522 | 11.95288 | 11.73298 |
| fsdp2_tp2_cp2 | 12.04324 | 11.97950 | 11.79677 |
| tp2_pp2_cp2 | 12.05400 | 11.96186 | 11.72728 |
| fsdp2_pp2_cp2 | 12.07922 | 11.98985 | 11.73709 |
| ep2_fsdp2 | 12.07740 | 11.99823 | 11.81797 |
| ep2_fsdp2_tp2_pp2 | FAIL | | |
| ep2_fsdp2_tp2_cp2 | FAIL | | |
| ep2_fsdp2_pp2_cp2 | FAIL | | |

Worst eager-vs-compiled difference on the ten that run: 0.0086
(`ep2_fsdp2` step 3, 11.83881 vs 11.81797), against a 0.041 spread across the
parallelism configurations themselves. So compile changes nothing beyond
arithmetic reordering where it runs.

All three failures share one cause, and it is not ours:
`torch._grouped_mm` rejects an operand whose contraction dimension is zero,
which is the shape the weight-gradient form takes when an expert group is
empty. Reproduced in five lines with no model, no compile and no distributed
(`GROUPED_MM_EMPTY_GROUP_2026-08-04.md`). Empty groups themselves are fine in
both eager and compiled.

Not worked around. A guard in `models/common/moe.py` would turn the three legs
green and hide the cause, which this repo's own rule forbids
("Investigate root cause before patching"). The fix belongs in the operator or
in inductor's lowering.

## What this does and does not establish

* Every parallelism combination this fork supports runs the #4025 model, on
  #4025's configuration, with gradient accumulation on, and agrees on loss to
  0.041 at step 1.
* It is **not** a numerical comparison against #4025's own tree. That tree
  refuses TP, CP and PP, so it has exactly one runnable cell, and a
  cross-codebase loss comparison would be confounded by init RNG order anyway.
  The architectural check available is that all thirteen land just above
  `ln(vocab)` at step 1, which they do.
* Compile is out of scope in #4025 and compile-off in everything we publish,
  so the three compiled EP failures block the "compile on and off" goal but
  nothing that has been claimed.


---

# 10-step and 100-step: 13/13 both

Same flavor and fixture, run with the small legs concurrent on disjoint GPU
sets and the 8-GPU legs serial. Full per-step numbers in `twin100_raw.txt`.

**Concurrency control**: steps 1-3 of the 10-step run reproduce the 3-step
serial matrix bit-for-bit on every leg, so running four legs at once did not
perturb anything.

## 100 steps

| leg | s1 | s25 | s50 | s75 | s100 |
|---|---|---|---|---|---|
| dp1 | 12.05575 | 3.81165 | 1.39091 | 0.52958 | 0.37123 |
| fsdp2 | 12.07003 | 3.77238 | 1.49744 | 0.56217 | 0.38411 |
| pp2 | 12.04905 | 3.82795 | 1.47136 | 0.56041 | 0.37325 |
| cp2 | 12.06936 | 3.81163 | 1.40859 | 0.54027 | 0.34736 |
| tp2 | 12.05239 | 3.74603 | 1.42770 | 0.55103 | 0.37299 |
| fsdp2_tp2_pp2 | 12.05729 | 3.68773 | 1.49043 | 0.50403 | 0.34810 |
| fsdp2_tp2_cp2 | 12.04102 | 3.68571 | 1.44435 | 0.59443 | 0.39448 |
| tp2_pp2_cp2 | 12.04419 | 3.69610 | 1.36390 | 0.57287 | 0.37996 |
| fsdp2_pp2_cp2 | 12.07969 | 3.70476 | 1.43364 | 0.53225 | 0.35862 |
| ep2_fsdp2 | 12.07003 | 3.74634 | 1.44065 | 0.56345 | 0.37278 |
| ep2_fsdp2_tp2_pp2 | 12.03851 | 3.67017 | 1.39659 | 0.53609 | 0.35122 |
| ep2_fsdp2_tp2_cp2 | 12.05770 | 3.77369 | 1.41237 | 0.58835 | 0.38370 |
| ep2_fsdp2_pp2_cp2 | 12.06318 | 3.58361 | 1.40409 | 0.54908 | 0.34978 |

All thirteen decrease monotonically. No NaN, no divergence.

## What this is NOT evidence of

Two things have to travel with these numbers.

**It is overfitting a smoke dataset.** `cc12m-test` is tiny; 12.06 -> 0.37 in
100 steps is memorization, not convergence quality. What the run establishes is
that thirteen parallelism configurations behave consistently and stably over a
100-step window, which is what it was for. Real convergence curves need the
H200 pretraining run.

**The spread grows in relative terms.** Absolute spread at step 100 is 0.047,
but that is ~13% relative, against ~0.3% at step 1. This is the ordinary
amplification of small absolute differences at low loss, not a defect signal --
and equally it does not support a claim that the thirteen "agree closely" at
step 100. State the step-1 agreement, not the step-100 one.

**The model is #4025's current architecture, not the report\'s.** Its final
layer is KDA where report sec 2.1 requires Gated MLA, and it has no final
aggregation over block representations (report sec 2.2). See
`STRUCTURE_AUDIT_2026-08-04.md`. That is deliberate for this matrix -- the
point is "our parallelism on their model" -- but it means these numbers must
not be presented as parallelism validated on the K3 architecture.

---

# Addendum: report-faithful architecture, and max-degree cells

## kimi_k3_debugmodel_report_arch, 100 steps: 13/13

The twin with one entry changed -- layer 13 is Gated MLA, per report sec 2.1.
Everything else identical.

| leg | s1 | s50 | s100 |
|---|---|---|---|
| dp1 | 12.05342 | 1.43325 | 0.38039 |
| fsdp2 | 12.05033 | 1.48655 | 0.40717 |
| pp2 | 12.04891 | 1.46258 | 0.40002 |
| cp2 | 12.04192 | 1.48153 | 0.35529 |
| tp2 | 12.07846 | 1.49733 | 0.39618 |
| fsdp2_tp2_pp2 | 12.05989 | 1.43384 | 0.35537 |
| fsdp2_tp2_cp2 | 12.10048 | 1.44392 | 0.40598 |
| tp2_pp2_cp2 | 12.05395 | 1.47912 | 0.38170 |
| fsdp2_pp2_cp2 | 12.03465 | 1.46866 | 0.39791 |
| ep2_fsdp2 | 12.05033 | 1.51492 | 0.39696 |
| ep2_fsdp2_tp2_pp2 | 12.05717 | 1.37986 | 0.35544 |
| ep2_fsdp2_tp2_cp2 | 12.06437 | 1.53136 | 0.42176 |
| ep2_fsdp2_pp2_cp2 | 12.07291 | 1.41445 | 0.34968 |

Step-1 band 12.035 to 12.100, against the twin's 12.039 to 12.080 -- the same
band, which is what one changed layer should produce.

## Max-degree cells on the twin, 3 steps

Run without touching the flavor. A cell needing a different layer, head or
expert count is recorded as not expressible, never accommodated: the twin's
value is "their exact config, our parallelism", and changing the config to make
a cell run would destroy exactly that.

| cell | result |
|---|---|
| `ep8_fsdp8` | 12.03346 11.97891 11.77869 |
| `pp4` | 12.06240 12.01471 11.81561 |
| `tp4` | 12.03252 11.98273 11.81679 |
| `cp4` | 12.07386 12.00828 11.83320 |
| `pp8` | **FAIL** -- see below |

`ep8_fsdp8` is the interesting one: 8 experts over 8 ranks is exactly one
expert per rank, so empty expert groups occur every step, and eager handles it
cleanly.

**Correction.** This was written as "the real-world trigger for the
`_grouped_mm` zero-contraction-dim defect". It is not. Run compiled WITHOUT the
shim, `ep8_fsdp8` passes (12.02227 ...). Empty groups are not sufficient; the
failing shape needs a rank with zero routed tokens in total, which the ep2
combinations with TP/PP/CP produce and ep8 does not. The claim was an
inference from "empty groups occur" and was never measured until now.

Not expressible on the twin's config, recorded rather than worked around:

| cell | why |
|---|---|
| `tp8`, `cp8`, `tp4 x cp4` | 4 attention heads (MLA and KDA both) -- nothing to shard 8 ways |
| `pp8 x vp4` | 13 layers cannot host 32 virtual stages |

High-degree coverage on our own flavors (tp8, cp8, PP8xVP4, EP@896) already
exists in this logbook and is citable; it was not re-run here.

## Open defect: pp8 on 13 layers

    ValueError: Tensors for P2P must be non-overlapping and dense
      torch/distributed/pipelining/schedules.py:814 _batch_p2p
      -> distributed_c10d.py:3682 batch_isend_irecv -> :3336 irecv

**Established**: `pp2` and `pp4` pass on the same flavor; `PP8xVP4` passes on
our own 32-layer flavor, so pipeline degree 8 is not broken generally. The
failure is 13 layers over 8 stages. Instrumenting the adapter's
`_finish_forward` produced no records at all, so the failure happens **before**
our code runs, while torch allocates the receive buffer.

**Suspected, NOT confirmed**: the adapter emits
`partial_out.new_zeros((0, *partial_out.shape))` when a stage commits no
blocks, and with block size 12 over 13 layers most of 8 stages commit nothing.
An empty payload would give a degenerate recv buffer. This is read off the code,
not measured -- the probe that would confirm it has to sit on the receive-buffer
allocation, not on our forward.

Same shape of problem as the `_grouped_mm` one: a validator rejecting a
legitimately empty tensor. Whether the fix belongs on our side (send a
1-element sentinel instead of an empty block stack) or upstream is not settled.

---

# pp8 fixed, and the PP8xVP4 pressure test after it

## The pp8 defect: cause, and two wrong turns

`pp8` over 13 layers died in torch's pipeline P2P with "Tensors for P2P must be
non-overlapping and dense". What torch ships backwards is the raw `grad_input`
autograd produced for a stage's PP inputs. The last stage aggregates the block
stack together with the partial block, so the gradient for the partial block
comes back as a slice of that wider buffer -- measured as `[1, 256, 256]` with
stride `[256, 768, 1]`, and 768 is 3 x 256: two blocks plus the partial.

Two wrong turns, recorded because each cost a cycle:

1. Making the **adapter's** outputs contiguous did nothing. The matrix never
   sets `TORCHTITAN_ATTNRES_CACHE`, so `CrossStageCacheAdapter` was not in the
   path at all -- which is also why probing `_finish_forward` produced zero
   records, a fact that should have been read as "not called" rather than
   "nothing to see".
2. Making the model's stage **outputs** contiguous did nothing either. The
   buffer belongs to the inputs.

Fix is `_DenseGrad` on the PP inputs: identity forward, `grad.contiguous()`
backward. Both wrong turns are reverted.

    pp8   ValueError -> 12.04408 / 11.96190 / 11.75889
    pp2   12.04905 / 12.00033 / 11.79082   (unchanged, bit for bit)
    pp4   12.06240 / 12.01471 / 11.81561   (unchanged, bit for bit)

So the matrix is **18/18**.

## PP8xVP4 after the change

The fix touches the PP input gradient path, and PP8xVP4 is the only
configuration that crosses it at every one of 32 virtual stage boundaries, so
it had to be re-run.

`kimi_k3_mini_pp8vp4` (32 layers), Interleaved1F1B, `layers_per_stage 1`,
`first/last_stage_less_layers 0`, **seq_len 1024**, 100 steps:

    step   1   loss 7.70028   grad_norm 8.3009
    step  25   loss 4.84887   grad_norm 4.4583
    step  50   loss 3.30078   grad_norm 2.5610
    step  75   loss 2.87262   grad_norm 0.8959
    step 100   loss 2.83465   grad_norm 0.7439

Monotone, no errors. And bit-identical to the pre-change tree at matched step
count (5 steps: 7.70028 / 7.39004 / 6.34917 / 5.60212 / 5.32743 both ways).

Two things this does NOT say, both of which have to travel with it:

* **seq_len 1024, not 8192.** The flavor's own 8192 OOMs on this box (15.48 GiB
  per GPU): 32 layers x 32 virtual stages x 8 microbatches of activations do
  not fit. Verified that the OOM is not ours by running the identical config on
  the pre-change file -- same OOM. So this run covers the gradient round-trip
  across every virtual stage boundary, which is what the change touches, and
  does not cover long-context activation or communication volume. The 8192
  configuration needs the H200.
* A first attempt at this comparison read as a regression (step 2 onward
  diverging badly) and was wrong: it compared a 100-step run against a 5-step
  control, and the LR schedule is computed over total steps, so step 2 already
  had a different learning rate. Matched step counts agree exactly. Recorded
  because the false signal was convincing.
