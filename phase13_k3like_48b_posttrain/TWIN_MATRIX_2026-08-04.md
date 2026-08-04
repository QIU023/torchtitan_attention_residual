# The #4025 twin flavor: 13/13 eager, 10/13 compiled

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
