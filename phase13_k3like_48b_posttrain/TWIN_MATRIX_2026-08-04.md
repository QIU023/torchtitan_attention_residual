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
