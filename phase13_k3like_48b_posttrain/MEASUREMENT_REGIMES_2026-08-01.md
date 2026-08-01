# Three measurement regimes, and which one actually measures TP correctness

Every per-parameter number in TP_GRAD_FINDING before today came from ONE step.
The probe dumped at the first `clip_grad_norm_` and stopped, so `--training.steps 3`
still produced a single comparison. It now dumps every step up to `GRADCHK_STEPS`.

Running five steps exposed that the single-step choice was not neutral, and that
the obvious fix -- just compare more steps -- measures the wrong thing.

## Regime 1: cold seed, step 1

Clean in the sense that both runs hold identical weights, so any difference is
the gradient. But it is the least informative point in training: everything
zero-initialized is still zero. LoRA's B is zero, so grad_A is exactly zero and
the adapter contributes nothing. The AttnRes pseudo-queries are zero, so the
block softmax is exactly uniform, which is where its cancellation is WORST.

Consequence, both directions: blind to LoRA entirely (that is how the LoRA defect
survived a check), and pessimistic for AttnRes-adjacent parameters.

## Regime 2: multi-step trajectory

Dumping steps 1-5 and comparing step N to step N does NOT measure gradient
correctness. After step 1 the runs' weights have diverged, so by step 5 the two
gradients are evaluated at different points in parameter space. What grows is
drift, not error.

  21-layer dense + AttnRes    max |r-1| tp2   0.00040 -> 0.00340 over steps 1-5
  4-layer MLA+MoE             max |r-1| tp2   0.05283 -> 0.26272
                              max |r-1| tp4   0.02989 -> 0.67734

Reporting 0.68 as a TP error would be wrong. It is two trajectories separating.

## Regime 3: warm checkpoint, one step -- the right instrument

Train to a non-degenerate point, checkpoint, load that SAME checkpoint into every
tp degree, compare one step. Weights are identical (so it measures gradients) and
the zero-init paths are active (so nothing is invisible).

4-layer MLA+MoE+AttnRes, warm 5 steps, one step compared:

  max |ratio-1|   tp2 0.00836   tp4 0.01518
  median          tp2 0.00054   tp4 0.00078

  for comparison, cold seed step 1:  max tp2 0.05283  tp4 0.02989  median 0.00162

Better than cold, not worse -- because at a warm point the AttnRes softmax is no
longer uniform and its cancellation is milder. So the headline TP number at a
realistic operating point is max 1.5%, median 0.08%.

## What this changes

- The MoE "residual" was overstated: 1.5% max at a warm point, not 5%.
- The 21-layer dense AttnRes result stands (0.0004 cold, and its 5-step growth is
  drift).
- The LoRA defect is real and only visible warm. That was found this way already.
- Anything else verified only at cold-seed step 1 should be re-run warm before
  being trusted -- which is most of TP_GRAD_FINDING. The conclusions about the two
  FIXED defects are not at risk (both showed exact 1/tp and sqrt(tp) signatures
  with ranks disagreeing, which drift cannot fake), but the residual magnitudes
  quoted throughout are cold-seed numbers.

## Regime 4: multi-step loss, for smoke -- which is a different question

The warning above ("multi-step comparison measures drift, not gradient error")
applies to per-parameter gradient checks. For a SMOKE test the question is the
opposite one -- does this parallelism combination still train the same way over
time -- and there a loss curve over several steps is exactly the right signal,
because a small per-step error shows up as a curve peeling away.

Eight steps, one seed checkpoint, 4-layer MLA+MoE+AttnRes, all legs on the same
global batch:

  ref            7.68273 7.32385 6.82931 6.18582 5.73000 5.39304 5.24443 5.04205
  tp2            max |dloss| 0.00301   final 5.04506
  tp2 x pp2      max |dloss| 0.00292   final 5.04497
  tp2 x cp2      max |dloss| 0.00824   final 5.04839
  tp2 x pp2 x cp2  max |dloss| 0.01007 final 5.04428
  fsdp2 x tp2 x cp2  max |dloss| 0.09920  final 5.03531

No curve peels away. The last leg deviates an order of magnitude more, but it is
not the same computation: dp_shard=2 splits the same global batch into 2
accumulation steps instead of 4, so its bf16 accumulation structure differs. It
should not be read alongside the first four.

A parsing trap worth recording, since the first run of this was garbage: every
rank prints the loss line, so a tp2 run emits 16 values for 8 steps. Comparing
that positionally against an 8-value reference produced max |dloss| of 1.4 to 2.6
and looked like catastrophic divergence. Dedupe by step number first.
