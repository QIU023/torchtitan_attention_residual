> **SUPERSEDED (2026-08-06) by `REPORT_ALIGNMENT_DEP_CP_MTP_2026-08-06.md`.**
> The reframing below was derived from my own timing analysis, not from the
> report. Report §5.2.3 prescribes the HYBRID: the first PP micro-batches' ViT
> forwards upfront, the remaining forwards injected into pipeline bubbles,
> backward analogously -- which strictly dominates Design A's parallel
> prologue. The dependency-inversion observation below is correct and is what
> the report works around the same way; the conclusion drawn from it is not.

# ViT DEP: distributing the vision tower across the PP axis

## The reframing that changes the design

The task was posed as "put the ViT compute into the PP bubbles". Working through
the timing, that is the wrong objective to optimise directly.

Bubbles are a **fixed** cost: warmup plus cooldown is O(P) slots regardless of
how many microbatches the step has. The vision encode is a **proportional**
cost: every microbatch carries images, so the tower's total work scales with M.
So as M grows the bubble fraction shrinks while the tower's share does not.

Which means the actual win is not "fill the idle" but **"stop doing all of the
tower's work on one rank"**. Today the whole tower runs on the stage that owns
`embed_tokens`, and at real scale MoonViT-V2 is 447.4M against k3mini's 80.9M
text side. One rank in P is doing a large serial prologue while P-1 wait. The
bubble is merely where the spare capacity happens to sit.

That reframing matters because it admits a much simpler and safer design than
true schedule interleaving, and gets most of the speedup.

## Why the naive bubble-filling version cannot work

Stage 0 needs microbatch 0's vision features at the **start** of its first
forward. Stages 1..P-1 are idle at that moment precisely because they are
waiting for stage 0. The idle that looks free is upstream of the work that would
fill it.

Formally: stage k is idle for t in [0, k). Stage 0 needs microbatch j's features
by t ~= j. So stage k can only serve microbatch j when k <= j, and it must
finish the encode inside its k idle slots. For small j that window is one or two
slots, and the tower's forward does not fit in one microbatch-forward of 1/V of
the layers. So the early microbatches cannot be offloaded at all, and the
offloadable fraction is (M - J0)/M for some J0 ~ P.

## Design A -- parallel prologue (recommended)

Before the schedule's action list starts, every PP stage encodes a disjoint
slice of the whole batch's images; the features are then gathered so the
embedding stage has all of them; then the schedule runs unchanged.

* Total tower time goes from `T_vit` to `T_vit / P + T_gather`.
* **No interleaving with the schedule**, so PP's action list is untouched and the
  ordering property the AttnRes adapter depends on -- "PP owns all NCCL, so no
  deadlock risk" -- survives. The exchange is one collective per step on its own
  communicator, issued at a point where every stage participates.
* Does NOT overlap with the bubble. It shortens the prologue rather than hiding
  it. That is the deliberate trade: P-fold reduction of the tower's serial cost,
  for none of the schedule risk.

**This is the CP image-sharding mechanism with the group swapped.** That code is
already landed and verified (`57c6728ed`): round-robin image ownership, output
lengths sized from the replicated `grid_thw` so the exchange is a fixed-shape
`funcol.all_gather_tensor` rather than an object gather, and the differentiable
collective whose transpose is the reduce-scatter that returns each rank the
gradient for the images it encoded. Swapping `_cp_group` for a PP-axis group is
the bulk of the change.

The prerequisite is the real work: **tower weights must exist on every PP
stage.** Today `pipeline_module_split` gives the tower to the `embed_tokens`
chunk and "nothing vision-side crosses a stage". Two options:

1. **Replicate the tower on every stage.** Simple, and 447.4M x 8 stages is not
   affordable at real scale.
2. **FSDP-shard the tower across the PP axis** and all-gather it for the encode.
   One all-gather of 447.4M parameters per step, amortised over M microbatches,
   against a P-fold cut in tower compute. This is the option that scales, and it
   is a new mesh interaction rather than a model-folder change.

## Design B -- true schedule interleaving

What was originally asked for. Stage k encodes microbatch j for j > k during its
warmup idle and delivers before stage 0 needs it.

Strictly better in theory: the tower's cost is hidden rather than divided. Three
costs, and the third is the one that should stop anyone reaching for this first:

* Only microbatches j >= J0 are offloadable, so the benefit is bounded by
  (M - J0)/M and vanishes for short pipelines.
* Delivery runs **against** the pipeline direction (stage k -> stage 0), and no
  such channel exists; PP's P2P is k -> k+1.
* It requires issuing sends interleaved with the schedule's action list. A send
  that does not pair with a matching recv **in the order the schedule expects**
  is exactly how PP deadlocks, and it gives up the property the AttnRes adapter
  was built around. Both of that adapter's gradient bridges are pure local
  Python plus a dict precisely to avoid this.

## Recommendation

Do Design A. It reuses verified code, keeps the schedule untouched, and delivers
the P-fold reduction that is the actual objective once the bubble framing is
dropped. Sequence it as:

1. Shard the tower across the PP axis (the real prerequisite; everything else is
   a group swap).
2. Move the CP image-sharding helper to take a group argument, so one
   implementation serves both axes.
3. A/B against the replicated path the same way CP was verified: step-1 loss and
   grad_norm bit-identical, since distributing images changes no arithmetic.

Design B is worth revisiting only after A is in and measured, and only if the
measurement shows the remaining prologue is still the bottleneck.

## What is NOT in the way

Worth recording because it was assumed to be: the vision tower under PP does not
need any change to the AttnRes cross-stage adapter. That adapter changes the
CONTENT of what crosses a PP hop (delta of committed blocks instead of the full
stack) and adds no collectives, so it is orthogonal to where the tower runs.
Verified independently: multimodal PP8xVP4 is bit-identical with the adapter on
and off, at seq 1024 through 8192.
