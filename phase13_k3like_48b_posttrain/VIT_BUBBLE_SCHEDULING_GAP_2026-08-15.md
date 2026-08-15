# Gap: the ViT's passes are not scheduled into pipeline bubbles

Read off the tech report's sec 5.2.3, "Encoder computation in PP bubbles". Recorded
as a gap rather than a nice-to-have, because it changes what DEP is for.

## What the report says

> In Kimi K2.5, we introduced the Decoupled Encoder Process (DEP), which splits ViT
> and text training into separate stages and balances vision forward and backward
> passes across PP stages. We observe that, under the interleaved 1F1B pipeline
> schedule, the text forward passes of the first PP micro-batches are all scheduled
> at the very beginning, while the text backward passes of the last PP micro-batches
> finish only at the very end. We therefore further decompose the ViT computation.
> The ViT forward passes of the first PP micro-batches are executed synchronously
> upfront, the remaining forward passes are scheduled into pipeline bubbles, and the
> backward passes are handled analogously. As a result, most of the ViT computation
> is hidden within pipeline bubbles, largely eliminating the effective overhead of
> the vision encoder.

Two things follow that we had wrong.

**The report gives no stage count.** DEP is K2.5's, cited, and described only as
"separate stages" plus balancing across PP stages. Our `vit_dep_stages` knob is our
own generalization; a value above 1 is not a K3 configuration and should never be
reported as fidelity to the report.

**Stage placement is not the mechanism K3 relies on.** The report's answer to the
vision encoder's cost is scheduling its passes into the bubbles the 1F1B schedule
already has, not giving it more stages.

## Why a whole stage for the tower is the wrong shape, measured

| | parameters |
| --- | --- |
| MoonViT-V2 | 447M |
| one K3 text layer (2.8T / 93) | ~30B |
| one PP stage at pp=8 (~12 layers) | ~360B |

The tower is **1.5% of a single text layer and 0.12% of a pipeline stage**. It does
not need multiple stages to fit -- it barely registers. What it actually costs is
activation memory on long videos and large images, which sec 5.2.3 addresses with
dynamic CP (partition along the patch dimension, gather-KV across CP ranks -- the
part we do implement and which 14 of the 56 gate cells execute), and time on the
critical path, which is what bubble scheduling addresses.

So the problem with `vit_dep_stages=1` is not that the tower is too big for one
stage. It is that a stage is too much to spend on it: vision stages come out of the
text budget, so pp=8 with one vision stage leaves 7 stages to carry 93 text layers
(13.3 per stage against 11.6), a 15% text-side imbalance bought in exchange for a
stage holding 0.12% of the parameters. Splitting the tower across two stages makes
that worse, not better -- two nearly empty stages instead of one.

That is the argument for prioritizing bubble scheduling over more stages. The
two-stage cells now in the gate (56 cells, mm_full arm, pp4 and pp8) exist to keep
our own implementation honest, not because two stages is the target.

## What implementing it would take here

Our PP path is the cross-stage adapter in `pipeline_adapter.py`. The pieces:

1. **A schedule hook.** The bubbles are known to the schedule, not to the model:
   which micro-batch indices leave a stage idle follows from the 1F1B (or
   interleaved 1F1B) ordering. torchtitan drives schedules through PyTorch's
   `PipelineSchedule`, so the insertion point is wherever the per-microbatch action
   list is materialized -- the ViT work has to become an action the schedule can
   place, not a call inside the model's forward.
2. **Splitting the tower's forward per micro-batch.** Today `encode_images` runs
   the whole tower for the whole micro-batch inside the stage that owns it. Bubble
   scheduling needs it callable per micro-batch and interleavable, which means the
   tower's forward has to be re-entrant with respect to micro-batch state and its
   output has to be parked until the text stage that splices it runs.
3. **The same for backward.** The report says the backward passes are handled
   analogously. That is the harder half: the tower's gradients have to be
   accumulated across bubbles without holding every micro-batch's activations,
   which interacts with the activation offload policy.
4. **A measurement.** The claim to verify is not correctness but hiding: step time
   with the tower's work in bubbles against step time with a dedicated stage. Our
   PP-adapter overhead figure (+2.7% step time on PCIe) is the precedent for how to
   state it -- and note it is a step-time claim, never to be conflated with the
   report's algorithmic overhead figures.

Until 1 and 2 exist, `vit_dep_stages` remains the only lever we have, and its cost
is the imbalance above.
