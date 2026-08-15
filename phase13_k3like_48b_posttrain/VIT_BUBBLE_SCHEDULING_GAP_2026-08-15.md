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

**Stage placement is not the mechanism K3 describes.** Their answer to the vision
encoder's cost is scheduling its passes into the bubbles the 1F1B schedule already
has, not giving it more stages. Whether that is reachable at all is settled further
down, and the answer is no through any route we are willing to take.

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

So neither one stage nor two is a good shape for the tower, and the mechanism that
would be is closed off (below). The two-stage cells now in the gate (56 cells,
mm_full arm, pp4 and pp8) exist to keep our own implementation from regressing
unnoticed -- multi-stage DEP had never been gated -- not because two stages is a
target.

## This was already investigated, and the route is closed

Superseding the design sketch this file first carried. `VIT_DEP_DESIGN_2026-08-07.md`
records a NEGATIVE RESULT from 2026-08-08 that answers the question, and it is
stronger evidence than anything in this file:

* `_PipelineScheduleRuntime.register_custom_function` validates `computation_type`
  against a closed set (FORWARD, FULL_BACKWARD, BACKWARD_INPUT, BACKWARD_WEIGHT,
  OVERLAP_F_B, UNSHARD, RESHARD, REDUCE_GRAD). **A `VIT_FORWARD` action cannot be
  registered**, so "insert ViT work into the bubble" is not expressible through the
  schedule at all. The sketch this file originally proposed -- hooking wherever the
  per-microbatch action list is materialized -- was exactly that, written without
  reading the earlier finding.
* Measured with torch's own dependency simulator (`_simulate_comms_compute` over
  `pipeline_order_with_comms`, bubbles counted as `None` slots), pp8 /
  Interleaved1F1B / 16 virtual stages / 16 microbatches: **non-DEP 2019 bubbles, DEP
  2019 bubbles -- identical**, with 64 vision slots on rank 0 and zero on the other
  seven. Making the ViT a stage swaps a text stage for a vision stage: same stage
  count, same dependency graph, same stalls, and the vision work occupies a COMPUTE
  slot rather than a bubble.

**Scaling the tower proportionally to the pretraining scale does not rescue it.**
That was the natural next thought and it addresses a different quantity: the
simulator counts dependency stalls, which do not depend on how long each slot takes.
A proportionally scaled tower would make the vision cost *visible* -- worth doing if
we ever want to state the overhead -- but it cannot move vision work from a compute
slot into a bubble, because the placement is structural.

## What that leaves

Standing, and worth keeping: ViT and text as separate stages. That is numerically
exact (pp2/1F1B, pp4/1F1B and pp2/Interleaved1F1B reproduce the non-DEP loss, the
last two bit-identical on grad_norm), and it makes the tower's placement a
configuration rather than a hard-coded attachment to the embed stage.

Not standing, and not to be claimed: any hiding of vision compute.

The only remaining reading of the report -- software-pipelining the ViT against the
text pipeline on the same rank, on a side stream, rather than inserting it into the
dependency chain -- needs a mechanism outside the schedule, and that gives up the
property the AttnRes adapter is built around: PP owns all NCCL. Issuing sends outside
the schedule's action list is how pipelines deadlock. So implementing it is not
merely original work; it trades away a property the rest of the PP path depends on.

**Recommendation: do not implement it.** The report describes it in prose only, and
the earlier analysis notes their "stage" may not even mean a PP stage, so there is no
concrete reference to follow -- but unlike the PP adapter, where the verbal
description had no conflicting constraint, here the one available route costs a
property we rely on. Record the bubble claim as unreproduced, with the simulator
numbers, and leave DEP as the decoupling it demonstrably is.
