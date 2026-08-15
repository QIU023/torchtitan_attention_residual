# Faithful DEP: vision encodes executed IN the schedule's idle slots

Design for `dep_exp_impl`. The prefetcher we have overlaps the encode with whatever
the main stream is doing, on a side CUDA stream. That is an approximation of the
report, not the report: sec 5.2.3 says the remaining ViT forwards are "scheduled into
pipeline bubbles". This is how to do that literally.

## Why the earlier refutation does not apply

`VIT_DEP_DESIGN_2026-08-07.md` refuted **ViT-as-a-PP-stage**: a stage sits in the
dependency chain, so its actions occupy compute slots and the bubble count is
unchanged (2019 either way). And `register_custom_function` validates against a
closed set, so no `VIT_FORWARD` action can be registered.

Neither blocks this design, because it does not make the ViT a stage and does not
register an action. It computes where the idle slots are and calls the encode there,
from our own action loop.

## The mechanism

1. **Locate the bubbles statically.** `_simulate_comms_compute` over
   `pipeline_order_with_comms` already yields, per rank, a slot list where `None` is an
   unmet-dependency idle slot. `dep_bubble_structure.py` does exactly this today.
2. **Plan.** Per rank, produce `[(slot_index, microbatch)]`: at slot i, encode
   micro-batch j. The plan is a pure function of (pp, vp, mb, schedule), all of which
   every rank knows before the step, so every rank computes the same plan. That is what
   keeps the vision collectives ordered consistently across ranks -- the property the
   side-stream route has to argue for rather than derive.
3. **Budget.** An encode costs r text-stage-forward units; a slot pays 1. Accumulate
   budget across consecutive idle slots and only place an encode where the budget
   before the consumption point covers it. This is the walk `dep_hiding_theory.py`
   already implements, reused as the planner rather than as an estimator.
4. **Execute.** Subclass `_PipelineScheduleRuntime`, keep its action semantics, and at
   planned slots call the encode on the MAIN stream before proceeding. Features land in
   the same per-micro-batch cache the prefetcher already fills, so the consumer side
   needs no change.
5. **Upfront prefix.** The first `pp` micro-batches' encodes run synchronously before
   the loop, which is the report's own design ("executed synchronously upfront"), not a
   concession.
6. **Backward.** Symmetric: the tower's backward is triggered when the spliced
   features' gradient arrives, and is placed at planned idle slots on the backward
   side.

## What is original here, and why it is inside the rule

Taking over the action loop is ours. The MECHANISM is not invented -- it is what the
report describes -- and no reference implementation exists to follow, which is the same
situation as the PP adapter. The cost is that PyTorch schedule changes have to be
tracked, since we subclass its runtime.

## Acceptance, and why it does not need the step-time win

The judgement is not a latency number: this box reports mfu 0.12%, so step time is
dominated by comms and Python, and the theoretical ceiling (~3.4% at pp8/vp2/mb32,
r=0.493) sits inside its noise. The criterion is occupancy, which the simulator counts
directly:

* **before**: N idle slots per rank, zero of them holding vision work;
* **after**: the planned slots hold vision encodes, and the idle count drops by the
  number placed.

Plus, unchanged from the prefetch work: loss identical to the DEP-off baseline step for
step, and pp8xvp4 green on the multimodal and LoRA flavors.

## Settings this is meant to run at, and why

| | value | reason |
| --- | --- | --- |
| seq_len | 4096 | puts r at 0.493, visual tokens 6.2% of the sequence; at seq 256 they are 100% and r = 14, where the hideable share is zero |
| pp | 8 | pp16 fragments each rank's idle run so a single gap cannot pay for one encode: 2/32 placed against 18/32 at pp8 |
| vp | 2 | interleaved 1F1B is what sec 5.2.2 says they run, so vp=1 is the naive baseline; vp=4 crushes the budget to a 2.65% ceiling |
| mb | 32 | the second lever, and steep: 18/32 placed at mb=32 against 2/16 at mb=16 |
| local batch | 32 | micro-batch count IS local_batch_size here; global = 32 with dp=1 on 8 ranks |
| memory | >= 60 GiB/GPU | seq 4096 x local 8 already OOMs at 15.5 GiB |
