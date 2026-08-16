# Faithful DEP: what is built, what is proven, and what needs a bigger box

Branch `dep_exp_impl`. Written so the three open items can be picked up without
re-deriving why they are open.

## What DEP is here, in three layers

**Decoupling.** `KimiK3ViTStage` occupies its own pipeline stage(s), configured by
`vit_dep_stages`. The report describes this ("splits ViT and text training into separate
stages") but gives no stage count, so a value above 1 is our generalization and must
never be reported as fidelity. Validated numerically earlier: pp2/1F1B, pp4/1F1B and
pp2/Interleaved1F1B reproduce the non-DEP loss, the last two bit-identical on grad_norm.

**Forward concurrency, two alternatives.** `KIMI_VIT_PREFETCH` issues the encode for
micro-batch m+k on a side CUDA stream during m's text compute. `KIMI_VIT_BUBBLE` instead
runs planned encodes on the MAIN stream inside the schedule's idle intervals. The
installer refuses both at once: the prefetch would satisfy every micro-batch before its
planned slot arrived, and the occupancy number would come out green for the wrong
mechanism.

**Backward.** The tower's output is spliced into the text embedding, so by default the
tower's backward runs inside the splicing stage's backward wherever the schedule put that
action. `cut_for_deferred_backward` detaches the output, splices the detached leaf, and
captures the gradient with a tensor hook; the tower's graph is replayed later at an idle
slot, with an unconditional drain at step end.

## What is proven, and what the evidence is

| claim | evidence |
| --- | --- |
| encodes run at the planned points | runtime counts them: 8/8 planned, 8 upfront, 8 synchronous |
| vision backwards run at planned slots | 24 ran, 0 drained at step end |
| the forward is numerically unchanged | loss identical step for step to the mechanism off, multimodal and LoRA |
| the backward is numerically unchanged | grad_norm identical across 8 ranks and 6 steps (multimodal) |
| no gradient is ever lost | unit tests: drain recovers when no slot comes, out-of-order replay equals one accumulated pass, `assert_empty` refuses a leak |

Measured at pp=8, `layers_per_stage=1` (32 stages, so vp=4), Interleaved1F1B, 24
micro-batches, seq 256, 8 ranks, dp=1.

## What is NOT proven: hiding

Occupancy is not hiding, and the two came apart in a way worth recording. The cell above
runs seq 256 and was given `KIMI_VIT_BUBBLE_COST_RATIO=0.493`, which `dep_cost_ratio.py`
measured at seq **4096**; at seq 256 the same probe puts one ViT forward at about **14**
text-stage forwards. So the planner was told an encode costs 0.493 units when it costs
roughly 14. The encodes ran at the planned points -- the counter is truthful about that
-- while overrunning the idle interval about 28-fold, which delays the following actions
instead of hiding anything.

With the honest ratio the planner places **0** at seq 256: rank 0 has 46 idle slots but
they are fragmented, and one encode needs 14 units of accumulated budget before its
consumer. That is the correct answer for this configuration, and the gate now passes 14.0
and treats 0/0 as normal while warning only on 0/N with N > 0 -- the plan and the
schedule disagreeing.

For reference, at the ratio that was passed: 120 placements across ranks consuming 59.2
of 312 idle slot-units, about 19% of the idle budget. That number describes a
configuration that does not exist.

## The three open items, and why they share one blocker

**1. LoRA's backward -- DONE, and it found a real defect.** Verified at seq 256:
grad_norm identical to the mechanism off, loss identical, rc=0.

It was not a formality. Under LoRA the tower's parameters are frozen, so its output does
not require grad, and the cut ran anyway and produced a detached leaf with
requires_grad=True. That turned the splicing stage's output from not-requiring-grad into
requiring it, and torch's stage_backward was dragged down a path it would not otherwise
take: "grad can be implicitly created only for scalar outputs". Full-parameter multimodal
passed throughout, because there the gradient path exists either way -- so this was
invisible to every check that did not run LoRA separately.

The fix is a missing precondition rather than a workaround: with no gradient path there is
no backward to defer, so the cut is skipped. Cutting a graph that has no gradient is not
harmlessly redundant, it manufactures a gradient path that was not there.

Worth generalising: the defect was not LoRA-specific, it was a condition the
implementation never checked. LoRA was the configuration that exposed it.

**2. Placement for the backward side.** It is greedy: one pending replay per idle
interval after a backward action. Planning it needs a model of when each vision backward
becomes runnable, which is when the text backward for its micro-batch produces the
gradient -- a schedule-dependent moment, unlike the forward's, where the pixels are
present from step entry. Worth doing only once hiding is measurable, since the greedy
choice matches the planned one in the common case.

**3. The memory bound.** The tower's forward graph must stay alive from the encode until
the replay, a longer window than the forward prefetch's. That window is what limits how
much of the backward can move, and it is unmeasured.

**The blocker for the two remaining items is the same as for hiding: a box that can hold the
configuration where hiding exists.** That needs, simultaneously:

* `seq_len 4096` -- puts the cost ratio at 0.493, visual tokens 6.2% of the sequence,
  which is the regime the report describes and the point of maximum observable effect;
* interleaved 1F1B with vp >= 2, which is what sec 5.2.2 says they run, and micro-batches
  >= stages, where the micro-batch count IS `local_batch_size`;
* so `local_batch_size >= 16` at pp8/vp2, and seq 4096 x local 8 already OOMs in 15.5
  GiB.

Roughly >= 60 GiB per GPU. On this box the three constraints cannot be satisfied at once,
which is also why the earlier step-time attempt died: first an OOM at seq 4096 x local 8,
then `Number of microbatches (1) must be greater than or equal to the number of stages`
when local batch was dropped to 1.

## What the theory says the answer will be

Worth stating before the measurement so it cannot be read as a post-hoc excuse. Latency
saving is `hidden_share(r) * ViT_share_of_step(r)`, which peaks near r = 0.5 and falls
away on both sides -- at r >= 10 the hideable share is zero, because the bubble budget is
fixed by the schedule and does not grow with the encoder's cost. The ceiling is the
usable idle fraction of the step, so a schedule good enough to have few bubbles is a
schedule with little to hide in: vp exists to remove bubbles, and DEP's bubble hiding
eats them.

That is also why high-resolution video is where this mechanism helps least, not most: it
raises r. The report's claim survives because dynamic CP divides the per-rank ViT cost by
the CP degree first, and because a 1M-token context makes the text side enormous -- the
two halves of sec 5.2.3 in the order they are written.
