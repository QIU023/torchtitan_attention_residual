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

## What the gate covers, and what it silently did not

58 cells: the three arms' 54, two `vit_dep_stages=2` cells, two pp8xvp4 cells with the
bubble runtime.

The two-stage DEP pair spent several gate runs doing nothing. `run_dep2.sh` called
`run_cells.sh` with `pp4`/`pp8`, and that script resolves a cell name against
`run13_flav.sh`, where those two do not live -- they are `run_maxdeg.sh` cells. It printed
"pp4: no such cell in run13_flav.sh" and continued, so every "58/58" was really 56 passing
and 2 never executed. My own tally had a second bug that hid the first: a glob counted the
pp8xvp4 pair twice, which made 56 look like 58.

Fixed on both levels. The cells are invoked directly now and report the wiring line that
proves multi-stage DEP is real -- "DEP vision stage wiring: 1 stage(s) on this rank, roles
[head]", against `[both]` for single-stage, which is what the 54-cell gate had always been
showing. And the gate itself now walks its output tree, counts logs against the expected
58, and warns when a cell produced no log at all. Counting logs that exist rather than
cells that were expected is what let this run for as long as it did.

## The first run where all 58 cells executed

    === cell accounting ===
      logs found: 58 of 58 expected; passed: 58

    dep2 pp4  : DEP vision stage wiring: 1 stage(s) on this rank, roles ['tail']
    dep2 pp8  : DEP vision stage wiring: 1 stage(s) on this rank, roles ['head']
    pp8vp4 mm : DEP bubble runtime: 0/0 planned
    pp8vp4 lora: DEP bubble runtime: 0/0 planned

The two DEP cells report different roles -- `tail` on pp4, `head` on pp8 -- which is
stronger than a loss count: the tower is genuinely split and different ranks hold
different shares. Single-stage DEP reports `both` on every rank.

`0/0 planned` on the pp8xvp4 pair is the correct answer at seq 256, where the cost ratio
is about 14 and no idle run can pay for an encode. The cell still exercises the mechanism:
the upfront prefix runs, the drain runs, and planned-but-not-fired would warn.

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

## First step-time measurement of the hiding claim (2026-08-17)

Occupancy was all this had ever measured. This is step time, which is the only thing that
can answer "does it hide anything", at the shapes that fit in 15.5 GiB:

| shape | cost ratio | tps off -> on | occupancy |
| --- | --- | --- | --- |
| pp8 x vp2, seq 4096, local 16 | 0.493 | **OOM** | -- |
| pp8 x vp2, seq 2048, local 16 | 2.0 | 432 -> 423 (**-2.08%**) | 2/2 placed, 8 upfront, 6 synchronous, 3.9 ms each |

**The bubble runtime makes this configuration 2.08% slower.** Not a defect -- the
arithmetic of the cell explains it, and it agrees with the theory's direction:

* at r = 2.0 the hideable share is already past its peak (which is near r = 0.5) and
  heading for zero;
* of sixteen micro-batches only TWO had their encode placed in a bubble. Eight fall in the
  report's own upfront prefix, which cannot be placed because nothing precedes them, and
  six stayed synchronous. So the ceiling on any gain was 2/16 before the mechanism ran;
* against that ceiling, the plan construction and the per-forward hook are a real cost.

The configuration where theory predicts a gain is the one that OOMs. That is the same
conclusion HANDOFF_2026-08-16 reached from arithmetic (about 60 GiB per GPU), now with the
actual failure point rather than an estimate: pp8 x vp2 halves the micro-batches in flight
against pp8 x vp4 and still does not fit seq 4096 at local 16.

The theoretical best point then ran too, on four GPUs where seq 4096 fits at local 8:

| shape | cost ratio | placed share | tps off -> on |
| --- | --- | --- | --- |
| pp8 x vp2, seq 2048, mb 16 | 2.0 | 2/16 = 12.5% | 432 -> 423 (-2.08%) |
| pp4 x vp2, seq 4096, mb 8 | **0.493** | 2/8 = 25% | 3726 -> 3690 (**-0.95%**) |

Two points, and together they show the mechanism rather than just a verdict. A better cost
ratio and a larger placed share both move the number toward zero, exactly as the theory
says they should -- and neither reaches a gain.

**The binding constraint is the upfront prefix, not the cost ratio.** The report's own
design cannot place the first few micro-batches' encodes, because nothing precedes them to
anchor on. At pp=4 with 8 micro-batches that is 4 of 8 unplaceable before anything else
happens; 2 more stayed synchronous; 2 were placed. So the ceiling on any gain was 25%, and
against it a 4.0 ms encode plus the plan and hook overhead nets out negative.

Placed share is roughly (mb - pp) / mb, so it only improves with mb >> pp -- and a larger
mb is exactly what does not fit at seq 4096 in 15.5 GiB. That is the real content of the
"about 60 GiB per GPU" figure: not that the configuration fails to start, but that the
configurations which DO start leave too small a placeable share for the mechanism to pay
for itself.

What this does NOT say: that bubble scheduling cannot hide anything. Reporting -0.95% as
"the bubble does not work" would be the mirror image of reporting 8/8 occupancy as "the
bubble works" -- both read a cell-specific number as a property of the design. What is now
established is a measurement, a direction, and the specific quantity (mb/pp) that a bigger
box would change.
