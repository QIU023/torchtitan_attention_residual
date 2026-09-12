# Reply to Tianyu's numerics comment on PR 4312 (id 3985333653, `parallelize.py` L267), 2026-09-11

Corrected 2026-09-12: section 4 by the step-1 bitwise campaign (`phase13_k3like_48b_posttrain/PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md`); steps past the reference's memorisation removed (CLAUDE.md numerics-table rule); the 4 x H100 PCIe rerun of 2026-09-12 added to section 3.

One protocol, one model depth, two boxes. Everything measured on the review head
(`pp_review4`, on upstream main `d9ca9e55a`) with main's 24-layer `debugmodel`, 1024 tokens per
step as four 256-token micro-batches, seed 42, `--debug.deterministic`, one shared seed
checkpoint, one warm step, 100 measured steps. The five cells are the same on both boxes.

Why not PR 4500's 256 tokens per step: measured, not assumed. A micro-batch is not tied to the
context length (`trainer.py` requires only that it divide by `tp * (2 * cp)`, which is 1 for these
cells, and the debug seq len is 2048), but the multimodal collator refuses a micro-batch narrower
than the rows it packs -- `pad_len = num_tokens_per_batch - input_ids.shape[0]` must not go
negative, `mm_collator.py:101`. On this flavour's `cc12m-test` stream, 64, 96, 128 and 192 tokens
per micro-batch are all refused and 256 is the first that runs (`dp1` at 4 x 256 reads
`12.51200` at step 1). `pp2 x vp2` is four stages and so needs at least four micro-batches, which
puts the floor for the four-stage cells at 4 x 256 = 1024 tokens per step. That is this table's
protocol, and it is the honest answer if the difference from #4500 is raised.

(An earlier draft of this note asserted the same 1024 figure from a rule that does not exist --
that a micro-batch is one sequence of the context length. That was the right number for the wrong
reason; it was retracted and re-established by the measurement above.)

Paste the block between the markers. The CP replication below it is a separate answer and goes in
its own comment; everything under "Background, superseded" is working material, not for posting.

--- PASTE ---

Four parts: what step 1 actually reads and where its difference comes from, the per-parameter
picture at step 1, the trajectory tables with a noise band around them, and what the transport
flag does.

## 1. Step 1, and a correction to what this PR claimed

The PR said step 1 was bit-identical across pipeline shapes. That was read off printed losses,
and it is wrong. At full precision, on the 33-layer shape, `dp1` reads `12.336345672607422` and
`pp2` reads `12.336344718933105` -- one float32 unit in the last place apart, `9.5e-07` absolute
and `7.7e-08` relative -- and their total gradient norms are `23.25` and `23.125`, one bf16 unit
apart. Where the printed five decimals agree, they are rounding a difference of this size, not
showing its absence. We are correcting that here rather than leaving it for you to find.

**Where the one ulp comes from: we had an explanation, tested it, and it failed.** Kimi's attention
residual groups layers into blocks of 12. Inside a block each layer adds into a running partial sum; a
finished block joins the stack. Whether a partial sum has to cross the stage boundary depends only on
where the split falls, and the two depths differ in exactly that way: at 24 layers `pp2` splits after
layer 11, so stage 1 opens at layer 12, a block start, and only finished blocks cross; at 33 layers it
splits after layer 16, five layers into a block, so the partial sum of layers 12 to 16 crosses and is
rebuilt on the far side. That predicted the one ulp at 33 layers and zero at 24, so we pinned the
33-layer split to open stage 1 at layer 24, a block start where no partial sum crosses, and predicted
bit-identity.

It did not happen. The pinned cut reads `12.375029563903809`, the same value as the mid-block cut, one
ulp from its `dp1`. The partial-block explanation of the step-1 loss is dead, and we are reporting it
dead rather than quietly replacing it.

What survives is smaller and measured: across all four configurations -- two depths, each with the cut
on and off a block start -- the step-1 loss agrees with its own single-GPU reference to within one
float32 unit in the last place, and **within a depth the position of the cut makes no difference to it
at all**. Whether a depth lands on zero or one ulp tracks the depth, not the cut, and we do not
attribute it further. We have not run a depth sweep to explain it, because at this granularity such a
sweep would produce a pattern we could not support.

## 2. Step 1, per parameter

Step-1 gradients of every parameter, `dp1` against `pp2` with the cache on, 33 layers, 1,002 tensors,
both cells resumed from the same step-0 checkpoint. Two micro-batches of 256 tokens, so there is no
accumulation-order term: the only difference between the two runs is the pipeline. Measured on the head
this PR was reviewed at (`75045fed5`). Per tensor: the fraction of elements that differ, the median
difference among them in bf16 units in the last place, and the relative difference of the norms.

| group | tensors | bitwise | diff frac | median ulps | normrel median | normrel max |
| --- | --- | --- | --- | --- | --- | --- |
| lm_head, norm, output_res_* | 4 | 4 | 0 | - | 0 | 0 |
| layer 32 (KDA) | 24 | 24 | 0 | - | 0 | 0 |
| layer 31 (MLA) | 24 | 23 | 2 elements | 1.03 | 0 | 4.8e-7 |
| layers 30, 29, 28 | 90 | 90 | 0 | - | 0 | 0 |
| layer 27 (MLA) | 24 | 18 | 0.001 | 1.38 | 0 | 2.5e-4 |
| layer 26 | 30 | 1 | 0.024 | 1.71 | 3.6e-4 | 1.7e-3 |
| layer 25 | 30 | 0 | 0.140 | 1.79 | 1.5e-3 | 2.8e-3 |
| layer 24 (block 2 start) | 30 | 0 | 0.283 | 1.95 | 3.1e-3 | 4.3e-3 |
| layers 23 .. 17 (stage 1) | 24-30 each | 0 | 0.21-0.31 | 1.83-1.91 | 2.0e-3 .. 2.6e-3 | 3.3e-3 .. 4.2e-3 |
| layer 16 (stage 0, first across the boundary) | 30 | 0 | 0.348 | 1.97 | 3.1e-3 | 3.7e-3 |
| layers 15 .. 13 | 24-30 | 0 | 0.44-0.60 | 2.05-2.59 | 3.4e-3 .. 5.8e-3 | 5.5e-3 .. 8.2e-3 |
| layer 12 (block 1 start) | 30 | 0 | 0.762 | 5.15 | 1.9e-2 | 2.4e-2 |
| layers 11 .. 1 | 24-38 | 0 | 0.65-0.83 | 2.5-4.2 | 6.0e-3 .. 1.3e-2 | 7.4e-3 .. 2.5e-2 |
| layer 0 | 29 | 0 | 0.880 | 5.82 | 2.1e-2 | 5.0e-2 |
| tok_embeddings | 1 | 0 | 0.000 (sparse rows) | 4.11 | 1.1e-2 | 1.1e-2 |
| vision encoder (stage 0) | 6 | 0 | 0.818 | 4.92 | 1.4e-2 | 2.0e-2 |

Three things in this table are what a bug would not produce. The top of the model -- `lm_head`, the final
norm, the output aggregation, and layers 32, 30, 29 and 28 -- is bit-identical, so the backward starts from
identical values. The first difference is two elements of one tensor, layer 31's `wq_b`, at one ulp. And
from there it grows smoothly layer by layer. The one sharp step is at layer 12, a block start, where the
median difference goes from about 2.6 ulps to 5.15 and the attention residual changes what the stack
holds; layer 24, the other block start, rises more gently. Across the stage boundary between layers 17
and 16, where the transport lives, nothing steps: the median goes from 1.91 ulps to 1.97. A wrong tensor on the wire would show up at
the boundary, large, in every element it touched.

For scale: upstream's own pipeline on upstream's own models, with the same harness and shared step-0
checkpoint on this box (2026-09-05, 4,096 tokens per step in 256-token micro-batches), reads `4.11391`
for `llama3_debugmodel` pp2 and `4.11384` for its dp1 at step 10 (1.7e-5), and `3.88192` and `3.88273`
for `deepseek_v3_debugmodel` (2.1e-4). The debug K3 flavour is the sensitive one; its no-pipeline reordering row in section 3
shows that sensitivity without any pipeline in it.

### 2b. The cut is not where the difference lives

The table above is one depth with one split. To ask whether the pipeline boundary is where the
difference is produced, we moved the boundary and left everything else alone. Four configurations, two
depths, the cut on and off a block start, each `pp2` cell against its own `dp1` from the same step-0
checkpoint; per layer we report the median relative L2 distance of the gradients,
`||g_pp2 - g_dp1|| / ||g_dp1||`, which needs no elementwise threshold and has no near-zero denominator.

| depth | stage 1 opens at | that layer is | spikes appear at | a spike at the cut? |
| --- | --- | --- | --- | --- |
| 24 | 12 (default) | a block start | 12 | coincides |
| 24 | 13 (pinned) | mid-block | 12 | no |
| 33 | 17 (default) | mid-block | 24, 12 | no |
| 33 | 24 (pinned) | a block start | 24, 12 | coincides |

Blocks are 12 layers, so layers 0, 12 and 24 open one. Every spike in every configuration sits on a
block start; no spike ever sits on a cut that is not also a block start. In the 24-layer mid-block cut,
layer 13 reads `7.02e-03`, indistinguishable from its neighbours, while layer 12 still spikes at
`1.97e-02`. In the 33-layer default cut, layer 17 reads `8.78e-03` between neighbours of `8.28e-03` and
`8.86e-03`.

The 24-layer pair says it more sharply than the shapes do. Cutting at 12 and cutting at 13 put different
layers on different ranks -- per-rank memory `8.20` / `7.76` GiB against `8.39` / `7.36` GiB, and the
generated split differs, both checked -- and the two runs' step-1 gradients are **bit-identical on all
750 tensors**. Moving the pipeline boundary changed which rank computes what, and changed no number
anywhere in the model.

That is the answer to how we know this is not a bug in the boundary machinery. A fault in what crosses
the wire, in how the stack is rebuilt, or in which columns are routed would show up at the cut. Nothing
shows up at the cut. What does show up is organised by the model's own structure -- the layers where the
block stack gains a column -- at the same layer indices no matter where the model is divided, and inside
those layers the largest distances are on `attention_res_norm` and `attention_res_proj`, the two modules
that read the stack.

Two controls make that comparison readable at all. Two `dp1` runs with the same seed and checkpoint are
bit-identical on all 750 tensors, so a `dp1`-against-`pp2` difference is produced by the pipeline and not
by the runtime. And each comparison is gated on its own step-1 loss: the profile is printed only if
`dp1` reproduces its reference and `pp2` is within one ulp of it, otherwise the driver reports that it
withheld the numbers.

## 3. The trajectory, with a band around it

The bottom row of each table is one ordering of the four accumulation groups. There are 24 of
them, and the ordering alone -- no pipeline anywhere -- is worth a band, not a point. Four of the
24 measured on the 5060 Ti box, all against the identity ordering, everything else equal:

| ordering of the four accumulation groups | step 10 | step 20 |
| --- | ---: | ---: |
| identity (the reference) | `3.297890` | `3.337080` |
| reversed | `3.349840` (+1.58%) | `3.396910` (+1.79%) |
| `1,3,0,2` | `3.214190` (-2.54%) | `3.501760` (+4.93%) |
| `2,0,3,1` | `3.078880` (-6.64%) | `3.399090` (+1.86%) |
| **band** | **-6.64% .. +1.58%** | **0.00% .. +4.93%** |

Against that band, the pipeline cells on the same box:

| cell | step 10 | step 20 |
| --- | --- | --- |
| `pp2` | `+13.7%`, **outside the band** | `+4.04%`, inside |
| `pp2 x vp2`, cached | `-3.72%`, inside | `+2.67%`, inside |
| `pp2 x vp2`, whole-stack | `+0.72%`, inside | `+0.74%`, inside |

We are not going to pretend that reads cleanly. At step 20 every pipeline cell sits inside the
band that reordering alone produces. At step 10 two of the three do and `pp2` does not: `+13.7%`
against a band whose upper edge is `+1.58%`. Four samples of 24 orderings is a lower bound on the
band, so a wider sample may contain it, and the H100 box puts the ordering row above `pp2` at the
same step (`4.27%` against `3.61%`) -- but on this box, at this step, it is outside what we
measured, and that is the honest statement. It is also why the correctness argument in sections 1
and 2 is made at step 1 in the gradients and not here: at step 10 the quantity is already a
chaotic function of a one-ulp difference, and a band of four samples is a weak instrument for it.


**The controls.** Five cells, 100 steps, seed 42, deterministic, every cell resumed from the SAME step-0 seed checkpoint so the weights they start from are bit-identical, 1024 tokens
per step as four 256-token micro-batches (the smallest this flavour's loader accepts, see the
note under the tables), on main's 24-layer `debugmodel`. `dp1` is the reference; its absolute loss
is printed at every step, so where it stops being a meaningful denominator is visible rather than
asserted.

**8 x RTX 5060 Ti.** Loss, then total gradient norm; `dp1`'s absolute value is printed at every step, so the reference is visible where the percentages are taken.


| cell | step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: |
| dp1 (reference) | `12.593920` | `3.297890` | `3.337080` |
| pp2 | `12.593920` (+0%) | `3.750270` (+13.7%) | `3.471810` (+4.04%) |
| pp2 x vp2, cached transport | `12.593920` (+0%) | `3.175340` (-3.72%) | `3.426070` (+2.67%) |
| pp2 x vp2, whole-stack transport | `12.593920` (+0%) | `3.321520` (+0.717%) | `3.361730` (+0.739%) |
| **dp1, accumulation groups reversed (no pipeline)** | `12.593920` (+0%) | `3.349840` (+1.58%) | `3.396910` (+1.79%) |


| cell | step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: |
| dp1 (reference) | `18.6250` | `7.4688` | `4.0625` |
| pp2 | `18.6250` (+0%) | `7.4688` (+0%) | `4.9062` (+20.8%) |
| pp2 x vp2, cached transport | `18.6250` (+0%) | `5.1562` (-31%) | `4.0312` (-0.77%) |
| pp2 x vp2, whole-stack transport | `18.6250` (+0%) | `4.8125` (-35.6%) | `4.5938` (+13.1%) |
| **dp1, accumulation groups reversed (no pipeline)** | `18.6250` (+0%) | `6.1875` (-17.2%) | `3.9531` (-2.69%) |

**2 x H100 PCIe.** The same five cells and the same protocol; `dp1` of this box is its own reference.

| cell | step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: |
| dp1 (reference) | `12.605700` | `3.114620` | `3.373330` |
| pp2 | `12.605700` (0%) | `3.227050` (3.61%) | `3.288290` (2.52%) |
| pp2 x vp2, cached transport | `12.605700` (0%) | `3.150940` (1.17%) | `3.349300` (0.712%) |
| pp2 x vp2, whole-stack transport | `12.605700` (0%) | `3.514970` (12.9%) | `3.281700` (2.72%) |
| **dp1, accumulation groups reversed (no pipeline)** | `12.605700` (0%) | `3.247610` (4.27%) | `3.295370` (2.31%) |

| cell | step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: |
| dp1 (reference) | `18.6250` | `5.4375` | `3.9844` |
| pp2 | `18.7500` (0.671%) | `5.6875` (4.6%) | `3.7344` (6.27%) |
| pp2 x vp2, cached transport | `18.6250` (0%) | `3.7188` (31.6%) | `4.0625` (1.96%) |
| pp2 x vp2, whole-stack transport | `18.6250` (0%) | `6.0312` (10.9%) | `3.6719` (7.84%) |
| **dp1, accumulation groups reversed (no pipeline)** | `18.6250` (0%) | `6.6562` (22.4%) | `4.2500` (6.67%) |

The tables stop at step 20: after it the reference memorises the debug set, and a percentage against it measures that, not the parallelism.

What the rows say, on both boxes, and nothing more.

- Step 1 prints the same loss in every cell on each box. What that print is worth is section 1: one float32 unit in the last place, not identity.
- The bottom row has no pipeline in it at all: it is `dp1` with the four accumulation groups consumed in the opposite order, everything else equal. On the H100 box it reads `4.27%` at step 10 and `2.31%` at step 20; on the 5060 Ti box `1.58%` and `1.79%`. That is the size of a pure association change on this flavour, with no pipeline available to blame.
- Rerun on 4 x H100 PCIe at the PR head `dbc425403` (2026-09-12): every H100 value above reproduces to the printed digit. A second no-pipeline control changes only how the four micro-batch gradients accumulate -- in float32, rounded once, as the pipeline's FSDP does, instead of in bf16 after each micro-batch: `+3.85%` at step 10, `-1.72%` at step 20.
- At step 20, in the readable range: `pp2` reads `2.52%` (H100) and `4.04%` (5060 Ti), the cached `vp2` `0.712%` and `2.67%`, the whole-stack `vp2` `2.72%` and `0.739%` -- every one of them inside the ordering band measured above. Step 10 is where `pp2` on this box is not; that is stated above and not softened here.
- At step 10 the two boxes order the cells differently: the H100 box has the whole-stack `vp2` largest at `12.9%` with `pp2` at `3.61%`, this box has `pp2` largest at `13.7%` with the whole-stack `vp2` at `0.717%`. That is single-sample scatter, and the noise band below is its scale: one ordering of the accumulation groups against another, with no pipeline anywhere, spans the same range. The widest pipeline-to-floor gap in the readable range is `pp2` at step 10 on this box, `13.7%` against a floor sample of `1.58%`; on the H100 box the floor is the larger of the pair at the same step (`4.27%` against `3.61%`). Read them against the band, not against each other.
- The two transports do not agree with each other at `vp2` on either box, and they agree at every printed step at one stage per rank on both. That is the flag's whole effect: at one stage per rank the delta is the whole stack, so the two are the same code path; at two, the cached path assembles received blocks next to locally held ones and the same contributions are summed in a different association.

**A reordering cannot move the step-1 loss, and pipelining can.** The step-1 loss is computed from
the step-0 weights, before the optimizer has consumed any gradient, so permuting the accumulation
groups cannot change it by construction -- for the ordering cells the load-bearing evidence is the
step-1 gradient norm and the trajectory from step 2 onwards. Pipelining is a different mechanism:
it changes the forward path itself, which is why `pp2` at 33 layers moves the step-1 loss by one
float32 unit in the last place where a reordering moves it not at all. Both belong in the reading of these
tables, and they should not blur into "step 1 agrees".

## 4. What the transport flag does

**How the flag changes accumulation.** At one stage per rank, not at all: no rank receives a block twice, the two settings are one code path, and they measure bitwise. With two stages per rank the cache changes which stage adds which gradient. Without it every hop hands the running gradient of the stack back to the previous stage, which keeps adding its own layers' reads onto it -- the order autograd uses on one GPU. With it, a stage that already holds a block on its rank reads it from the rank store, sums its own reads of that block from zero and deposits the subtotal; the stage that brought the block onto the rank adds the deposit when its backward runs. Same terms, grouped differently, and only for blocks read from a store. Measured at step 1 on the 24-layer model (pp2 x vp2, 8 x RTX 5060 Ti, every parameter's gradient): cache on and off are bitwise on layers 12-23 and the head; the parameters that produce a block read from a store (layers 0-11, the embeddings, the vision tower) differ at a median of 2-3 bf16 ulps. Deleting one deposit, a real bug in this path, moves the same tensors by a relative L2 of 0.7-0.9, about a hundred times more.

**How PP changes accumulation against one GPU.** Two differences, neither in the transport. First, micro-batch accumulation: torch pipelining turns FSDP gradient sync off until the last backward, so FSDP2 adds the four micro-batch gradients in float32 and rounds once, where one GPU with gradient accumulation rounds to bf16 after every micro-batch; two micro-batches agree (a two-term bf16 sum is exact in float32), four do not. With `training.mixed_precision_reduce=bfloat16` (upstream #4597) both paths accumulate in bf16, and one GPU accumulating either way is bitwise for 100 steps. Second, a residual that starts inside the last layer's attention backward on the last stage, before any gradient has crossed a stage boundary: with the accumulation matched, `lm_head`, the final norms and the output aggregation are bitwise and the first differing tensors are that layer's attention projections. It is identical in pp2 and in pp2 x vp2 with the cache on and off, and does not change with flex attention's autotuning or activation checkpointing turned off; its source is not identified, and it is not the transport. The earlier version of this paragraph said the stage boundary adds a rounding step; that was wrong -- with the cache off the handed-back gradient continues the same running sum. Both effects are bf16-ulp perturbations at step 1; what turns them into percents by step 10 is this flavour's sensitivity, which the no-pipeline controls in section 3 measure.
--- PASTE END ---


## A runtime fact recorded on the way

**A runtime fact, recorded and not chased.** In the (void) forward dumps the single-GPU path made
380 instrumented calls to the attention residual on one step and the pipeline path made 452, a
difference of 72, at 24 layers with four micro-batches. The two paths therefore do not recompute
the same amount under activation checkpointing. That is worth knowing next to the activation
save/release question from the previous round; the 72 is not investigated here.

**Why the forward A/B is not the instrument.** Those dumps were keyed by call index, and with the
two paths making different numbers of calls the pairing compared different work: the comparison
reported distances of about 2^31 units in the last place, which is an instrument reporting on
itself. The per-parameter gradient profile pairs by parameter name instead, so the call-sequence
problem cannot reach it, and it is the instrument that answers the correctness question anyway. If
a forward A/B is rebuilt it will pair by (layer, site, micro-batch) with an explicit recompute
flag.

## Background, superseded

Working subsections from the boundary investigation, kept for the record; section 2b above is the version to read.

### 2c. Both depths on one instrument: the steps are at block starts, not at the stage boundary

The 33-layer profile was re-measured with the instrument and gates used at 24 layers. Its gate
passed: `dp1` `12.375030517578125` against `pp2` `12.375029563903809`, one float32 unit in the last
place apart, `dp1` reproducing its reference `12.37503`. Median relative L2 per layer, from the last
layer down to the first; the stage boundary is at layer 17, five layers inside a block.

| layer | median | layer | median | layer | median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.83e-03 | 21 | 7.08e-03 | 10 | 1.09e-02 |
| 31 | 2.93e-03 | 20 | 7.81e-03 | 9 | 1.06e-02 |
| 30 | 4.09e-03 | 19 | 6.83e-03 | 8 | 1.11e-02 |
| 29 | 4.40e-03 | 18 | 8.28e-03 | 7 | 1.08e-02 |
| 28 | 4.77e-03 | **17 (boundary)** | **8.78e-03** | 6 | 1.26e-02 |
| 27 | 4.18e-03 | 16 | 8.86e-03 | 5 | 1.34e-02 |
| 26 | 5.53e-03 | 15 | 8.37e-03 | 4 | 1.38e-02 |
| 25 | 6.42e-03 | 14 | 9.61e-03 | 3 | 1.26e-02 |
| **24** | **1.47e-02** | 13 | 1.01e-02 | 2 | 1.40e-02 |
| 23 | 5.83e-03 | **12** | **1.96e-02** | 1 | 1.46e-02 |
| 22 | 6.86e-03 | 11 | 9.22e-03 | 0 | 2.03e-02 |

This corrects the reading of the 24-layer profile above. Taking both depths together, with a spike
defined as a layer more than 1.8 times the mean of its two neighbours:

| depth | spikes at | block start? | stage boundary? | value at the boundary itself |
| --- | --- | --- | --- | --- |
| 33 layers | 24, 12 | yes, yes | no, no | layer 17: `8.78e-03` against neighbours `8.28e-03` and `8.86e-03` -- no step |
| 24 layers | 12 | yes | yes | layer 12: `1.97e-02` against `7.02e-03` and `6.22e-03` |

The block size is 12, so layers 0, 12 and 24 open blocks. Every spike sits on one. The stage
boundary produces no step where it does not coincide with a block start: at 33 layers the boundary
falls at layer 17, and layer 17 is indistinguishable from its neighbours. At 24 layers the only
spike is at layer 12, which is both the boundary and a block start -- which is why that profile,
read alone, looked like a boundary effect. It is not one.

So neither of the two mechanisms on the table survives as stated. A partial block crossing the wire
predicts a step at layer 17 of the 33-layer model, and there is none. Re-materialising the stack at
the boundary predicts the same thing, and there is none. What the data localises is the block
structure itself: the difference between the two paths concentrates at block starts, at the same
three layer indices regardless of where the model is cut. That a block start is where the stack
gains a column is a candidate explanation for why, not something these profiles establish. At 24 layers
the largest tensors in the whole model are that layer's `attention_res_norm` and
`attention_res_proj`, the two modules that read the stack.

Layer 0 is the largest at both depths (`2.03e-02` and `2.24e-02`), and it discriminates nothing: it
is both the deepest point of the backward pass and a block start.

### 2f. Where the cut falls does not matter; where the blocks start does

Prediction A, registered before these cells: with the cut moved off a block start the spike stays
where the blocks are. It holds at both depths, and at 24 layers it holds more strongly than it was
stated.

| depth | cut | spikes at | spike at the cut? |
| --- | --- | --- | --- |
| 24 layers | 12, a block start | 12 | coincides |
| 24 layers | 13, mid-block | 12 | **no** |
| 33 layers | 17, mid-block | 24, 12 | **no** |
| 33 layers | 24, a block start | 24, 12 | coincides |

Blocks are 12 layers, so layers 0, 12 and 24 open one. Every spike in all four configurations sits
on a block start. No spike ever appears at a cut that is not one, and the profile of the 24-layer
mid-block cut is the same shape as its default: layer 12 at `1.97e-02` between neighbours of
`7.02e-03` and `6.22e-03`, and layer 13 -- the cut -- at `7.02e-03`, indistinguishable from its
surroundings.

The 24-layer pair is stronger than a matching shape. The two `pp2` runs, one cut at layer 12 and one
at layer 13, hold different layers on different ranks -- their per-rank memory differs, `8.20` and
`7.76` GiB against `8.39` and `7.36` -- and their step-1 gradients are **bit-identical on all 750
tensors**. Moving the stage boundary changed which rank computes what, and changed no number
anywhere. At 33 layers the corresponding pair is close but not bitwise, so this is stated for the
depth where it was measured.

What the four configurations support: the step-1 gradient difference between one GPU and two is
organised by the model's block structure and not by the pipeline's cut. That a block start is where
the stack gains a column remains a candidate explanation for why, not something these measurements
establish.

### 2e. The step-1 loss: closed, and the partial-block reading of it killed

Prediction B was that the step-1 loss would return to bit-identical when the cut was moved onto a
block start, because no partial block would cross. It failed, on the falsifier registered before
the cell ran.

| configuration | cut | partial block crosses | step-1 loss | distance from its own dp1 |
| --- | --- | --- | --- | ---: |
| 24 layers, default | layer 12, a block start | no | `12.593917846679688` | 0 ulp |
| 33 layers, default | layer 17, mid-block | yes | `12.375029563903809` | 1 ulp |
| 33 layers, pinned | layer 24, a block start | no | `12.375029563903809` | 1 ulp |
| 24 layers, pinned | layer 13, mid-block | yes | `12.593917846679688` | 0 ulp |

Moving the 33-layer cut onto a block start returned exactly the same loss as the mid-block cut, not
a bit-identical one; moving the 24-layer cut off a block start likewise returned exactly the same
loss as before, still bit-identical. Within a depth the cut position makes no difference to the
step-1 loss at all. The zero-against-one difference tracks the depth, not the cut, and whatever
decides it is not whether a partial block crosses the wire.

**This is where B stops.** The statement that survives is the one the table supports: across every
configuration measured, the step-1 loss agrees with its own single-GPU reference to within one
float32 unit in the last place -- zero at one depth, one at the other, on either side of a block
boundary. The distinction between zero and one unit in the last place is the distinction between a
value and the next representable value, and we have no measurement that attributes it to anything.
We are not running a depth sweep to explain it: a sweep would likely produce a pattern that invites
an explanation we could not support. The forward is identical to within one unit in the last place;
that is the claim, and it is stronger than a story about why it is sometimes zero.

### 2d. Predictions registered before the boundary cells report (2026-09-11)

The two profiles separate the evidence into two phenomena that do not compete, and the six boundary
cells test both. Writing the predictions and their falsifiers down before the cells land, so the
reading cannot be chosen afterwards.

**A. Where the step-1 gradient difference concentrates: block starts, not the cut.** Observed:
spikes at layers 0, 12 and 24 at both depths, and the 33-layer cut at layer 17 invisible against its
neighbours. The candidate explanation -- that a block start is where the stack gains a column -- is
a candidate and labelled as one; the observation is the spike positions.

| cell | prediction | falsifier |
| --- | --- | --- |
| 24 layers, stage 1 pinned to start at layer 13 (mid-block cut) | the spike stays at layer 12; no spike at 13 | a spike at 13 |
| 33 layers, stage 1 pinned to start at layer 24 (block-start cut) | spikes stay at 24 and 12, no new feature, otherwise the same shape as the default cut | a new feature at the cut, or a spike that moves |

If a spike follows the boundary in either cell, the block-start reading is wrong and the cut matters
after all.

**B. Whether the step-1 loss moves: whether a partial block crosses the cut.** Observed: 24 layers
cut on a block start, zero float32 units in the last place; 33 layers cut five layers inside a
block, one unit.

| cell | prediction | falsifier |
| --- | --- | --- |
| 33 layers cut at 24, no partial crosses | zero ulp, bit-identical with its own dp1 | any non-zero distance |
| 24 layers cut at 13, a partial crosses | one ulp, no longer bit-identical | zero ulp |

Either failing kills the partial-block reading of the loss, and it will be reported as killed.

If both hold, the two mechanisms divide cleanly: a partial block crossing the wire explains the
forward, the block structure explains where the gradient difference concentrates, and neither
explains the other.

### 2b. The same profile at 24 layers, on the review head (8 x RTX 5060 Ti, 2026-09-11)

The published tables use the 24-layer shared debug model, so the profile is repeated there, on the
review head, with both gates checked before anything was computed. Two identical `dp1` runs on this
box are bit-identical on all 750 gradient tensors, so a `dp1`-against-`pp2` difference is produced
by pipelining and not by the runtime. At step 1 the two agree exactly: both
`12.593917846679688`, zero float32 units in the last place apart, `dp1` reproducing the table's
`12.59392`. The forward is identical; everything here is inside the gradients.

Median relative L2 distance per layer, `||g_pp2 - g_dp1|| / ||g_dp1||`, from the last layer down to
the first. The boundary is at layer 12: stage 0 holds layers 0-11, stage 1 holds 12-23.

| layer | median | layer | median | layer | median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 2.81e-03 | 15 | 5.02e-03 | 7 | 7.76e-03 |
| 22 | 2.90e-03 | 14 | 6.63e-03 | 6 | 9.80e-03 |
| 21 | 3.31e-03 | 13 | 7.02e-03 | 5 | 9.75e-03 |
| 20 | 3.76e-03 | **12** | **1.97e-02** | 4 | 9.84e-03 |
| 19 | 3.53e-03 | 11 | 6.22e-03 | 3 | 8.59e-03 |
| 18 | 4.84e-03 | 10 | 8.15e-03 | 2 | 1.16e-02 |
| 17 | 5.40e-03 | 9 | 7.99e-03 | 1 | 1.29e-02 |
| 16 | 5.85e-03 | 8 | 8.65e-03 | 0 | 2.24e-02 |

- The distance grows smoothly as the backward pass deepens, from `2.8e-03` at layer 23 to
  `1.3e-02` at layer 1, about five-fold over 22 layers.
- Layer 12 breaks that pattern: `1.97e-02`, roughly three times layers 13 (`7.0e-03`) and 11
  (`6.2e-03`), with the profile dropping straight back afterwards. It is the only step in the list.
  It is the first layer after the stage boundary, and the two largest distances in the whole model
  sit inside it: `attention_res_norm` at `3.30e-02` and `attention_res_proj` at `3.25e-02`, the
  modules that read the block stack.
- **This cuts against the partial-sum mechanism rather than for it.** At 24 layers the boundary is
  at layer 12 and `12 % 12 == 0`, so layer 12 opens a block and no partial block crosses there. A
  mechanism locating the difference in a re-materialised partial sum predicts nothing special at
  this boundary, and a step is what the data shows. What a boundary does at a block start is
  re-materialise the stack itself -- the single-GPU path grows it in place, the pipeline path
  rebuilds it from the received columns. That is a candidate, not a conclusion; the six boundary
  cells arbitrate.
- Layer 0 is the largest at `2.24e-02` and discriminates nothing: it is the deepest point of the
  backward pass, so the accumulated difference is expected to be largest there under any mechanism,
  including one with no block stack.
- Two tensors are bit-identical: `lm_head.weight` and `norm.weight`, both on the last stage with
  gradients computed before anything crosses a boundary. Consistent with the mechanism's
  prediction, not proof of it.
- **The localisation, independent of which mechanism explains it.** The two largest distances in
  the entire model are `attention_res_norm` and `attention_res_proj` at the first layer after the
  boundary -- the two modules that read the block stack. The difference is not spread across the
  model; it is concentrated in the modules that consume the thing the pipeline rebuilds.
- The 33-layer profile in section 2 is **not comparable with this one**; it has been re-measured and
  the result is in section 2c, which corrects the boundary reading below.

Written before the 33-layer re-run, so the reading is not chosen afterwards:

- If 33 layers also steps at its first post-boundary layer (17), a boundary produces a step whether
  or not a partial block crosses it, and the mechanism is the stack re-materialisation rather than
  the partial sum.
- If 33 layers is genuinely smooth on this instrument while 24 steps, then something distinguishes
  a block-start boundary from a mid-block one, and the six boundary cells become essential rather
  than confirmatory.


Working material behind the answer above: earlier runs on the PR head at 33 layers, the step-1
gradient profiles, the probe hacks and the commands. None of it is for posting, and the 33-layer
numbers in particular are a different model from the one the answer uses.

The earlier H100 table in this section used the same 1024-token protocol as the answer above but
quoted only steps 1, 10 and 100; it is superseded by the table in the paste block, which adds the
absolute reference and steps 20 and 50.

| row | cell | step 1 loss / grad norm | step 2 | step 3 | step 10 |
| --- | --- | --- | --- | --- | --- |
| a | dp1 (4 x 256, four accumulation groups) | 12.38454 / 23.5000 | 11.09969 / 17.6250 | 8.22154 / 14.8750 | 3.64265 / 4.8125 |
| b | pp2, cache on (4 pipeline micro-batches, 1F1B) | 12.38454 / 23.5000 | 11.08361 / 17.3750 | 8.39830 / 15.3750 | 3.73389 / 4.9688 |
| c | pp2, cache off (one stage per rank, so the delta is the whole stack) | bitwise with b, all ten steps | | | |
| d | dp1, the four groups accumulated in reverse order (noise floor: order only) | 12.38454 / 23.5000 | 11.09241 / 17.6250 | 8.13135 / 13.6875 | 3.52680 / 4.3125 |
| e | dp1 again on a fresh compile cache | bitwise with a, all ten steps | | | |
| f | dp1 with 2 x 256, plain vs reversed | bitwise, all ten steps (12.42445 ... 4.00793) | | | |

Relative to row a: pp2 (b) reads +0.14% / +2.2% / +2.5% at steps 2 / 3 / 10; the order-only row d reads -0.07% / -1.1% / -3.2%.

Row d is the point: dp1 with nothing changed but the order in which the four micro-batch gradients are accumulated (the four groups run in reverse; each group's forward and backward is identical, only the `.grad` accumulation order differs) separates from dp1 by the same class as pp2 does, at steps 2, 3 and 10. With two micro-batches the same reversal is bitwise for ten steps (`a + b == b + a`; row f), so the hack changes order and nothing else. Two dp1 runs on fresh compile caches are bitwise for ten steps (row e).

**Why it is not a bug** — the checks we have, in the order they exclude things:

1. Transport: at one stage per rank, where the delta is the whole stack by construction, cache on vs cache off is bitwise for ten steps (rows b, c) on three separate heads; at two stages per rank they differ as above, and step 1 is identical in every pp x vp cell of the PR's matrix either way, so the routing never changes what the model computes forward.
2. Step 1: loss and total gradient norm agree to within one unit in the last place (`9.5e-07` absolute, `7.7e-08` relative on the loss; `0.125`, one bf16 ulp, on the total gradient norm) for pp2, pp4 and pp8 against dp1 in every results table of the PR -- measured at full precision, not read off the printed five decimals, which round that difference away. The divergence that grows is inside the step-1 gradients, not in what the model computes forward.
3. Step-1 gradient profile on this head (dp1 vs pp2, 1.40 B parameters): two micro-batches (so no accumulation-order term), 1002 tensors: `lm_head`, `norm`, `output_res_norm/proj` and layers 32, 30, 29, 28 bitwise; the first non-identical tensor is layer 31's `attention.wq_b` (2 of 786,432 elements, one bf16 ulp); layer 27's `wq_a` / `q_norm` / `wq_b` / `attention_norm` / `attention_res_*` (1 to 962 elements, ~1.5 ulps); from layer 26 down every tensor differs, 1-3% of elements at ~1.7 ulps, growing to 15% at layer 25, 28% at layer 24 (block 2's first layer), 20-30% through layers 23-17, 35% at layer 16 (the first layer across the stage boundary, no jump), 76% at layer 12 (block 1's first layer), 88% at layer 0. So the difference is born in the MLA layers' query-side gradients ten layers above the boundary and grows smoothly down the backward; the steps are at the block starts (where a stack column closes), not at the stage boundary. The same profile in float32 (13-layer alias, 2026-09-08) put the first non-identical tensors at the last attention layer's `wq_b` / `wq_a` / `q_norm` at 1e-7 with `dO`, `K`, `V` bitwise, and the growth at 3-5x per layer with no boundary step.
4. Precision: with float32 parameters and optimizer states and bf16 compute (torchtitan's default regime, 9-layer alias, 2026-09-08) the pair reads step 1 bitwise / grad norm 1.1e-4, then 2.7e-4, 1.7e-3, 1.5e-2, 7.9e-4 at steps 2-5; float32 does not remove the spread, it only delays it, which is what an order-of-summation difference does and what a wrong tensor would not.
5. Upstream's stock pipeline on upstream's own models, same protocol (measured 2026-09-05): `llama3_debugmodel` pp2 (1F1B) vs dp1 reads 8.02759 / 7.10431 / 4.11391 against 8.02759 / 7.10430 / 4.11384 at steps 1 / 3 / 10 (1.7e-5), `deepseek_v3_debugmodel` 8.15954 / 4.87379 / 3.88192 against 8.15954 / 4.87367 / 3.88273 (2.1e-4); the debug K3 flavor is the sensitive one, as its own no-PP row d shows.

If it helps, I can run rows a-d on H100s under the same protocol so the table is on your hardware.

--- END PASTE ---

## Tables and commands

### Superseded: raw series (10 steps), PR head at 33 layers, 1024 tokens per step as 4 x 256

Measured on the **PR head `75045fed5`** (base `6e2ac3dcd`, 33-layer debug model), before the branch was rebased. The two 100-step sections below are on the rebased review head at main's 24-layer debug model, so their absolute losses are not comparable with these; the branch on today's main is not bitwise with the PR head (upstream #4347 gives the KDA layer the document mask where the branch passed None), which is why the evidence was re-measured there.

- dp1 (chain A, GPU 4, with the step-1 dump): 12.38454/23.5000; 11.09969/17.6250; 8.22154/14.8750; 7.24563/12.2500; 6.93756/7.7188; 5.32022/8.0000; 4.96387/5.3438; 3.99455/3.9062; 4.13019/4.5938; 3.64265/4.8125
- dp1 again, fresh caches (chain C, GPU 7): identical to the line above at every step
- pp2 cache on (rerun on GPUs 4,5 after the first pass was killed at step 6 by a stray SIGTERM; steps 1-6 of the first pass equal the rerun): 12.38454/23.5000; 11.08361/17.3750; 8.39830/15.3750; 6.91475/11.0000; 6.58242/8.5000; 5.02071/6.3125; 4.96762/6.6875; 4.09698/5.8438; 4.41748/5.1562; 3.73389/4.9688
- pp2 cache off (`kimi_k3_debugmodel_ppnaive`, an uncommitted alias whose only change is `functools.partial(pipeline_kimi_k3, attn_res_cache=False)` as the spec's `pipelining_fn`): identical to pp2 cache on at every step
- dp1 with the four accumulation groups reversed (`MB_REVERSE=1`): 12.38454/23.5000; 11.09241/17.6250; 8.13135/13.6875; 7.36877/9.8750; 6.84851/8.1250; 4.83627/6.4062; 4.57047/5.0312; 3.98946/4.0938; 3.93381/4.0312; 3.52680/4.3125

512 tokens per step as 2 x 256: dp1 plain and dp1 reversed both read 12.42445/31.8750; 11.33692/24.3750; 9.22832/16.5000; 8.59245/14.3750; 6.51163/10.4375; 6.63974/8.1875; 4.87923/5.9375; 4.54911/7.7500; 4.38689/6.4062; 4.00793/6.5000.

### The two probe hacks (uncommitted, in the scratch worktree of the PR head)

`torchtitan/trainer.py`, `train_step`: after `microbatch_groups` is built, `if os.environ.get("MB_REVERSE") == "1": microbatch_groups.reverse()` (the groups are the same batches; each group's forward and backward is unchanged; only the order in which their gradients accumulate into `.grad` changes). Before `clip_grad_norm_`, when `GRAD_DUMP` is set and `self.step == 1`, `torch.save({fqn: grad.float().cpu()})` per rank (`full_tensor()` on DTensors). The SM120 KDA capability guard is lifted in `kda.py` for the box, as in every local run.

### Commands

```text
MX=phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh   # seed checkpoint, warm pass, measured pass, one inductor cache per cell
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/attn_gym_up SEED_ROOT=/workspace/.mx3_seeds_ppnum SEED_CFG=kimi_k3_debugmodel MEASURE_STEPS=10 WARM_STEPS=1
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"; PD="--parallelism.spmd_backend partial_dtensor"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_ppnum CFG=kimi_k3_debugmodel BATCH="$B4" CELLS="dp1_4mb|1|$D $PD" $MX ppnum_dp1                # + GRAD_DUMP=... for the step-1 dump
TITAN=/tmp/wt_ppnum CFG=kimi_k3_debugmodel BATCH="$B4" CELLS="pp2on_4mb|2|$D $P $PD" $MX ppnum_pp2on
TITAN=/tmp/wt_ppnum CFG=kimi_k3_debugmodel_ppnaive BATCH="$B4" CELLS="pp2off_4mb|2|$D $P $PD" $MX ppnum_pp2off
MB_REVERSE=1 TITAN=/tmp/wt_ppnum CFG=kimi_k3_debugmodel BATCH="$B4" CELLS="dp1rev_4mb|1|$D $PD" $MX ppnum_dp1rev4
python cmp_profile2.py dp1.rank0.step1.pt pp2on.rank0.step1.pt pp2on.rank1.step1.pt     # /workspace/gd_ppnum
```

Split under pp2 (from the log): stage 0 = `tok_embeddings`, `layers.0` .. `layers.16` (and the vision tower); stage 1 = `layers.17` .. `layers.32`, the final norms, `output_res_*`, `lm_head`. Blocks of 12 layers: block 0 = layers 0-11 (committed on stage 0), block 1 = 12-23 (its start on stage 0, committed on stage 1 at layer 24), block 2 = 24-32 (partial).

### Step-1 gradient profiles

Metrics per tensor: fraction of elements that differ, the median difference among them in bf16 ulps (ulp at x = |x| * 2^-8), and the relative norm difference; per layer: the mean fraction, the median of the per-tensor medians, the median and max normrel.

dp1 vs pp2 (cache on), 2 x 256 tokens (the transport alone):

| group | tensors | bitwise | diff frac | median ulps | normrel median | normrel max |
| --- | --- | --- | --- | --- | --- | --- |
| lm_head, norm, output_res_* | 4 | 4 | 0 | - | 0 | 0 |
| layer 32 (KDA) | 24 | 24 | 0 | - | 0 | 0 |
| layer 31 (MLA) | 24 | 23 | 2 elements | 1.03 | 0 | 4.8e-7 |
| layers 30, 29, 28 | 90 | 90 | 0 | - | 0 | 0 |
| layer 27 (MLA) | 24 | 18 | 0.001 | 1.38 | 0 | 2.5e-4 |
| layer 26 | 30 | 1 | 0.024 | 1.71 | 3.6e-4 | 1.7e-3 |
| layer 25 | 30 | 0 | 0.140 | 1.79 | 1.5e-3 | 2.8e-3 |
| layer 24 (block 2 start) | 30 | 0 | 0.283 | 1.95 | 3.1e-3 | 4.3e-3 |
| layers 23 .. 17 (stage 1) | 24-30 each | 0 | 0.21-0.31 | 1.83-1.91 | 2.0e-3 .. 2.6e-3 | 3.3e-3 .. 4.2e-3 |
| layer 16 (stage 0, first across the boundary) | 30 | 0 | 0.348 | 1.97 | 3.1e-3 | 3.7e-3 |
| layers 15 .. 13 | 24-30 | 0 | 0.44-0.60 | 2.05-2.59 | 3.4e-3 .. 5.8e-3 | 5.5e-3 .. 8.2e-3 |
| layer 12 (block 1 start) | 30 | 0 | 0.762 | 5.15 | 1.9e-2 | 2.4e-2 |
| layers 11 .. 1 | 24-38 | 0 | 0.65-0.83 | 2.5-4.2 | 6.0e-3 .. 1.3e-2 | 7.4e-3 .. 2.5e-2 |
| layer 0 | 29 | 0 | 0.880 | 5.82 | 2.1e-2 | 5.0e-2 |
| tok_embeddings | 1 | 0 | 0.000 (sparse rows) | 4.11 | 1.1e-2 | 1.1e-2 |
| vision encoder (stage 0) | 6 | 0 | 0.818 | 4.92 | 1.4e-2 | 2.0e-2 |

dp1 vs pp2 (cache on), 4 x 256 tokens (the transport plus the micro-batch accumulation): `lm_head` and `norm` bitwise; `output_res_*` 13% of elements at 1.5 ulps; layer 32 31% at 1.5 ulps (normrel 2.8e-3); then 33%, 47%, 54%, 56%, 57%, 63%, 66% for layers 31 .. 25; 75% / 3.4 ulps at layer 24; 66-73% through 23 .. 17; 75% at layer 16 (no jump); 82% / 4.8 ulps at layer 12; 89% / 5.5 ulps at layer 0. pp2 cache on vs cache off at 4 x 256: every tensor bitwise on both ranks (548 + 454 tensors).



## Removed 2026-09-12

Two wide tables stood here -- the H100 controls in their original layout and the 33-layer stress shape on 2 x H100 PCIe -- both with step-50 and step-100 columns past the reference's memorisation. They are removed under the numerics-table rule; steps 1 / 10 / 20 of the same runs are in the paste block.

### The step-1 print, scoped by box and depth

Three measured facts, no mechanism attached:

- 24 layers on this box: all five cells print the same step-1 loss, `12.605700`.
- 33 layers on this box: `pp2`, `pp2 x vp2` and the whole-stack cell print `12.336340` where `dp1` and the reversed-accumulation cell print `12.336350` -- a difference in the last printed digit.
- 33 layers on the other box: every pipeline shape printed the same step-1 loss.

A printed loss agrees only to the precision printed, and that is all these three facts assert. They are not claims about tensors: on the other box the step-1 gradient dump at this depth already showed the deepest attention layer's `wq_b` differing by one bf16 unit in the last place in 2 of 786,432 elements, so the step-1 gradients were not identical even in the runs whose printed losses matched. Measured, rather than left at the printed digits: a step-1 run of each cell at this depth with the loss logged at full precision reads `dp1` `12.336345672607422` and `pp2` `12.336344718933105`. The difference is `9.536743e-07` absolute, `7.7e-08` relative -- exactly one float32 unit in the last place at that magnitude, so the two step-1 losses are adjacent float32 values rather than the same one. The total gradient norm reads `23.25` against `23.125`, one bf16 unit in the last place apart. The printed five decimals were rounding a real difference of one ulp in each, not showing agreement.
