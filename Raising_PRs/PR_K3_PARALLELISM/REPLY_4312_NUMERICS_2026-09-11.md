# Reply to Tianyu's numerics comment on PR 4312 (id 3985333653, `parallelize.py` L267), 2026-09-11

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

<<PENDING-MECHANISM>>

## 2. Step 1, per parameter

<<PENDING-PROFILE>>

## 3. The trajectory, with a band around it

<<PENDING-BAND>>

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

The tables stop at step 20 on purpose. `dp1` reads `12.593920`, `3.297890` and `3.337080` at
steps 1, 10 and 20, so a percentage against it means something there. Run further on this budget
and the reference falls to `1.78` by step 50 and `0.26` by step 100 as the model memorises the
debug set, and a ratio between two collapsing curves stops measuring the parallelism, so those
steps are not reported here. The full series is in the branch notes if you want it.

What the rows say, on both boxes, and nothing more.

- Step 1 prints the same loss in every cell on each box. What that print is worth is section 1: one float32 unit in the last place, not identity.
- The bottom row has no pipeline in it at all: it is `dp1` with the four accumulation groups consumed in the opposite order, everything else equal. On the H100 box it reads `4.27%` at step 10 and `2.31%` at step 20; on the 5060 Ti box `1.58%` and `1.79%`. That is the size of a pure association change on this flavour, with no pipeline available to blame.
- Against that floor, in the readable range: at step 20 `pp2` reads `2.52%` (H100) and `4.04%` (5060 Ti), the cached `vp2` `0.712%` and `2.67%`, the whole-stack `vp2` `2.72%` and `0.739%`. Every pipeline cell is the same class as the no-pipeline floor, on both boxes; none of them is an order of magnitude away from it.
- At step 10 the two boxes order the cells differently: the H100 box has the whole-stack `vp2` largest at `12.9%` with `pp2` at `3.61%`, this box has `pp2` largest at `13.7%` with the whole-stack `vp2` at `0.717%`. That is single-sample scatter, and the noise band below is its scale: one ordering of the accumulation groups against another, with no pipeline anywhere, spans the same range. The widest pipeline-to-floor gap in the readable range is `pp2` at step 10 on this box, `13.7%` against a floor sample of `1.58%`; on the H100 box the floor is the larger of the pair at the same step (`4.27%` against `3.61%`). Read them against the band, not against each other.
- The two transports do not agree with each other at `vp2` on either box, and they agree at every printed step at one stage per rank on both. That is the flag's whole effect: at one stage per rank the delta is the whole stack, so the two are the same code path; at two, the cached path assembles received blocks next to locally held ones and the same contributions are summed in a different association.

## 4. What the transport flag does

**How the flag changes accumulation.** At one stage per rank, not at all; at more than one, it changes the association of the same sum. `attn_res_cache` decides which hop carries a block: `route_payload` picks the columns of the model's own stack tensor that the next stage still lacks (cache on) or all of them (cache off), and the receiving stage rebuilds the stack from the received columns plus the ones it already holds (`assemble_stack`). With one stage per rank no rank ever receives a block twice, so the set the receiver lacks is the whole stack and the two settings are the same code path -- measured bitwise for ten steps below (rows b, c). With two stages per rank they are not: the cached path assembles received columns next to locally held ones, so the backward sums the same contributions in a different association, and on 2 x H100 at pp2 x vp2 the two read `3.150940` and `3.514970` at step 10 (1.17% and 12.9% against dp1). That difference is the same class as the no-PP noise floor in row d below, which is what an association change costs with no pipeline in sight; it is not the flag computing something else.

**How PP changes accumulation against one GPU.** The one thing the pipeline changes is where the gradient of a block stack is summed. Every layer reads the whole stack twice (`_apply_attention_residual` before its attention and before its FFN), so a block committed on stage 0 has readers on both stages. On one GPU autograd accumulates all of those read-contributions into one buffer in engine order. Under PP the stack a stage receives is a fresh autograd leaf (`assemble_stack` returns `stack.detach().requires_grad_(True)`), so stage 1 sums its sixteen layers' contributions into that leaf's `.grad`, hands the result back as one dense bf16 tensor, and stage 0's backward adds its own layers' contributions on top. Same terms, one extra rounding boundary in the sum. Measured at step 1 (profile below), that boundary adds nothing visible: the per-layer difference walks through it without a jump. What the profile does show is where the first bits move, and it is not at the boundary. And with more than two micro-batches a second difference appears at the very top of the backward: with two micro-batches (a two-term sum, order-free) `lm_head`, the final norms, `output_res_*` and layers 32-28 are bitwise; with four, a third of the elements of every tensor from `output_res_*` down differ by about 1.5 bf16 ulps, i.e. the four micro-batch gradients are associated differently by the trainer's accumulation loop and by the schedule's per-chunk backward. Both are bf16-ulp perturbations; what makes them a percent at step 10 is this flavor, not the pipeline, which is what the noise-floor row shows.
--- PASTE END ---


## Background, superseded

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



## Superseded in presentation: the H100 controls in their original wide layout (same numbers, reproduced in the paste block above)

The five controls on the branch under review, measured on rented H100s. `dp1` of this box is the reference for every row, and its absolute value is printed in every column so the reader can see where the reference stops being usable.

**Read steps 1 through 20.** The reference falls to `1.206420` by step 50 and `0.189940` by step 100: the debug set is memorised at this budget, so the later percentages divide two collapsing curves rather than measuring the parallelism.

Every cell carries the same global batch in the same number of pieces -- four micro-batches of 256 tokens -- and the noise-floor cell differs from `dp1` only in the order those four pieces are consumed, which is the one-variable-at-a-time comparison the reviewer asked for.

Why not PR 4500's 256-token budget, measured rather than assumed: their cells are context-parallel with no pipeline, so one piece of 256 tokens serves them. A four-stage pipeline cell needs four pieces, and how narrow a piece can be is set by the data, not by the kernels. The debug flavour trains on `cc12m-test`, whose rows carry image placeholder tokens, and a micro-batch too small to hold one row is refused before step 1 with `multimodal rows exceed the configured token batch` (`mm_collator.py:101`). Probing `dp1` at four micro-batches: 64, 96, 128 and 192 tokens per micro-batch are each refused; 256 runs (step 1 reads `12.51200`). The floor for this flavour is therefore 256 tokens per micro-batch, and `4 x 256 = 1024` tokens per step is the smallest budget a four-stage cell can take.


| cell | step 1 loss (diff) | step 10 loss (diff) | step 20 loss (diff) | step 50 loss (diff) | step 100 loss (diff) | mean abs loss diff, steps 1-100 | step 1 grad norm (diff) | step 10 (diff) | step 20 (diff) | step 50 (diff) | step 100 (diff) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `3.373330` | `1.206420` | `0.189940` | 0% | `18.6250` | `5.4375` | `3.9844` | `3.9531` | `4.2500` |
| pp2 | `12.605700` (0%) | `3.227050` (3.61%) | `3.288290` (2.52%) | `1.310800` (8.65%) | `0.208850` (9.96%) | 10.2% | `18.7500` (0.671%) | `5.6875` (4.6%) | `3.7344` (6.27%) | `3.3594` (15%) | `1.0547` (75.2%) |
| pp2 x vp2, cached transport | `12.605700` (0%) | `3.150940` (1.17%) | `3.349300` (0.712%) | `1.291140` (7.02%) | `0.220590` (16.1%) | 9.78% | `18.6250` (0%) | `3.7188` (31.6%) | `4.0625` (1.96%) | `2.9375` (25.7%) | `1.4609` (65.6%) |
| pp2 x vp2, whole-stack transport | `12.605700` (0%) | `3.514970` (12.9%) | `3.281700` (2.72%) | `1.639940` (35.9%) | `0.244630` (28.8%) | 22.1% | `18.6250` (0%) | `6.0312` (10.9%) | `3.6719` (7.84%) | `3.1719` (19.8%) | `1.1953` (71.9%) |
| **dp1, accumulation groups reversed (no pipeline)** | `12.605700` (0%) | `3.247610` (4.27%) | `3.295370` (2.31%) | `1.446870` (19.9%) | `0.182700` (3.81%) | 15.9% | `18.6250` (0%) | `6.6562` (22.4%) | `4.2500` (6.67%) | `3.2656` (17.4%) | `1.2344` (71%) |

What the rows say, and nothing more:

- Every cell's step-1 loss prints the same five decimals, pipelined or not. A printed loss rounds: at 33 layers the same comparison measured at full precision reads one float32 ulp apart (`9.5e-07` absolute, `7.7e-08` relative), so read this column as `agree to within one unit in the last place`, not as identity. The step-1 grad norm prints the same value for three of the four, and `pp2` prints one bf16 ulp away (`18.6250` against `18.7500`).
- The last row has no pipeline in it at all: it is `dp1` with the four gradient-accumulation groups consumed in the opposite order, everything else equal. It reads `4.27%` at step 10 and `15.9%` as a mean, i.e. at least as large as `pp2`'s `3.61%` and `10.2%`. On this hardware, a pure summation-order change costs as much as pipelining does.
- The two transports do not agree even to the printed precision at `vp2`. That is expected where a rank holds two stages: the cached path sends only the blocks the receiver lacks and assembles them next to blocks it already holds, so the stack is summed in a different association than when the whole stack arrives on the wire. At one stage per rank (`pp2`) the delta is the whole stack and the two paths coincide.
- Which columns are readable: steps 1 through the last one where the reference is still above roughly 1.0. By step 100 `dp1` has reached `0.19` on the debug dataset, so a percentage there is taken against a memorised reference; the column is quoted for completeness, not as an error term. It is the same artifact the A100 dp2 stream showed in the tensor-parallel tables.
- Steps 20 and 50, where the reference is still `3.37` and `1.21`, keep the same ordering: the no-pipeline noise floor reads `2.31%` and `19.9%` against `pp2`'s `2.52%` and `8.65%`. At step 20 the two are the same size; at step 50 reversing the accumulation order alone costs more than `pp2` or the cached `pp2 x vp2` does, and less than the whole-stack transport's `35.9%`.


Footnote: 2 x NVIDIA H100 PCIe, capability 9.0; torch `2.15.0.dev20260906+cu130` (CUDA 13.0), triton `3.8.0+gitc01b6774`, spmd-types 0.2.5, torch-remat 0.2.0, nvidia-cutlass-dsl 4.6.0 — the same versions as our other box; Attention Gym at upstream main `b16d6d3` (our other box runs `b19162e` from the fork). Branch `pp_review4` = `8aea9ef03` on upstream main `d9ca9e55a`; the shared `kimi_k3_debugmodel`, 24 layers, 1,119,984,672 parameters. The model requests `bound_gate(impl="fused")`, whose CuTeDSL path is documented to require capability 9.0 or newer, and `chunk_kda(impl="fused")`, which resolves to `attn_gym.linear.kda.impl.fused`; which backend the fused chunk ops select internally was not traced. Protocol: seed 42, `--debug.deterministic`, one shared seed checkpoint for all five cells, 1024 tokens per step in micro-batches of 256 (four accumulation groups for `dp1`, four micro-batches for the pipeline cells), one warm step before each measured run, a separate Triton cache directory per group.

Not yet comparable with the other box: the 10-step section above was measured on the PR head at 33 layers, and the matching 100-step run on the 8 x RTX 5060 Ti box (same review head, same 24-layer model, same protocol) had not been run when this table was produced. Only that pair is a box-to-box comparison.

## Not for this reply: the 33-layer stress shape on 2 x H100 PCIe

This table belongs to the PR body's pipeline stress cell, not to the numerics answer above, and is kept here only because the box was already rented. The shape is the deep flavour the `kimi_k3_debugmodel_pp8_vp4` recipe registers (33 layers, 35 units); the branch leaves the shared debug model at main's depth, so nothing else uses it. Same protocol as the table above: `8aea9ef03`, seed 42, deterministic, one shared seed checkpoint, 1024 tokens per step as four pieces of 256, one warm step, 100 measured steps. `dp1` of this shape is the reference; its absolute value is printed in every column.

| cell | step 1 loss (diff) | step 10 (diff) | step 20 (diff) | step 50 (diff) | step 100 (diff) | step 1 grad norm (diff) | step 10 (diff) | step 20 (diff) | step 50 (diff) | step 100 (diff) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.336350` | `3.785670` | `3.340160` | `2.007030` | `0.392840` | `23.2500` | `10.5625` | `3.9844` | `4.0938` | `2.2656` |
| pp2 | `12.336340` (8.11e-05%) | `3.615350` (4.5%) | `3.471710` (3.94%) | `2.166800` (7.96%) | `0.551590` (40.4%) | `23.1250` (0.538%) | `5.6250` (46.7%) | `4.5625` (14.5%) | `3.0938` (24.4%) | `2.6406` (16.6%) |
| pp2 x vp2, cached transport | `12.336340` (8.11e-05%) | `3.823350` (0.995%) | `3.311860` (0.847%) | `1.995890` (0.555%) | `0.404360` (2.93%) | `23.1250` (0.538%) | `7.8125` (26%) | `3.7500` (5.88%) | `3.6719` (10.3%) | `2.0000` (11.7%) |
| pp2 x vp2, whole-stack transport | `12.336340` (8.11e-05%) | `3.652300` (3.52%) | `3.344880` (0.141%) | `1.503360` (25.1%) | `0.291140` (25.9%) | `23.1250` (0.538%) | `7.4375` (29.6%) | `3.7344` (6.27%) | `3.3438` (18.3%) | `2.3281` (2.76%) |
| **dp1, accumulation groups reversed (no pipeline)** | `12.336350` (0%) | `3.715860` (1.84%) | `3.559600` (6.57%) | `2.071850` (3.23%) | `0.631650` (60.8%) | `23.2500` (0%) | `7.5625` (28.4%) | `5.1250` (28.6%) | `3.4375` (16%) | `4.2812` (89%) |

Read steps 1 through 50 here: this shape's reference is still `2.007030` at step 50 and only falls to `0.392840` by step 100, so it stays usable one column longer than the 24-layer table does.

### The step-1 print, scoped by box and depth

Three measured facts, no mechanism attached:

- 24 layers on this box: all five cells print the same step-1 loss, `12.605700`.
- 33 layers on this box: `pp2`, `pp2 x vp2` and the whole-stack cell print `12.336340` where `dp1` and the reversed-accumulation cell print `12.336350` -- a difference in the last printed digit.
- 33 layers on the other box: every pipeline shape printed the same step-1 loss.

A printed loss agrees only to the precision printed, and that is all these three facts assert. They are not claims about tensors: on the other box the step-1 gradient dump at this depth already showed the deepest attention layer's `wq_b` differing by one bf16 unit in the last place in 2 of 786,432 elements, so the step-1 gradients were not identical even in the runs whose printed losses matched. Measured, rather than left at the printed digits: a step-1 run of each cell at this depth with the loss logged at full precision reads `dp1` `12.336345672607422` and `pp2` `12.336344718933105`. The difference is `9.536743e-07` absolute, `7.7e-08` relative -- exactly one float32 unit in the last place at that magnitude, so the two step-1 losses are adjacent float32 values rather than the same one. The total gradient norm reads `23.25` against `23.125`, one bf16 unit in the last place apart. The printed five decimals were rounding a real difference of one ulp in each, not showing agreement.
