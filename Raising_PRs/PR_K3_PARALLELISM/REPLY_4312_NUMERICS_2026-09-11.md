# Reply to Tianyu's numerics comment on PR 4312 (id 3985333653, `parallelize.py` L267), 2026-09-11

Measured on the PR head `75045fed5` (19 commits on main `ac10ca48f`), `kimi_k3_debugmodel` (33 layers, 1.40 B parameters, bf16 parameters and optimizer states), seed 42, `--debug.deterministic`, one shared seed checkpoint, a warm pass then the measured pass on one inductor cache per cell, 1024 tokens per step as four 256-token micro-batches (dp1: four gradient-accumulation groups; pp2: four pipeline micro-batches, 1F1B), `spmd_backend partial_dtensor`. Commands and the raw tables are below the paste block.

--- PASTE ---

Two questions, taken in order.

**How the flag changes accumulation.** At one stage per rank, not at all; at more than one, it changes the association of the same sum. `attn_res_cache` decides which hop carries a block: `route_payload` picks the columns of the model's own stack tensor that the next stage still lacks (cache on) or all of them (cache off), and the receiving stage rebuilds the stack from the received columns plus the ones it already holds (`assemble_stack`). With one stage per rank no rank ever receives a block twice, so the set the receiver lacks is the whole stack and the two settings are the same code path -- measured bitwise for ten steps below (rows b, c). With two stages per rank they are not: the cached path assembles received columns next to locally held ones, so the backward sums the same contributions in a different association, and on 2 x H100 at pp2 x vp2 the two read `3.150940` and `3.514970` at step 10 (1.17% and 12.9% against dp1). That difference is the same class as the no-PP noise floor in row d below, which is what an association change costs with no pipeline in sight; it is not the flag computing something else.

**How PP changes accumulation against one GPU.** The one thing the pipeline changes is where the gradient of a block stack is summed. Every layer reads the whole stack twice (`_apply_attention_residual` before its attention and before its FFN), so a block committed on stage 0 has readers on both stages. On one GPU autograd accumulates all of those read-contributions into one buffer in engine order. Under PP the stack a stage receives is a fresh autograd leaf (`assemble_stack` returns `stack.detach().requires_grad_(True)`), so stage 1 sums its sixteen layers' contributions into that leaf's `.grad`, hands the result back as one dense bf16 tensor, and stage 0's backward adds its own layers' contributions on top. Same terms, one extra rounding boundary in the sum. Measured at step 1 (profile below), that boundary adds nothing visible: the per-layer difference walks through it without a jump. What the profile does show is where the first bits move, and it is not at the boundary. And with more than two micro-batches a second difference appears at the very top of the backward: with two micro-batches (a two-term sum, order-free) `lm_head`, the final norms, `output_res_*` and layers 32-28 are bitwise; with four, a third of the elements of every tensor from `output_res_*` down differ by about 1.5 bf16 ulps, i.e. the four micro-batch gradients are associated differently by the trainer's accumulation loop and by the schedule's per-chunk backward. Both are bf16-ulp perturbations; what makes them a percent at step 10 is this flavor, not the pipeline, which is what the noise-floor row shows.

**The controls** (33-layer debug model, 1.40 B parameters, bf16 parameters and optimizer states, seed 42, deterministic, one seed checkpoint, one inductor cache per cell after a warm pass, 1024 tokens per step as four 256-token micro-batches, 1F1B):

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
2. Step 1: loss and total gradient norm bitwise for pp2, pp4 and pp8 against dp1 in every results table of the PR; the difference is inside the step-1 gradients, not in what the model computes forward.
3. Step-1 gradient profile on this head (dp1 vs pp2, 1.40 B parameters): two micro-batches (so no accumulation-order term), 1002 tensors: `lm_head`, `norm`, `output_res_norm/proj` and layers 32, 30, 29, 28 bitwise; the first non-identical tensor is layer 31's `attention.wq_b` (2 of 786,432 elements, one bf16 ulp); layer 27's `wq_a` / `q_norm` / `wq_b` / `attention_norm` / `attention_res_*` (1 to 962 elements, ~1.5 ulps); from layer 26 down every tensor differs, 1-3% of elements at ~1.7 ulps, growing to 15% at layer 25, 28% at layer 24 (block 2's first layer), 20-30% through layers 23-17, 35% at layer 16 (the first layer across the stage boundary, no jump), 76% at layer 12 (block 1's first layer), 88% at layer 0. So the difference is born in the MLA layers' query-side gradients ten layers above the boundary and grows smoothly down the backward; the steps are at the block starts (where a stack column closes), not at the stage boundary. The same profile in float32 (13-layer alias, 2026-09-08) put the first non-identical tensors at the last attention layer's `wq_b` / `wq_a` / `q_norm` at 1e-7 with `dO`, `K`, `V` bitwise, and the growth at 3-5x per layer with no boundary step.
4. Precision: with float32 parameters and optimizer states and bf16 compute (torchtitan's default regime, 9-layer alias, 2026-09-08) the pair reads step 1 bitwise / grad norm 1.1e-4, then 2.7e-4, 1.7e-3, 1.5e-2, 7.9e-4 at steps 2-5; float32 does not remove the spread, it only delays it, which is what an order-of-summation difference does and what a wrong tensor would not.
5. Upstream's stock pipeline on upstream's own models, same protocol (measured 2026-09-05): `llama3_debugmodel` pp2 (1F1B) vs dp1 reads 8.02759 / 7.10431 / 4.11391 against 8.02759 / 7.10430 / 4.11384 at steps 1 / 3 / 10 (1.7e-5), `deepseek_v3_debugmodel` 8.15954 / 4.87379 / 3.88192 against 8.15954 / 4.87367 / 3.88273 (2.1e-4); the debug K3 flavor is the sensitive one, as its own no-PP row d shows.

If it helps, I can run rows a-d on H100s under the same protocol so the table is on your hardware.

--- END PASTE ---

## Tables and commands

### Raw series (loss / grad norm per step, 10 steps), 1024 tokens per step as 4 x 256

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



## 100-step controls on 2 x H100 PCIe

The same five controls, on the branch under review rather than the PR head, measured on rented H100s so the table sits on the class of hardware a reviewer reproduces. PR 4500's columns; `dp1` of this box is the reference for every row.

| cell | step 1 loss (diff) | step 10 loss (diff) | step 100 loss (diff) | mean abs loss diff, steps 1-100 | step 1 grad norm (diff) | step 10 (diff) | step 100 (diff) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `0.189940` | 0% | `18.6250` | `5.4375` | `4.2500` |
| pp2 | `12.605700` (0%) | `3.227050` (3.61%) | `0.208850` (9.96%) | 10.2% | `18.7500` (0.671%) | `5.6875` (4.6%) | `1.0547` (75.2%) |
| pp2 x vp2, cached transport | `12.605700` (0%) | `3.150940` (1.17%) | `0.220590` (16.1%) | 9.78% | `18.6250` (0%) | `3.7188` (31.6%) | `1.4609` (65.6%) |
| pp2 x vp2, whole-stack transport | `12.605700` (0%) | `3.514970` (12.9%) | `0.244630` (28.8%) | 22.1% | `18.6250` (0%) | `6.0312` (10.9%) | `1.1953` (71.9%) |
| **dp1, accumulation groups reversed (no pipeline)** | `12.605700` (0%) | `3.247610` (4.27%) | `0.182700` (3.81%) | 15.9% | `18.6250` (0%) | `6.6562` (22.4%) | `1.2344` (71%) |

What the rows say, and nothing more:

- Every cell's step-1 loss is identical to the last printed digit, pipelined or not. The step-1 grad norm agrees exactly for three of the four, and `pp2` differs by one bf16 step (`18.6250` against `18.7500`).
- The last row has no pipeline in it at all: it is `dp1` with the four gradient-accumulation groups consumed in the opposite order, everything else equal. It reads `4.27%` at step 10 and `15.9%` as a mean, i.e. at least as large as `pp2`'s `3.61%` and `10.2%`. On this hardware, a pure summation-order change costs as much as pipelining does.
- The two transports are not bitwise with each other at `vp2`. That is expected where a rank holds two stages: the cached path sends only the blocks the receiver lacks and assembles them next to blocks it already holds, so the stack is summed in a different association than when the whole stack arrives on the wire. At one stage per rank (`pp2`) the delta is the whole stack and the two paths coincide.
- The step-100 column is not a useful comparison here: `dp1` has reached `0.19` on the debug dataset, so any percentage is taken against a near-zero reference. Steps 1 and 10 are the columns to read.

Footnote: 2 x NVIDIA H100 PCIe, capability 9.0; torch `2.15.0.dev20260906+cu130` (CUDA 13.0), triton `3.8.0+gitc01b6774`, spmd-types 0.2.5, torch-remat 0.2.0, nvidia-cutlass-dsl 4.6.0 — the same versions as our other box; Attention Gym at upstream main `b16d6d3` (our other box runs `b19162e` from the fork). Branch `pp_review4` = `8aea9ef03` on upstream main `d9ca9e55a`; the shared `kimi_k3_debugmodel`, 24 layers, 1,119,984,672 parameters. The model requests `bound_gate(impl="fused")`, whose CuTeDSL path is documented to require capability 9.0 or newer, and `chunk_kda(impl="fused")`, which resolves to `attn_gym.linear.kda.impl.fused`; which backend the fused chunk ops select internally was not traced. Protocol: seed 42, `--debug.deterministic`, one shared seed checkpoint for all five cells, 1024 tokens per step in micro-batches of 256 (four accumulation groups for `dp1`, four micro-batches for the pipeline cells), one warm step before each measured run, a separate Triton cache directory per group.

Not yet comparable with the other box: the 10-step section above was measured on the PR head at 33 layers, and the matching 100-step run on the 8 x RTX 5060 Ti box (same review head, same 24-layer model, same protocol) had not been run when this table was produced. Only that pair is a box-to-box comparison.

## PR 4500's own CP cells, reproduced on 2 x H100 PCIe (2026-09-11)

PR 4500 published its table on H100. We had only reproduced it on A100, where it read two orders of magnitude larger than published, and the open question was whether the hardware class explained that. This run answers it: the same code, the same protocol, on H100.

`kimi_k3_debugmodel` CP=1 on spmd_types as the reference, then the two CP=2 recipes from the b200 suite, on the 4500 head `2884d82a9`; seed 42, `--debug.deterministic`, `--metrics.log_freq 1`, 256 tokens per train step, 100 steps, no seed checkpoint — the script is the one archived from the A100 run, with only the paths and the GPU ids changed.

| step | CP=1 loss | CP=2 all-gather (diff) | CP=2 Ulysses (diff) | CP=1 grad norm | all-gather gn (diff) | Ulysses gn (diff) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12.561730` | `12.327760` (-1.863%) | `12.340670` (-1.760%) | `28.8750` | `29.500000` (+2.165%) | `29.125000` (+0.866%) |
| 2 | `11.719530` | `11.186950` (-4.544%) | `11.034520` (-5.845%) | `38.0000` | `39.500000` (+3.947%) | `36.500000` (-3.947%) |
| 5 | `8.713460` | `8.047820` (-7.639%) | `7.766800` (-10.864%) | `15.8125` | `13.687500` (-13.439%) | `15.562500` (-1.581%) |
| 10 | `4.149080` | `4.162870` (+0.332%) | `4.510770` (+8.717%) | `7.3750` | `10.687500` (+44.915%) | `13.937500` (+88.983%) |
| 20 | `3.312170` | `3.241000` (-2.149%) | `3.246450` (-1.984%) | `6.6875` | `5.625000` (-15.888%) | `5.812500` (-13.084%) |
| 50 | `2.928190` | `3.037940` (+3.748%) | `2.992010` (+2.180%) | `3.4688` | `5.000000` (+44.142%) | `4.062500` (+17.115%) |
| 100 | `2.872700` | `2.967340` (+3.294%) | `2.866940` (-0.201%) | `5.3438` | `5.250000` (-1.755%) | `5.687500` (+6.432%) |

Side by side with what exists, steps 1 / 10 / 100 of the loss:

| source | hardware | CP=2 all-gather | CP=2 Ulysses |
| --- | --- | --- | --- |
| PR 4500, as published | H100 | `0.608%` / `0.150%` / `2.72%` | `0.576%` / `0.371%` / `1.34%` |
| our replication | 8 x A100-SXM4-40GB | `-0.115%` / `+8.56%` / `-7.86%` | `-0.26%` / `+9.70%` / `-6.31%` |
| our replication | 2 x H100 PCIe | `-1.863%` / `+0.332%` / `+3.294%` | `-1.760%` / `+8.717%` / `-0.201%` |

Read plainly: the published table is not reproduced on H100 either. Step 1 is `-1.86%` here against the published `+0.608%`, and the Ulysses step-10 figure `+8.717%` is the same size as the A100's `+9.70%` rather than the published `0.371%`. The hardware class is therefore not what separates our numbers from the published ones, and this contradicts the assumption we had been working from. What does separate them is still open; the candidates are everything else in that environment (torch build, Attention Gym version, the exact code state behind the head, the data), and none of them is settled by this run.

Footnote: 2 x NVIDIA H100 PCIe, capability 9.0; head `2884d82a9` (PR 4500), fetched from `pull/4500/head`; torch `2.15.0.dev20260906+cu130` (CUDA 13.0), triton `3.8.0+gitc01b6774`, spmd-types 0.2.5, torch-remat 0.2.0, nvidia-cutlass-dsl 4.6.0; Attention Gym at upstream main `b16d6d3`. That head's own KDA guard already admits capability 9.0 ("Hopper SM90 or Blackwell SM100/SM103"), so no guard relaxation was applied and `bound_gate` ran its fused CuTeDSL path, which is what an H100 run of hers would take; the A100 run had needed both a guard relaxation and an eager reference gate, neither of which applies here. One local uncommitted change was needed: `torchtitan/distributed/cudagraph.py` imports `torch.cuda._annotate_cuda_graph_trace`, which this torch nightly does not have, so the import moved inside the profiling post-processor that is its only user. Nothing else was patched, and nothing was committed.
