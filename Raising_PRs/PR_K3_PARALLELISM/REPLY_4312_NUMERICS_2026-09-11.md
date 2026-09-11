# Reply to Tianyu's numerics comment on PR 4312 (id 3985333653, `parallelize.py` L267), 2026-09-11

Measured on the PR head `75045fed5` (19 commits on main `ac10ca48f`), `kimi_k3_debugmodel` (33 layers, 1.40 B parameters, bf16 parameters and optimizer states), seed 42, `--debug.deterministic`, one shared seed checkpoint, a warm pass then the measured pass on one inductor cache per cell, 1024 tokens per step as four 256-token micro-batches (dp1: four gradient-accumulation groups; pp2: four pipeline micro-batches, 1F1B), `spmd_backend partial_dtensor`. Commands and the raw tables are below the paste block.

--- PASTE ---

Two questions, taken in order.

**How the flag changes accumulation.** It does not. `attn_res_cache` only decides which hop carries a block: `route_payload` picks the columns of the model's own stack tensor that the next stage still lacks (cache on) or all of them (cache off), and the receiving stage rebuilds the stack from the received columns plus the ones it already holds (`assemble_stack`). The tensors on the wire are the same views; no reduction happens in the transport. In the backward the two transports differ only in where a block's gradient columns are handed back (`split_stack_grad`: the received columns return over the wire, the stored ones are deposited on the rank and added when that rank's earlier stage collects them, `RankStore.deposit` / `_collect_into`); with one stage per rank there are no deposits at all, so pp2 cache on and cache off run the identical arithmetic. Measured: the two transports are bitwise for ten steps (table below, rows b and c).

**How PP changes accumulation against one GPU.** The one thing the pipeline changes is where the gradient of a block stack is summed. Every layer reads the whole stack twice (`_apply_attention_residual` before its attention and before its FFN), so a block committed on stage 0 has readers on both stages. On one GPU autograd accumulates all of those read-contributions into one buffer in engine order. Under PP the stack a stage receives is a fresh autograd leaf (`assemble_stack` returns `stack.detach().requires_grad_(True)`), so stage 1 sums its sixteen layers' contributions into that leaf's `.grad`, hands the result back as one dense bf16 tensor, and stage 0's backward adds its own layers' contributions on top. Same terms, one extra rounding boundary in the sum. Measured at step 1 (profile below), that boundary adds nothing visible: the per-layer difference walks through it without a jump. What the profile does show is where the first bits move, and it is not at the boundary. And with more than two micro-batches a second difference appears at the very top of the backward: with two micro-batches (a two-term sum, order-free) `lm_head`, the final norms, `output_res_*` and layers 32-28 are bitwise; with four, a third of the elements of every tensor from `output_res_*` down differ by about 1.5 bf16 ulps, i.e. the four micro-batch gradients are associated differently by the trainer's accumulation loop and by the schedule's per-chunk backward. Both are bf16-ulp perturbations; what makes them a percent at step 10 is this flavor, not the pipeline, which is what the noise-floor row shows.

**The controls** (33-layer debug model, 1.40 B parameters, bf16 parameters and optimizer states, seed 42, deterministic, one seed checkpoint, one inductor cache per cell after a warm pass, 1024 tokens per step as four 256-token micro-batches, 1F1B):

| row | cell | step 1 loss / grad norm | step 2 | step 3 | step 10 |
| --- | --- | --- | --- | --- | --- |
| a | dp1 (4 x 256, four accumulation groups) | 12.38454 / 23.5000 | 11.09969 / 17.6250 | 8.22154 / 14.8750 | 3.64265 / 4.8125 |
| b | pp2, cache on (4 pipeline micro-batches, 1F1B) | 12.38454 / 23.5000 | 11.08361 / 17.3750 | 8.39830 / 15.3750 | 3.73389 / 4.9688 |
| c | pp2, cache off | bitwise with b, all ten steps | | | |
| d | dp1, the four groups accumulated in reverse order (noise floor: order only) | 12.38454 / 23.5000 | 11.09241 / 17.6250 | 8.13135 / 13.6875 | 3.52680 / 4.3125 |
| e | dp1 again on a fresh compile cache | bitwise with a, all ten steps | | | |
| f | dp1 with 2 x 256, plain vs reversed | bitwise, all ten steps (12.42445 ... 4.00793) | | | |

Relative to row a: pp2 (b) reads +0.14% / +2.2% / +2.5% at steps 2 / 3 / 10; the order-only row d reads -0.07% / -1.1% / -3.2%.

Row d is the point: dp1 with nothing changed but the order in which the four micro-batch gradients are accumulated (the four groups run in reverse; each group's forward and backward is identical, only the `.grad` accumulation order differs) separates from dp1 by the same class as pp2 does, at steps 2, 3 and 10. With two micro-batches the same reversal is bitwise for ten steps (`a + b == b + a`; row f), so the hack changes order and nothing else. Two dp1 runs on fresh compile caches are bitwise for ten steps (row e).

**Why it is not a bug** — the checks we have, in the order they exclude things:

1. Transport: cache on vs cache off bitwise for ten steps (rows b, c), and the same at step 1 in every pp x vp cell of the PR's matrix.
2. Step 1: loss and total gradient norm bitwise for pp2, pp4 and pp8 against dp1 in every results table of the PR; the difference is inside the step-1 gradients, not in what the model computes forward.
3. Step-1 gradient profile on this head (dp1 vs pp2, 1.40 B parameters): two micro-batches (so no accumulation-order term), 1002 tensors: `lm_head`, `norm`, `output_res_norm/proj` and layers 32, 30, 29, 28 bitwise; the first non-identical tensor is layer 31's `attention.wq_b` (2 of 786,432 elements, one bf16 ulp); layer 27's `wq_a` / `q_norm` / `wq_b` / `attention_norm` / `attention_res_*` (1 to 962 elements, ~1.5 ulps); from layer 26 down every tensor differs, 1-3% of elements at ~1.7 ulps, growing to 15% at layer 25, 28% at layer 24 (block 2's first layer), 20-30% through layers 23-17, 35% at layer 16 (the first layer across the stage boundary, no jump), 76% at layer 12 (block 1's first layer), 88% at layer 0. So the difference is born in the MLA layers' query-side gradients ten layers above the boundary and grows smoothly down the backward; the steps are at the block starts (where a stack column closes), not at the stage boundary. The same profile in float32 (13-layer alias, 2026-09-08) put the first non-identical tensors at the last attention layer's `wq_b` / `wq_a` / `q_norm` at 1e-7 with `dO`, `K`, `V` bitwise, and the growth at 3-5x per layer with no boundary step.
4. Precision: with float32 parameters and optimizer states and bf16 compute (torchtitan's default regime, 9-layer alias, 2026-09-08) the pair reads step 1 bitwise / grad norm 1.1e-4, then 2.7e-4, 1.7e-3, 1.5e-2, 7.9e-4 at steps 2-5; float32 does not remove the spread, it only delays it, which is what an order-of-summation difference does and what a wrong tensor would not.
5. Upstream's stock pipeline on upstream's own models, same protocol (measured 2026-09-05): `llama3_debugmodel` pp2 (1F1B) vs dp1 reads 8.02759 / 7.10431 / 4.11391 against 8.02759 / 7.10430 / 4.11384 at steps 1 / 3 / 10 (1.7e-5), `deepseek_v3_debugmodel` 8.15954 / 4.87379 / 3.88192 against 8.15954 / 4.87367 / 3.88273 (2.1e-4); the debug K3 flavor is the sensitive one, as its own no-PP row d shows.

If it helps, I can run rows a-d on H100s under the same protocol so the table is on your hardware.

--- END PASTE ---

## Tables and commands

### Raw series (loss / grad norm per step, 10 steps), 1024 tokens per step as 4 x 256

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


