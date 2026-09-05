# The numerics behind PR 4312's matrix (2026-09-05)

The PR body carries the matrix, the step-1 census table and one paragraph of conclusion, and points here for the reading. The raw per-step data, the ladder and the per-group census are in `PP_STEP10_SPREAD_2026-09-04.md`.

## What the evidence says

1. Step-1 gradients differ from a single GPU only by float32 rounding, and the difference is in the summation order of the assembled block stack -- not at any stage boundary.
2. Adam's first update amplifies that rounding: it is $lr \cdot \mathrm{sign}(g)$ per element, so elements below rounding noise flip sign between any two runs that sum differently, and this flavor does not average it out in ten steps.
3. The few-percent spread at step 10 is that reordering amplified by this model, not a pipeline defect: upstream's own pipelines on upstream's own models, which reorder nothing, track dp1 to 1e-5 (dense) and 3e-4 (MoE) at step 10, while K3 runs with no pipeline but another rounding-level change spread by the same few percent.
4. Over a hundred steps on data with no repeats, the pipeline cells track a single GPU inside the curves' own movement.

## 1. The float32 end-to-end probe

In float32 end to end (parameters, activations, the expert GEMM as a per-expert float32 loop), the pipeline's step-1 gradients differ from one GPU's by 6e-3 to 1e-2 median, growing from 3e-6 at the last layer to 1e-2 at the first with no step at any stage boundary, on one boundary without a store, on the interleaved schedule with it, and on the naive transport alike. A single GPU whose only change is the rounding of the expert backward (forward and routing untouched) shows the same shape of profile a factor above it, and two single-GPU runs that differ in nothing arithmetic are bitwise.

The pipeline's backward differs from autograd's by the summation order of the assembled block stack, a float32-rounding difference at the top that this network's backward amplifies a hundredfold (logbook, `PP_STEP10_SPREAD_2026-09-04.md` section 11).

## 2. The mechanism, and the controls with no pipeline in them

The later steps spread by a few percent in either direction, and the spread is not a property of the pipeline. Adam's first update is $lr \cdot \mathrm{sign}(g)$ per element, so the elements whose gradient sits below bf16 rounding noise flip sign between any two runs that sum in a different order and each moves by $2 \cdot lr$ the other way (a 0.27 percent flip fraction is a 10 percent change of the first update); this flavor (bf16 parameters and optimizer states, lr 8e-4 with 2 warm-up steps, the loss falling from 12.4 to 3.3 in ten steps) does not average that out.

The same few percent appear with no pipeline in the run: the dp1 cell moves 3.2 percent at step 10 when only the grad-norm precision changes, pure data parallel at 1 / 2 / 4 / 8 spreads 2.5 percent (with the batch composition changing), and expert parallel moves the same-data row by -3.9 to +2.1 percent (dp2 x ep2, dp4 x ep2 / ep4, dp8 x ep2 / ep4 / ep8; the logbook's `PP_STEP10_SPREAD_2026-09-04.md` has the per-step data, the per-group census and the ladder).

On the previous head the same step-1 comparison also covered delta against naive on one topology and the subclass against the reviewed adapter, both at a 1e-4 median, and the pp x vp matrix from 2 to 32 stages read step 1 bit-identical to dp1 in every cell (logbook, `REVIEW_ANSWERS_PP_CP_2026-09-04.md` 3.3).

## 3. A hundred steps

A hundred steps on the debug flavor's 32-sample data is a memorization curve (every cell at 0.04 to 0.05 by step 100, the crossings of 1.0 spread over 15 steps; at a second seed the 32-stage delta cell is again the last to cross, 38 against 31, next to the 0 to 5 steps one configuration moves between the seeds).

On streamed cc12m (no sample repeats, same seed and batch; step 1 reads 12.35295 on dp1 and 12.35294 on the pipeline cells, the summation order of sixteen float32 micro-batch losses, with the routing and the head's gradient bitwise the same), the four curves track each other: the gap between any pipeline cell and dp1 over steps 51 to 100 (mean absolute 0.017 to 0.049) sits under the curves' own step-to-step movement (0.12) and their spread (sd 0.10), and the 32-stage delta transport is the lowest of the four rather than the slowest.

| cell (streamed cc12m, 100 steps) | step 10 | step 50 | mean of steps 51 to 100 (sd) | step 100 |
|---|---|---|---|---|
| dp1 | 3.672 | 2.653 | 2.540 (0.105) | 2.539 |
| pp2 x vp4 | 3.836 | 2.621 | 2.525 (0.109) | 2.509 |
| pp8 x vp4 | 3.617 | 2.588 | 2.490 (0.110) | 2.504 |
| pp8 x vp4, naive | 3.776 | 2.628 | 2.524 (0.106) | 2.534 |

## 4. The control: upstream's own pipelines on upstream's own models

`llama3_debugmodel` and `deepseek_v3_debugmodel` (6 layers each) on pure upstream/main `390e2985b`, `pipeline_llm` and the stock `PipelineStage`, no Kimi K3 code in the run; the same protocol as the K3 tables (one seed checkpoint per flavor, 10 steps, `--debug.deterministic`, 4096 tokens per step in 256-token micro-batches, `partial_dtensor`, CUDA graphs off in every cell). `ctl_pp_matrix.sh` and `ctl_pp8_matrix.sh`.

| cell | stages | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| llama3 dp1 | - | 8.02759 | 7.10430 | 4.11384 |
| llama3 pp2 (Interleaved1F1B, the flavor's default) | 4 | 8.02759 | 7.10430 | 4.11385 |
| llama3 pp2 x vp4 | 8 | 8.02759 | 7.10431 | 4.11399 |
| llama3 pp4 x vp2 | 8 | 8.02759 | 7.10431 | 4.11391 |
| deepseek_v3 dp1 | - | 8.15954 | 4.87367 | 3.88273 |
| deepseek_v3 pp2 (Interleaved1F1B) | 4 | 8.15954 | 4.87379 | 3.88192 |
| deepseek_v3 pp2 x vp4 | 8 | 8.15954 | 4.87379 | 3.88172 |
| deepseek_v3 pp4 x vp2 | 8 | 8.15954 | 4.87379 | 3.88171 |
| deepseek_v3 pp8 (1F1B, one unit per stage) | 8 | 8.15954 | 4.87379 | 3.88189 |

Reading. A plain pipeline changes no arithmetic: every layer's forward and backward is the same computation wherever it runs, and the block gradients accumulate in autograd's order on every rank, so the runs track dp1 to 1e-5 (dense) and 3e-4 relative (MoE; the pipeline's 8 micro-batches of 512 tokens against dp1's 16 of 256 regroup the router's batches). Carrying the block stack across stages does change an order: under the delta transport a block's gradient contributions are summed per rank before the hop, under the naive transport per hop, where autograd sums them once on one graph; that is the float32-rounding-level difference the exact test measures at the top of the network (3e-6 at the last layer), and this network amplifies it a hundredfold through its backward, the router turns it into other gradients, and Adam's first step into the few-percent step-10 spread. The same is noted publicly of cross-stage caching in general (the accumulation order follows the pipeline configuration, so loss and norm cannot be made identical across it); the K3 runs with no pipeline but another rounding-level change (the grad-norm precision, expert parallel) spread by the same few percent because the amplification is the model's, not the pipeline's. The pure-dp ladder is not a clean control for this (the loader shards documents by rank, so dp2 reads another batch); the tables above are.

