# PR title: [Kimi K3] Pipeline parallelism for the text decoder: the block attention residual crosses stages

PR 4312. Branch `pp_review3` on the fork (`fe34932ee`): the reviewed PR head `087c4d177` squashed onto post-expert-parallel main as `a4d68655c`, the review-round commits replayed on top, the whole line rebased onto upstream/main `6e2ac3dcd` on 2026-09-04 (clean; main touched nothing under the model), then two commits of the evening: the split takes any layer count, and the debug model is 33 layers. The PR branch `k3_pp_text` sits at `0e7cc5ea1` until the sync is approved. The PR branch `k3_pp_text` was synced to it on 2026-09-04 (forced update over `087c4d177`, lease-protected). Paste between the markers into the PR body. Design history, the rejected designs and the per-comment answers are in `phase13_k3like_48b_posttrain/REVIEW_ANSWERS_PP_CP_2026-09-04.md` (logbook); the body carries what the branch does and the evidence.

--- PASTE BEGIN ---

### Summary

Adds pipeline parallelism to the Kimi K3 text decoder. Before this change `parallelize.py` rejects `pipeline_parallel_degree > 1`; core's `pipeline_llm` would split the model at layer boundaries and carry one hidden-state tensor per hop, which cannot express Block Attention Residuals: every later stage needs every earlier block's residual, and the final aggregation (`output_res_proj`, then `output_res_norm`) must run only on the stage that owns `lm_head`.

After it `pipeline_kimi_k3` (in `parallelize.py`) splits the model with this model's names and builds the schedule on `AttnResPipelineStage`, a `torch.distributed.pipelining.PipelineStage` subclass: a hop carries (*hidden*, *delta*), *delta* being the block residuals the receiving rank has not seen yet; each rank keeps the blocks it has seen in one store shared by its virtual stages; the backward returns every block's gradient along the same routes.

Step 1 is bit-identical to a single GPU on every pp x vp cell of the irregular debug model, two to thirty-two stages, with the delta transport and with the whole stack on every hop.

### Design

- The stage protocol, in the subclass (`pipeline_stage.py`)
  - `forward_one_chunk` assembles the full block stack the model expects from the rank's store plus the received *delta*, runs the stage, keeps the blocks the stage committed, and sends on only what the next rank lacks. The model takes and returns the whole stack and knows nothing of the transport; the chunk id comes with the call.
  - `backward_one_chunk` reads the gradient of the assembled stack, an autograd leaf the stage owns: the received columns go back as the *delta*'s gradient, dense and in wire order, and the stored columns are deposited in the rank store.
  - The stage that committed or received a block collects those deposits into its own incoming gradient (`_retrieve_recv_grads`) before its backward, which every schedule orders after the later stages' backward on the rank. No tensor hook, no autograd Function, no detach trick.
  - A micro-batch's blocks are released after the rank's last stage forward for it, so the store holds only the in-flight micro-batches.
  - Metadata inference runs the same assembly (`_compute_outputs`); `_compute_input_grads` returns dense gradients, which is where the P2P buffer finding below is handled.
- The routing tables (`layout.py`)
  - `BlockLayoutTables` simulates one micro-batch's forward in stage order over the split the trainer actually applied and tabulates, per stage, the blocks it commits, the blocks its rank already holds, and the blocks its P2P must carry; sender and receiver compute the same tables, so nothing but the delta travels.
  - The layer-to-stage map is one all-gather over the pipeline group; the stage-to-rank map is the schedule's own `stage_index_to_group_rank`. Uneven stages are allowed; a block boundary inside a stage is a partial block on the wire.
  - Why the delta is bounded: with $P$ ranks a block committed at stage $S$ is fresh on the wire for $P-1$ hops; from $S+P$ on every receiving rank already holds it, because its previous virtual stage was $S-P$. The per-hop payload is bounded by the commits of the last $P-1$ stages, independent of depth.
  - `attn_res_cache=False` (a `functools.partial` on the pipelining function, so every rank resolves it identically) sends the whole stack on every hop (naive); the two transports differ only in the tables, which makes them the A/B in the results. Plain `1F1B`, one stage per rank, is the naive transport by construction.
- The split and the entry (`parallelize.py`, where every model keeps its parallelism entry points): `kimi_k3_module_fqns_per_model_part` is a pure function of the config -- core's layer distribution, `lm_head` where core says `output`, the AttnRes aggregation modules with the head, the vision tower with the embedding.
- The core hook: `pipeline_llm(..., stage_class=...)` (`pipeline_parallel.py`), the one generic change, so a model can run its stages on a `PipelineStage` subclass.
- The model (`model.py`): the first layer of a block joins the stack before its sub-layers attend, so a stage boundary at a block start needs nothing special and the stack a stage receives is exactly the stack the layers read; the head-owning stage alone runs the aggregation.
- What this replaced: the reviewed version carried the same protocol in a 1228-line adapter that wrapped `forward_one_chunk`, `backward_one_chunk` and `step`, kept a thread-local micro-batch id, and bridged the same-rank gradient path with a tensor grad hook and an autograd Function. The subclass implements it once, on the stage's own methods, in 388 lines; the adapter's numerics are reproduced to within one bf16 rounding (table below).

### Results

`kimi_k3_debugmodel` is 33 layers with a block size of 12 and MLA at every fourth layer and the last, so it is irregular the way the 93-layer model is: two blocks of 12 and a partial block of 9 (93 = 7 x 12 + 9), the stack ending on two adjacent MLA layers, and 35 units with the embedding and the head, which no pipeline shape divides. Every split in the table is uneven; the stage count is the multiple of `pipeline_parallel_degree` nearest to units / `layers_per_stage`, and core sees the split rather than the knob (its ceiling would refuse 35 units at 4 per stage).

`--debug.seed 42 --debug.deterministic`, one seed checkpoint per model shape, 4096 tokens per step in micro-batches of 256, 8 pipeline micro-batches, `first/last_stage_less_layers` at their default 1 so the embedding and the head count as units; every cell runs twice and the second run is read. The runner with the seed-load assertion is `phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh` in the logbook.

```
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
P="--parallelism.pipeline_parallel_degree"; L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
cell dp1 1
cell pp2_vp4 2 $P 2 $L 4 $IL;  cell pp4_vp4 4 $P 4 $L 2 $IL;  cell pp8_vp4 8 $P 8 $L 1 $IL
```

Every virtual-pipeline cell twice, with the delta transport and naive (every hop carries the whole stack), and the last six rows with data and expert parallel around the pipeline. The bf16 columns are the branch as it is; the float32 columns carry the total gradient norm in float32 (the `clip_grad_norm_` reduction of pytorch PR 194033 / torchtitan PR 4135, applied to the run tree and not on this branch). The dp2 rows read a different batch (the loader shards the dataset by data-parallel rank), so step 1 is compared within a data-parallel group.

| cell | stages | ranks | layers per stage | transport | step 1 | step 3, step 10 (bf16 norm) | step 3, step 10 (float32 norm) |
|---|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | - | 12.41967 | 7.56783, 3.45908 | 7.57490, 3.34752 |
| pp2 x vp4 | 8 | 2 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 (embedding on the first, head on the last) | delta | 12.41967 | 7.47862, 3.42131 | 7.49055, 3.33238 |
| pp2 x vp4 | 8 | 2 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | naive | 12.41967 | 7.66420, 3.32480 | 7.61479, 3.35875 |
| pp4 x vp4 | 16 | 4 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.41967 | 7.57579, 3.36337 | 7.57446, 3.43256 |
| pp4 x vp4 | 16 | 4 | 2 / 3 / 3 / 2 ... 2 / 1 | naive | 12.41967 | 7.64929, 3.49334 | 7.68891, 3.43122 |
| pp8 x vp4 | 32 | 8 | 1 / 2 / 2 / 1 ... 1 / 0 (a head-only last stage) | delta | 12.41967 | 7.51825, 3.37366 | 7.49769, 3.49425 |
| pp8 x vp4 | 32 | 8 | 1 / 2 / 2 / 1 ... 1 / 0 | naive | 12.41967 | 7.60614, 3.42516 | 7.51799, 3.30288 |
| dp2 | - | 2 | - | - | 12.40417 | 7.37116, 3.30135 | 7.37116, 3.30122 |
| dp2 x ep2 | - | 2 | - | - | 12.40257 | 7.45076, 3.38303 | 7.45076, 3.37020 |
| dp2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | 12.40417 | 7.28299, 3.40680 | 7.29014, 3.42404 |
| dp2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | naive | 12.40417 | 7.32403, 3.36641 | 7.32403, 3.36488 |
| dp2 x ep2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | 12.40257 | 7.49486, 3.24775 | 7.49486, 3.24388 |
| dp2 x ep2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | naive | 12.40257 | 7.39084, 3.34226 | 7.39084, 3.34489 |
| dp2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.40417 | 7.48020, 3.33841 | 7.49642, 3.31773 |
| dp2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | naive | 12.40417 | 7.60047, 3.25333 | 7.61220, 3.25558 |
| dp2 x ep2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.40257 | 7.39910, 3.24169 | 7.40208, 3.31594 |
| dp2 x ep2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | naive | 12.40257 | 7.30184, 3.25535 | 7.30184, 3.26341 |

Step 1 is the number that can be compared, and it is the same number in every cell of a data-parallel group with the pipeline on or off: 12.41967 in the dp1 rows, 12.40417 in dp2 and dp2 x pp2 / pp4, 12.40257 in dp2 x ep2 and its pipeline rows, on all three splits, both transports, and both grad-norm precisions (under the float32 norm the totals agree to 2e-4: dp1 16.1631, pp2 x vp4 16.1661, pp4 x vp4 16.1646, pp8 x vp4 16.1656). Step-1 gradients, all 1,399,095,936 elements: the per-parameter norms of pp2 x vp4 and pp8 x vp4 sit within bf16 rounding of dp1's (median relative difference 2.0e-4 and 2.2e-4, no parameter group off: element-wise every group differs by 1.3 to 1.5 percent alike, the head 0.5 percent), and 0.27 percent of the elements flip sign, pp8 with 32 stages no more than pp2 with 8, 80 percent of them below a hundredth of their tensor's rms; two dp1 runs on fresh compile caches are bitwise. For scale, a gradient that is actually different (dp2, which reads another batch) flips 23 percent of the elements, few of them near zero, and moves the per-parameter norms by percent.

| pair (step 1) | sha1-identical parameters | per-parameter norm, relative difference median / p90 / max | sign flips | implied first-update difference |
|---|---|---|---|---|
| dp1 vs pp2 x vp4 | 2 of 1002 | 2.0e-4 / 1.7e-3 / 1.2e-2 | 0.267% | 10.3% |
| dp1 vs pp8 x vp4 | 2 of 1002 | 2.2e-4 / 1.5e-3 / 1.4e-2 | 0.277% | 10.5% |

In float32 end to end (parameters, activations, the expert GEMM as a per-expert float32 loop), the pipeline's step-1 gradients differ from one GPU's by 1e-2 median, growing from 3e-6 at the last layer to 1e-2 at the first with no step at any stage boundary, on one boundary without a store, on the interleaved schedule with it, and on the naive transport alike; a single GPU whose only change is the rounding of the expert backward (forward and routing untouched) shows the same magnitude and the same layer profile, and two single-GPU runs that differ in nothing arithmetic are bitwise. The pipeline's backward differs from autograd's by the summation order of the assembled block stack, a float32-rounding difference at the top that this network's backward amplifies a hundredfold (logbook, `PP_STEP10_SPREAD_2026-09-04.md` section 11). The later steps spread by a few percent in either direction, and the spread is not a property of the pipeline: Adam's first update is $lr \cdot \mathrm{sign}(g)$ per element, so the elements whose gradient sits below bf16 rounding noise flip sign between any two runs that sum in a different order and each moves by $2 \cdot lr$ the other way (a 0.27 percent flip fraction is a 10 percent change of the first update), and this flavor (bf16 parameters and optimizer states, lr 8e-4 with 2 warm-up steps, the loss falling from 12.4 to 3.3 in ten steps) does not average that out. The same few percent appear with no pipeline in the run: the dp1 cell moves 3.2 percent at step 10 when only the grad-norm precision changes, pure data parallel at 1 / 2 / 4 / 8 spreads 2.5 percent (with the batch composition changing), and expert parallel moves the same-data row by -3.9 to +2.1 percent (dp2 x ep2, dp4 x ep2 / ep4, dp8 x ep2 / ep4 / ep8; the logbook's `PP_STEP10_SPREAD_2026-09-04.md` has the per-step data, the per-group census and the ladder). On the previous head the same step-1 comparison also covered delta against naive on one topology, the subclass against the reviewed adapter, and 32 stages on 2 GPUs, all at a 1e-4 median (logbook, `REVIEW_ANSWERS_PP_CP_2026-09-04.md` 3.3). A hundred steps on the debug flavor's 32-sample data is a memorization curve (every cell at 0.04 to 0.05 by step 100, the crossings of 1.0 spread over 15 steps). On streamed cc12m (no sample repeats, same seed and batch), the four curves track each other: the gap between any pipeline cell and dp1 over steps 51 to 100 (mean absolute 0.017 to 0.049) sits under the curves' own step-to-step movement (0.12) and their spread (sd 0.10), and the 32-stage delta transport is the lowest of the four rather than the slowest.

| cell (streamed cc12m, 100 steps) | step 10 | step 50 | mean of steps 51 to 100 (sd) | step 100 |
|---|---|---|---|---|
| dp1 | 3.672 | 2.653 | 2.540 (0.105) | 2.539 |
| pp2 x vp4 | 3.836 | 2.621 | 2.525 (0.109) | 2.509 |
| pp8 x vp4 | 3.617 | 2.588 | 2.490 (0.110) | 2.504 |
| pp8 x vp4, naive | 3.776 | 2.628 | 2.524 (0.106) | 2.534 |

Memory at this scale does not move: a cached block of the debug model is 256 tokens x 1024 x 2 bytes, so a rank's store is a few MB against activations of GiB. The saving the per-micro-batch release buys is blocks x T x D x 2 bytes no longer resident per micro-batch, which is GB at K3's width and needs a measurement at that shape.

### A `torch.distributed.pipelining` finding

With one layer per stage the last stage holds only the head, whose first op on the block stack is a `cat`, so autograd hands the stage's input gradients back as views; `PipelineStage._backward_metadata_inference` records those strides, `_create_grad_recv_info` allocates the receive buffer with `torch.empty_strided`, and c10d rejects it at the first `RECV_B` with "Tensors for P2P must be non-overlapping and dense".

Every other split passed because a later op consumed the input and autograd accumulated a dense gradient. The subclass returns dense gradients from `_compute_input_grads`; the library-side fix would be a dense `torch.empty` receive buffer and `.contiguous()` before the send.

### Changed files

    torchtitan/distributed/
      pipeline_parallel.py                  +4/-1    pipeline_llm(stage_class=...)
    torchtitan/models/kimi_k3/
      pipeline_stage.py                     +395/-0  AttnResPipelineStage and the rank store (new)
      layout.py                             +233/-0  BlockLayoutTables from the split the trainer applied (new)
      parallelize.py                        +169/-3  the pipelining entry: the split (any layer count), the tables, the transport switch; pipeline parallel off the unsupported list
      model.py                              +39/-22  the block stack in and out of a stage; the block's first layer joins the stack before attending
      __init__.py                           +17/-7   registers the pipelining_fn; the debug model at 33 layers, the 93-layer model's partial block
    tests/unit_tests/cpu/
      test_kimi_k3_pp_fqn_injection.py      +131/-0  the split, including a layer count no shape divides (new)
      test_kimi_k3_pp_layout.py             +122/-0  the tables: uneven split, cache on and off, the local map (new)
      test_kimi_k3_pp_stage.py              +79/-0   assembly, routing, the gradient split, the store (new)
    tests/integration_tests/features.py     +8/-0    the pp2 cell
    torchtitan_recipes/tests/features.py    +32/-0   the pp2 and pp8 x vp4 configurations

### CI/CD Coverage

Three CPU unit tests (the split, the layout tables, the stage's carrier handling) run in the default suite; a pp2 integration cell on two GPUs. The pp8 x vp4 configuration (32 stages, one unit per stage, so the residual crosses every boundary the schedule has) is in the recipes for the 8-GPU run above and is not a CI cell.

### Review round 1

- The one-line comments are applied as asked (comment revert, `first_layer_in_block`, the split function public, the pp8 x vp4 CI cell dropped).
- The even-split precondition is gone: the tables follow whatever split the trainer applied, learned by one all-gather; a 5/7/6/6 split is a unit test.
- The transport switch left the model config for the pipelining entry; the split became a pure function of the config; the stale naive-mode probe was deleted.
- The block's first layer joins the stack before its sub-layers attend ("cat at the start"); the rank store releases a micro-batch's blocks when the rank is done with them, not at step end.
- The 32-layer flavor is replaced by making the one debug model irregular (now 33 layers, the 93-layer model's partial block of 9, 35 units no pipeline shape divides) and the whole pp x vp matrix rerun on it, which is what surfaced the P2P buffer finding.
- The adapter and its wrappers were replaced by the `PipelineStage` subclass above; the reconstruction of how the adapter got there, the rejected designs, and why torch's per-stage `fwd_cache` cannot serve a non-adjacent consumer are in the logbook document linked from the top.

### Review round 2

- `pipeline_llm` keeps only the `stage_class` parameter, no docstring (the file has none) and the pyrefly suppression the hook had dropped is back; the core diff is the seam and nothing else.
- The pipelining entry, the split and the stage lookup moved from a `pipeline.py` into `parallelize.py`, where every model keeps its parallelism entry points; `layout.py` and `pipeline_stage.py` stay as files.
- The full-attention helper's docstring, the debug registry comment and the residual docstring are one line or upstream's own; the helper is the deduction asked for ("(3 KDA + 1 MLA) * k + remainder"), shared by the 30-layer and the 93-layer model.

--- PASTE END ---
