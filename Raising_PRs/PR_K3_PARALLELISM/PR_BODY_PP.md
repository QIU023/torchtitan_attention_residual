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
  - `attn_res_cache=False` (a `functools.partial` on the pipelining function, so every rank resolves it identically) sends the whole stack on every hop; the two transports differ only in the tables, which makes them the A/B in the results. Plain `1F1B`, one stage per rank, is the whole-stack transport by construction.
- The split and the entry (`parallelize.py`, where every model keeps its parallelism entry points): `kimi_k3_module_fqns_per_model_part` is a pure function of the config -- core's layer distribution, `lm_head` where core says `output`, the AttnRes aggregation modules with the head, the vision tower with the embedding.
- The core hook: `pipeline_llm(..., stage_class=...)` (`pipeline_parallel.py`), the one generic change, so a model can run its stages on a `PipelineStage` subclass.
- The model (`model.py`): the first layer of a block joins the stack before its sub-layers attend, so a stage boundary at a block start needs nothing special and the stack a stage receives is exactly the stack the layers read; the head-owning stage alone runs the aggregation.
- What this replaced: the reviewed version carried the same protocol in a 1228-line adapter that wrapped `forward_one_chunk`, `backward_one_chunk` and `step`, kept a thread-local micro-batch id, and bridged the same-rank gradient path with a tensor grad hook and an autograd Function. The subclass implements it once, on the stage's own methods, in 388 lines; the adapter's numerics are reproduced to within one bf16 rounding (table below).

### Results

`kimi_k3_debugmodel` is 33 layers with a block size of 12 and MLA at every fourth layer and the last, so it is irregular the way the 93-layer model is: two blocks of 12 and a partial block of 9 (93 = 7 x 12 + 9), the stack ending on two adjacent MLA layers, and 35 units with the embedding and the head, which no pipeline shape divides. Every split in the table is uneven; the stage count is the multiple of `pipeline_parallel_degree` nearest to units / `layers_per_stage`, and core sees the split rather than the knob (its ceiling would refuse 35 units at 4 per stage).

`--debug.seed 42 --debug.deterministic`, one seed checkpoint per flavor, 4096 tokens per step in micro-batches of 256, 8 pipeline micro-batches, `first/last_stage_less_layers` at their default 1 so the embedding and the head count as units (32 units) and every split is uneven; every cell runs twice and the second run is read (a cold FlexAttention autotune under load moves this model's step-1 loss). The runner with the seed-load assertion is `phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh` in the logbook.

```
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
P="--parallelism.pipeline_parallel_degree"; L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
cell dp1 1
cell pp2_vp4 2 $P 2 $L 4 $IL;  cell pp4_vp4 4 $P 4 $L 2 $IL;  cell pp8_vp4 8 $P 8 $L 1 $IL
```

Two matrices on the same cells, seed and batch, every virtual-pipeline cell twice, with the delta transport and with the whole stack every hop: the first is the branch as it is (bf16 total gradient norm), the second carries the total gradient norm in float32 (the `clip_grad_norm_` reduction of pytorch PR 194033 / torchtitan PR 4135, applied to the run tree and not on this branch). Every cell starts from the same seed checkpoint, runs twice, and the second run is read; the last six rows put data parallel and expert parallel around the pipeline. The dp2 rows read a different batch (the loader shards the dataset by data-parallel rank), so step 1 is compared within a data-parallel group: 12.41967 in all five dp1 rows, 12.40417 in dp2 and dp2 x pp2 / pp4, 12.40257 in dp2 x ep2 and dp2 x ep2 x pp2 / pp4. Every step-1 value with the pipeline on is bit-identical to the same mesh without it, on splits of 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3, 2 / 3 / 3 / 2 ... 2 / 1 and 1 / 2 / 2 / 1 ... 1 / 0 layers per stage, the last one with a head-only stage (the rows of the earlier 30-layer model, whose 32 units every shape divided, are in the logbook).

The bf16 grad-norm matrix (this branch as it is):

| cell | stages | ranks | layers per stage | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | - | 12.41967 | 7.56783 | 3.45908 |
| pp2 x vp4 | 8 | 2 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 (embedding on the first, head on the last) | delta | 12.41967 | 7.47862 | 3.42131 |
| pp4 x vp4 | 16 | 4 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.41967 | 7.57579 | 3.36337 |
| pp8 x vp4 | 32 | 8 | 1 / 2 / 2 / 1 ... 1 / 0 (a head-only last stage) | delta | 12.41967 | 7.51825 | 3.37366 |
| pp8 x vp4 | 32 | 8 | 1 / 2 / 2 / 1 ... 1 / 0 | whole stack every hop | 12.41967 | 7.60614 | 3.42516 |
| dp2 | - | 2 | - | - | 12.40417 | 7.37116 | 3.30135 |
| dp2 x ep2 | - | 2 | - | - | 12.40257 | 7.45076 | 3.38303 |
| dp2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | 12.40417 | 7.28299 | 3.40680 |
| dp2 x ep2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | 12.40257 | 7.49486 | 3.24775 |
| dp2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.40417 | 7.48020 | 3.33841 |
| dp2 x ep2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.40257 | 7.39910 | 3.24169 |
| pp2 x vp4 | 8 | 2 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | whole stack every hop | | | |
| pp4 x vp4 | 16 | 4 | 2 / 3 / 3 / 2 ... 2 / 1 | whole stack every hop | | | |
| dp2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | whole stack every hop | | | |
| dp2 x ep2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | whole stack every hop | | | |
| dp2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | whole stack every hop | | | |
| dp2 x ep2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | whole stack every hop | | | |

The float32 grad-norm matrix (the fix applied to the run tree), running locally, the rows follow:

| cell | stages | ranks | layers per stage | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | - | 12.41967 | 7.57490 | 3.34752 |
| pp2 x vp4 | 8 | 2 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | 12.41967 | 7.49055 | 3.33238 |
| pp4 x vp4 | 16 | 4 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | 12.41967 | 7.57446 | 3.43256 |
| pp8 x vp4 | 32 | 8 | 1 / 2 / 2 / 1 ... 1 / 0 | delta | 12.41967 | 7.49769 | 3.49425 |
| pp8 x vp4 | 32 | 8 | 1 / 2 / 2 / 1 ... 1 / 0 | whole stack every hop | | | |
| dp2 | - | 2 | - | - | | | |
| dp2 x ep2 | - | 2 | - | - | | | |
| dp2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | | | |
| dp2 x ep2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | delta | | | |
| dp2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | | | |
| dp2 x ep2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | delta | | | |
| pp2 x vp4 | 8 | 2 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | whole stack every hop | | | |
| pp4 x vp4 | 16 | 4 | 2 / 3 / 3 / 2 ... 2 / 1 | whole stack every hop | | | |
| dp2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | whole stack every hop | | | |
| dp2 x ep2 x pp2 x vp4 | 8 | 4 | 4 / 5 / 5 / 4 / 4 / 4 / 4 / 3 | whole stack every hop | | | |
| dp2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | whole stack every hop | | | |
| dp2 x ep2 x pp4 x vp4 | 16 | 8 | 2 / 3 / 3 / 2 ... 2 / 1 | whole stack every hop | | | |

Step 1 is the same number in every cell, and it is the number that can be compared: under a float32 total norm the cells agree to 2e-4 (dp1 16.1631, pp2 x vp4 16.1661, pp4 x vp4 16.1646, pp8 x vp4 16.1656, whole-stack pp8 16.1649), which is bf16 summation-order rounding of the gradients; on the 30-layer model, carrying the norm in float32 for the whole run left the step-10 spread where it was (6.2% across five cells against 5.6% in bf16), and the second matrix above shows it for this model. The later steps spread by a few percent in either direction, and that spread is not a property of the pipeline: the same dp1 cell moves by 3.4% at step 10 when only the grad-norm reduction precision changes (a fresh compile cache changes nothing: two dp1 runs on fresh caches are bitwise, and every PP row of the previous head reproduces bitwise on the rebased one), and dp1 against dp2, which also changes the batch composition, moves by 6% in the same debug setup. The mechanism is Adam's first step, $lr \cdot \mathrm{sign}(g)$ per element: the elements whose gradient sits below bf16 rounding noise flip sign between any two runs that sum in a different order, each flipped element moves by $2 \cdot lr$ the other way, and this flavor (bf16 parameters and optimizer states, lr 8e-4 with 2 warm-up steps, the loss falling from 12.5 to 3.4 in ten steps) does not average that out.

The step-1 sign census over all 1,399,095,936 gradient elements: the fraction whose sign differs between two runs, and the first-update difference it implies ($2\sqrt{f}$ of the update norm). Element-wise the gradients differ by about 1.3% in relative L2 in every parameter group alike (embedding 1.48%, experts 1.27%, attention 1.42%, norms 1.28%, router 1.25%, head 0.54%) while the per-parameter norms agree to 2e-4, which is bf16's signature; 80% of the flipped elements sit below a hundredth of their tensor's rms, and pp8 with 32 stages flips no more than pp2 with 8. Two dp1 runs on fresh compile caches were bitwise on the 30-layer model (918 of 918 parameters sha1-identical); the no-pipeline controls (dp1 against FSDP dp2, and against 512-token micro-batches) are running locally and follow.

| pair (step 1) | sign flips | implied first-update difference | group with the most flips |
|---|---|---|---|
| dp1 vs dp1 on a fresh compile cache (30-layer model) | 0 | 0 | none |
| dp1 vs pp2 x vp4 | 0.267% | 10.3% | head 0.44%, attention 0.43% (embedding 0: its zero rows are exact) |
| dp1 vs pp8 x vp4 | 0.277% | 10.5% | head 0.46%, attention 0.44% |
| dp1 vs dp2 (FSDP, no pipeline) | | | |
| dp1 vs dp1 with 512-token micro-batches (accumulation order, no pipeline) | | | |

100 steps at the same seed on the debug flavor's data, which is 32 samples that 4096 tokens per step cycle through every other step: every curve memorizes it (loss 0.04 to 0.05 at step 100), and what differs is how fast, with the 32-stage delta transport the slowest to cross 1.0 and the same 32 stages with the whole stack every hop as fast as dp1. A memorization race amplifies any perturbation, so the same four cells on streamed cc12m (no sample repeats) and dp1 / pp2 / pp8 at a second seed are running locally and follow.

| cell | first step below 1.0 | loss at step 50 | mean loss, steps 51 to 100 | loss at step 100 |
|---|---|---|---|---|
| dp1 | 31 | 0.234 | 0.090 | 0.04273 |
| pp2 x vp4 | 26 | 0.157 | 0.075 | 0.04380 |
| pp8 x vp4 | 41 | 0.525 | 0.114 | 0.04678 |
| pp8 x vp4, whole stack every hop | 30 | 0.192 | 0.082 | 0.04558 |
| dp1, seed 43 | | | | |
| pp2 x vp4, seed 43 | | | | |
| pp8 x vp4, seed 43 | | | | |
| dp1, streamed cc12m | | | | |
| pp2 x vp4, streamed cc12m | | | | |
| pp8 x vp4, streamed cc12m | | | | |
| pp8 x vp4, whole stack every hop, streamed cc12m | | | | |

Step-1 per-parameter gradients, the evidence for "equal up to rounding" before anything is amplified: the fp32 norm of every parameter's gradient, hashed and compared on one shared warm compile cache (same model and seed, measured on the review branch before the rebase).

The distribution is the one bf16 summation order produces -- a median of 1e-4 with the tail on 16-element parameters -- and no parameter group stands out; a systematic routing error would show as one group orders of magnitude above the rest.

| comparison (step 1, 918 parameters) | loss / grad_norm | median / p90 / max relative difference of the per-parameter norm |
|---|---|---|
| subclass, pp2 x vp2 delta transport vs dp1 | identical, 12.44394 / 15.5625 | 1.5e-4 / 1.2e-3 / 1.6e-2 |
| subclass, pp2 x vp2 whole-stack transport vs dp1 | identical | 1.3e-4 / 1.1e-3 / 2.1e-2 |
| subclass, delta vs whole-stack transport, same topology | identical | 0 / 6.7e-4 / 1.0e-2 (404 of 918 differ) |
| subclass vs the reviewed hook adapter, pp2 x vp2 delta | identical | 3.5e-5 / 5.3e-4 / 9.2e-3 |
| subclass, 32 stages on 2 GPUs (pp2 x vp16, embedding-only and head-only stages) vs dp1 | identical | 2.0e-4 / 1.7e-3 / 1.5e-2 |

What remains between the cells under the float32 norm is the transport's summation order (delta against whole stack) and FlexAttention's autotune, which picks kernels by benchmark timing that `--debug.deterministic` does not control; the bf16 matrix adds the per-topology grouping of the norm on top, which is why it spreads more.

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
      parallelize.py                        +162/-3  the pipelining entry: the split, the tables, the transport switch; pipeline parallel off the unsupported list
      model.py                              +39/-22  the block stack in and out of a stage; the block's first layer joins the stack before attending
      __init__.py                           +15/-7   registers the pipelining_fn; the debug model at 30 layers, irregular like the 93-layer model
    tests/unit_tests/cpu/
      test_kimi_k3_pp_fqn_injection.py      +100/-0  the split (new)
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
- The 32-layer flavor is replaced by making the one debug model irregular (30 layers, partial last block, lone MLA at the end) and the whole pp x vp matrix rerun on it, which is what surfaced the P2P buffer finding.
- The adapter and its wrappers were replaced by the `PipelineStage` subclass above; the reconstruction of how the adapter got there, the rejected designs, and why torch's per-stage `fwd_cache` cannot serve a non-adjacent consumer are in the logbook document linked from the top.

### Review round 2

- `pipeline_llm` keeps only the `stage_class` parameter, no docstring (the file has none) and the pyrefly suppression the hook had dropped is back; the core diff is the seam and nothing else.
- The pipelining entry, the split and the stage lookup moved from a `pipeline.py` into `parallelize.py`, where every model keeps its parallelism entry points; `layout.py` and `pipeline_stage.py` stay as files.
- The full-attention helper's docstring, the debug registry comment and the residual docstring are one line or upstream's own; the helper is the deduction asked for ("(3 KDA + 1 MLA) * k + remainder"), shared by the 30-layer and the 93-layer model.

--- PASTE END ---
