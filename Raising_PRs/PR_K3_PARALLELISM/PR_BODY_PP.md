# PR title: [Kimi K3] Pipeline parallelism for the text decoder: the block attention residual crosses stages

PR 4312. Branch `pp_review3` on the fork (`d6b1ffe47`): the reviewed PR head `087c4d177` squashed onto upstream/main `9b5f60c40` (post expert-parallel merge) as `a4d68655c`, then the nine review-round commits replayed on top. The PR branch `k3_pp_text` is synced to it only on the user's approval. Paste between the markers into the PR body. Design history, the rejected designs and the per-comment answers are in `phase13_k3like_48b_posttrain/REVIEW_ANSWERS_PP_CP_2026-09-04.md` (logbook); the body carries what the branch does and the evidence.

--- PASTE BEGIN ---

### Summary

Adds pipeline parallelism to the Kimi K3 text decoder. Before this change `parallelize.py` rejects `pipeline_parallel_degree > 1`; core's `pipeline_llm` would split the model at layer boundaries and carry one hidden-state tensor per hop, which cannot express Block Attention Residuals: a block residual is defined over the whole layer stack, so every later stage needs every earlier block's residual, and the final aggregation (`output_res_proj`, then `output_res_norm`) must run only on the stage that owns `lm_head`. After it `pipeline_kimi_k3` splits the model with this model's names and builds the schedule on `AttnResPipelineStage`, a `torch.distributed.pipelining.PipelineStage` subclass: a hop carries (*hidden*, *delta*), *delta* being the block residuals the receiving rank has not seen yet; each rank keeps the blocks it has seen in one store shared by its virtual stages; the backward returns every block's gradient along the same routes. Step 1 is bit-identical to a single GPU on every pp x vp cell of the irregular debug model, two to thirty-two stages, with the delta transport and with the whole stack on every hop.

### Design

- The stage protocol, in the subclass (`pipeline_stage.py`)
  - `forward_one_chunk` assembles the full block stack the model expects from the rank's store plus the received *delta*, runs the stage, keeps the blocks the stage committed, and sends on only what the next rank lacks. The model takes and returns the whole stack and knows nothing of the transport; the chunk id comes with the call.
  - `backward_one_chunk` reads the gradient of the assembled stack, which is an autograd leaf the stage owns: the received columns go back as the *delta*'s gradient, dense and in wire order; the stored columns are deposited in the rank store, and the stage that committed or received a block collects those deposits into its own incoming gradient (`_retrieve_recv_grads`) before its backward, which every schedule orders after the later stages' backward on the rank. No tensor hook, no autograd Function, no detach trick.
  - A micro-batch's blocks are released after the rank's last stage forward for it, so the store holds only the in-flight micro-batches.
  - Metadata inference runs the same assembly (`_compute_outputs`); `_compute_input_grads` returns dense gradients, which is where the P2P buffer finding below is handled.
- The routing tables (`layout.py`)
  - `BlockLayoutTables` simulates one micro-batch's forward in stage order over the split the trainer actually applied -- every rank learns the layer-to-stage map with one all-gather over the pipeline group, and the stage-to-rank map is the schedule's own `stage_index_to_group_rank` -- and tabulates, per stage, the blocks it commits, the blocks its rank already holds, and the blocks its P2P must carry. Sender and receiver compute the same tables, so nothing but the delta travels. Uneven stages are allowed; a block boundary inside a stage is a partial block on the wire.
  - Why the delta is bounded: with $P$ ranks a block committed at stage $S$ is fresh on the wire for $P-1$ hops; from $S+P$ on every receiving rank already holds it, because its previous virtual stage was $S-P$. The per-hop payload is bounded by the commits of the last $P-1$ stages, independent of depth.
  - `attn_res_cache=False` (a `functools.partial` on the pipelining function, so every rank resolves it identically) sends the whole stack on every hop; the two transports differ only in the tables, which makes them the A/B in the results. Plain `1F1B`, one stage per rank, is the whole-stack transport by construction.
- The split (`pipeline.py`): `kimi_k3_module_fqns_per_model_part` is a pure function of the config -- core's layer distribution, `lm_head` where core says `output`, the AttnRes aggregation modules with the head, the vision tower with the embedding.
- The core hook: `pipeline_llm(..., stage_class=...)` (`pipeline_parallel.py`), the one generic change, so a model can run its stages on a `PipelineStage` subclass.
- The model (`model.py`): the first layer of a block joins the stack before its sub-layers attend, so a stage boundary at a block start needs nothing special and the stack a stage receives is exactly the stack the layers read; the head-owning stage alone runs the aggregation.
- What this replaced: the reviewed version carried the same protocol in a 1228-line adapter that wrapped `forward_one_chunk`, `backward_one_chunk` and `step`, kept a thread-local micro-batch id, and bridged the same-rank gradient path with a tensor grad hook and an autograd Function. The subclass implements it once, on the stage's own methods, in 388 lines; the adapter's numerics are reproduced to within one bf16 rounding (table below).

### Results

`kimi_k3_debugmodel` is 30 layers with a block size of 12 and MLA at every fourth layer and the last, so it is irregular the way the 93-layer model is: the last block is partial and the stack ends on a lone MLA layer. `--debug.seed 42 --debug.deterministic`, one seed checkpoint per flavor, 4096 tokens per step in micro-batches of 256, 8 pipeline micro-batches, `first/last_stage_less_layers` at their default 1 so the embedding and the head count as units (32 units) and every split is uneven; every cell runs twice and the second run is read, because a cold FlexAttention autotune under load moves this model's step-1 loss. The runner with the seed-load assertion is `phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh` in the logbook.

```
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
P="--parallelism.pipeline_parallel_degree"; L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
cell dp1 1
cell pp2_vp4 2 $P 2 $L 4 $IL;  cell pp4_vp4 4 $P 4 $L 2 $IL;  cell pp8_vp4 8 $P 8 $L 1 $IL
```

Training loss on this branch (rebased onto main after the expert-parallel merge), 32 units (30 layers plus the embedding and the head) over the pipeline: <!-- TBD main30: fill from /workspace/mx3_main30_pp* -->

| cell | stages | ranks | units per stage | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | - | | | |
| pp2 x vp4 | 8 | 2 | 4 | delta | | | |
| pp4 x vp4 | 16 | 4 | 2 | delta | | | |
| pp8 x vp4 | 32 | 8 | 1 (embedding-only and head-only stages) | delta | | | |
| pp8 x vp4 | 32 | 8 | 1 | whole stack every hop | | | |

Step-1 per-parameter gradients, the evidence for "equal up to rounding" before anything is amplified: fp32 norm of every parameter's gradient, hashed and compared, on one shared warm compile cache (the same model and seed, measured on the review branch before the rebase). The distribution is the one bf16 summation order produces -- a median of 1e-4 with the tail on 16-element parameters whose norm is 1e-4 -- and no parameter group stands out; a systematic error in the gradient routing would show as a group orders of magnitude above the rest.

| comparison (step 1, 918 parameters) | loss / grad_norm | median / p90 / max relative difference of the per-parameter norm |
|---|---|---|
| subclass, pp2 x vp2 delta transport vs dp1 | identical, 12.44394 / 15.5625 | 1.5e-4 / 1.2e-3 / 1.6e-2 |
| subclass, pp2 x vp2 whole-stack transport vs dp1 | identical | 1.3e-4 / 1.1e-3 / 2.1e-2 |
| subclass, delta vs whole-stack transport, same topology | identical | 0 / 6.7e-4 / 1.0e-2 (404 of 918 differ) |
| subclass vs the reviewed hook adapter, pp2 x vp2 delta | identical | 3.5e-5 / 5.3e-4 / 9.2e-3 |
| subclass, 32 stages on 2 GPUs (pp2 x vp16, embedding-only and head-only stages) vs dp1 | identical | 2.0e-4 / 1.7e-3 / 1.5e-2 |

The later steps spread in both directions for two reasons that are not this PR: `torch.nn.utils.get_total_norm` reduces the per-tensor norms in the gradients' dtype and PP groups the parameters differently per topology (pytorch PR 194033 carries the reduction in fp32; with that patch the whole-stack cells collapse pairwise), and FlexAttention's autotune picks kernels by benchmark timing, which `--debug.deterministic` does not control.

Memory at this scale does not move: a cached block of the debug model is 256 tokens x 1024 x 2 bytes, so a rank's store is a few MB against activations of GiB. The saving the per-micro-batch release buys is blocks x T x D x 2 bytes no longer resident per micro-batch, which is GB at K3's width and needs a measurement at that shape.

### A `torch.distributed.pipelining` finding

With one layer per stage the last stage holds only the head, whose first op on the block stack is a `cat`, so autograd hands the stage's input gradients back as views; `PipelineStage._backward_metadata_inference` records those strides and `_create_grad_recv_info` allocates the receive buffer with `torch.empty_strided`, which c10d rejects at the first `RECV_B` with "Tensors for P2P must be non-overlapping and dense". Every other split passed because a later op in the stage consumed the input and autograd accumulated a dense gradient. The subclass returns dense gradients from `_compute_input_grads`; the library-side fix would be a dense `torch.empty` receive buffer, and `.contiguous()` before the send.

### Changed files

    torchtitan/distributed/
      pipeline_parallel.py                  +10/-2   pipeline_llm(stage_class=...)
    torchtitan/models/kimi_k3/
      pipeline_stage.py                     +388/-0  AttnResPipelineStage and the rank store (new)
      pipeline.py                           +166/-0  the pipelining_fn: the split, the tables, the transport switch (new)
      layout.py                             +241/-0  BlockLayoutTables from the split the trainer applied (new)
      model.py                              +49/-21  the block stack in and out of a stage; the block's first layer joins the stack before attending
      __init__.py                           +27/-6   registers the pipelining_fn; the debug model at 30 layers, irregular like the 93-layer model
      parallelize.py                        +2/-3    pipeline parallel off the unsupported list
    tests/unit_tests/cpu/
      test_kimi_k3_pp_fqn_injection.py      +95/-0   the split (new)
      test_kimi_k3_pp_layout.py             +122/-0  the tables: uneven split, cache on and off, the local map (new)
      test_kimi_k3_pp_stage.py              +78/-0   assembly, routing, the gradient split, the store (new)
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

--- PASTE END ---
