# PR title: [Kimi K3] Pipeline parallelism for the text decoder: the block attention residual crosses stages

Results updated 2026-09-12: 4 x H100 PCIe on `pp_review4` = `dbc425403`; the dp2 stream is running and marked pending. The numerics answer to Tianyu is `REPLY_4312_NUMERICS_SHORT_2026-09-12.md`; the long form, corrected the same day, is `REPLY_4312_NUMERICS_2026-09-11.md`.

PR 4312. PR branch `k3_pp_text` = `dbc425403` since 2026-09-12 (fast-forward from `dd1c0b925`: the B200 cells, the per-rank stage count removed, the comment and docstring trims; GitHub: 31 commits, 16 files, +1324/-51, still `dirty` -- `torchtitan/config/configs.py` conflicts with main `56a721b64`, 17 commits past the base). Before that it was `dd1c0b925` (moved with lease from `75045fed5`, which was 19 commits on upstream/main `6e2ac3dcd`, to `66601a7fb`, then a test commit and the stage rebuild on top); that head is `pp_review4`: the same runtime minus the two transport commits, plus round 3, rebased onto main `d9ca9e55a` (23 commits). The transport port alone is `k3_pp_transport` = `8126172f8`, stacked on it. GitHub has reported the PR unmergeable since PR 4527 landed on 2026-09-09.

Candidate `pp_review5` = `6042863a4` (2026-09-10): the same 19 commits rebased onto upstream/main `d398a8fb9`. Two files conflicted, both against PR 4527: `model.py`, where the empty stack a first stage starts from is now `h_TD.unsqueeze(1)[:, :0]` (the stack a later stage receives is unchanged), and `parallelize.py`, where the `annotate_replicated_parameters` import and the vision tower's `cpu_offload` / `dp_mesh_dims` kwargs sit beside `pp_enabled=parallel_dims.pp_enabled`. The interdiff against `75045fed5` is those two `model.py` lines; the diff against main is the same 14 files, +1693/-34. Checked on the Windows box: no conflict markers, `compileall` clean, `test_pipeline_neighbor_transport.py` and `test_integration_test_definitions.py` pass (the one failure there is upstream's `/tmp` path assertion, failing identically on main); the three K3 test files import attn-gym's CuTeDSL backend (Linux only) and run on the GPU box.

Superseded on 2026-09-11 (`pp_review4` = `66601a7fb`, not `pp_review5`, was synced onto `k3_pp_text`); the earlier plan read: before `pp_review5` is synced onto `k3_pp_text` (lease on `75045fed5`): the GPU box runs the three K3 CPU test files and the pp8 x vp4 cell on `6042863a4`, then re-pairs the Results tables on that head. Every table below was measured on the `6e2ac3dcd` base, and main since then carries PR 4484 (router backward in fp32, bf16x9 fp32 matmuls on SM 10.0+), which moves the flavor's numbers; the step-1 bitwise bar, dp1 against each pp cell, is re-read on the new head before the body is pasted. Main also carries PR 4474 (optimizer state materialized for parameters without gradients, superseding the second of Elfie's PR 4281 fixes) and PR 4527. One trap on SM 10.0+ boxes (5060 Ti, B200): `init_distributed` now calls `enable_fp32_matmul_emulation_with_bf16x9()`, which raises unless torch accepts `fp32_precision = "bfx9"`.

Paste between the markers into the PR body. Design history, the rejected designs and the per-comment answers are in `phase13_k3like_48b_posttrain/REVIEW_ANSWERS_PP_CP_2026-09-04.md` and the design note `phase13_k3like_48b_posttrain/PP_DESIGN_WORKFLOW_2026-09-07.en.md` (logbook); the body carries what the branch does and the evidence.

TODO before pasting: upload svg-1 `pp_attnres_dependencies.svg` and svg-2 `pp_stage_grid.svg` (both in this folder) by dragging them into the PR comment box, then replace the two `UPLOAD-SVG-N` placeholders below with the URLs GitHub returns. Cross-repo raw SVG links do not render (camo blocks them), so the files have to be uploaded, or the two figure lines dropped.

--- PASTE BEGIN ---

### Summary

Adds pipeline parallelism to the Kimi K3 text decoder. Before this change `parallelize.py` rejects `pipeline_parallel_degree > 1`; core's `pipeline_llm` would split the model at layer boundaries and carry one hidden-state tensor per hop, which cannot express Block Attention Residuals: every later stage needs every earlier block's residual, and the final aggregation (`output_res_proj`, then `output_res_norm`) must run only on the stage that owns `lm_head`.

After it `pipeline_kimi_k3` (in `parallelize.py`) splits the model with this model's names and builds the schedule on `AttnResPipelineStage`, a `torch.distributed.pipelining.PipelineStage` subclass: a hop carries (*hidden*, *delta*), *delta* being the block residuals the receiving rank has not seen yet; each rank keeps the blocks it has seen in one store shared by its virtual stages; the backward returns every block's gradient along the same routes.

Step 1 matches a single GPU at the printed precision in every cell of the table below (pp2's gradient norm is one bf16 ulp off); steps 10 and 20 sit inside what the same run moves with no pipeline in it at all.

### Design

What forces the protocol: every layer attends over all earlier blocks plus the running partial block, so (R1) the block stack must cross every stage boundary with the hidden state, (R2) the final aggregation runs only on the stage that owns `lm_head`, and (R3) the stack grows with depth, with a boundary inside a block putting a partial block on the wire. Sending the whole stack every hop satisfies all three and costs bytes that grow with the stage index; the delta transport sends only what the receiver lacks, which needs the routing tables, a micro-batch key that survives P2P, and a way home for a cached block's gradient.

![UPLOAD-SVG-1: the stack across stages, partial blocks on the wire, aggregation on the head stage](UPLOAD-SVG-1)

Why the rank store is enough: the schedule assigns stages $S = v \cdot P + R$, so a micro-batch returns to the same rank every $P$ stages and that rank already holds every block committed at stages $\le S-P$. A freshly committed block is therefore new on the wire for $P-1$ hops and no longer, the same-rank consumers of a block are exactly the stages congruent to its producer modulo $P$, and their number is what the tables expect as gradient deposits.

![UPLOAD-SVG-2: the looped stage grid, rows are ranks and columns virtual stages, one store per row](UPLOAD-SVG-2)

- The stage protocol, in the subclass (`pipeline_stage.py`)
  - `forward_one_chunk` assembles the full block stack the model expects from the rank's store plus the received *delta*, runs the stage, keeps the blocks the stage committed, and sends on only what the next rank lacks. The model takes and returns the whole stack and knows nothing of the transport; the chunk id comes with the call.
  - `backward_one_chunk` reads the gradient of the assembled stack, an autograd leaf the stage owns: the received columns go back as the *delta*'s gradient, dense and in wire order, and the stored columns are deposited in the rank store.
  - The stage that committed or received a block collects those deposits into its own incoming gradient (`_retrieve_recv_grads`) before its backward, which every schedule orders after the later stages' backward on the rank. No tensor hook, no autograd Function, no detach trick.
  - A micro-batch's blocks are released after the rank's last stage forward for it, so the store holds only the in-flight micro-batches.
  - Metadata inference runs the same assembly (`_compute_outputs`); `_compute_input_grads` returns dense gradients, which is where the P2P buffer finding below is handled.
- The routing tables (`layout.py`)
  - `BlockLayoutTables` simulates one micro-batch's forward in stage order over the split the trainer actually applied and tabulates, per stage, the blocks it commits, the blocks its rank already holds, and the blocks its P2P must carry; sender and receiver compute the same tables, so nothing but the delta travels.
  - The layer-to-stage map is read off the split every rank computes, with no collective; the stage-to-rank map is the schedule's own `stage_index_to_group_rank`. Uneven stages are allowed; a block boundary inside a stage is a partial block on the wire.
  - Why the delta is bounded: with $P$ ranks a block committed at stage $S$ is fresh on the wire for $P-1$ hops; from $S+P$ on every receiving rank already holds it, because its previous virtual stage was $S-P$. The per-hop payload is bounded by the commits of the last $P-1$ stages, independent of depth.
  - `attn_res_cache=False` (a `functools.partial` on the pipelining function, so every rank resolves it identically) sends the whole stack on every hop (naive); the two transports differ only in the tables, which makes them the A/B in the results. Plain `1F1B`, one stage per rank, is the naive transport by construction, so the two are bitwise there; where a rank holds more than one stage they are not, because the cached path assembles received blocks next to locally held ones and the backward sums the same contributions in a different association (at pp2 x vp2 on 2 x H100, step 10 reads `3.150940` cached against `3.514970` whole-stack, 1.17% and 12.9% against dp1 -- the class an accumulation-order change costs with no pipeline at all).
- The split (`parallelize.py`, where every model keeps its parallelism entry points): core's `llm_split_with_pinned_modules` builds it, with the vision tower pinned to the first stage through #4560's `pipeline_with_first_stage_modules` and the AttnRes aggregation to the last; K3 computes it once and hands the same object to core and to the layer map.
- The stages: core's `pipeline_llm` constructs plain `PipelineStage`s, as on main; K3 rebuilds each one the schedule holds as an `AttnResPipelineStage` from the constructed stage's own fields and puts it back in the schedule.
- The model (`model.py`): the first layer of a block joins the stack before its sub-layers attend, so a stage boundary at a block start needs nothing special and the stack a stage receives is exactly the stack the layers read; the head-owning stage alone runs the aggregation.
- What this replaced: the reviewed version carried the same protocol in a 1228-line adapter that wrapped `forward_one_chunk`, `backward_one_chunk` and `step`, kept a thread-local micro-batch id, and bridged the same-rank gradient path with a tensor grad hook and an autograd Function. The subclass implements it once, on the stage's own methods, in 388 lines; the adapter's numerics are reproduced to within one bf16 rounding (table below).

### Results

```bash
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 100 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
cell dp1 1; cell pp2 2 $P; cell pp2_vp2 2 $P --parallelism.pipeline_parallel_schedule Interleaved1F1B
```

4 x H100 PCIe, `kimi_k3_debugmodel` (24 layers), one seed checkpoint, 1024 tokens per step as four 256-token micro-batches; the naive row sets `attn_res_cache=False`; percentages against dp1.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `3.373330` | `18.625` | `5.4375` | `3.9844` |
| pp2 | `12.605700` | `3.227050` (+3.61%) | `3.288290` (-2.52%) | `18.75` (+0.67%) | `5.6875` (+4.60%) | `3.7344` (-6.27%) |
| pp2 x vp2, cached | `12.605700` | `3.150940` (+1.17%) | `3.349300` (-0.71%) | `18.625` | `3.7188` (-31.6%) | `4.0625` (+1.96%) |
| pp2 x vp2, naive | `12.605700` | `3.514970` (+12.85%) | `3.281700` (-2.72%) | `18.625` | `6.0312` (+10.9%) | `3.6719` (-7.84%) |

1024 tokens because four stages need four micro-batches and the multimodal loader needs 256 tokens per micro-batch. Steps stop at 20 because the reference memorises the 32-sample debug set after that.

dp2 x pp2 and dp2 x pp2 x vp2 against dp2, with dp2 x ep2 beside them (2048 tokens per step): pending, running.

The KDA capability guard was widened locally to admit SM 9.0 for these runs; it is not part of this PR.

### A `torch.distributed.pipelining` finding

With one layer per stage the last stage holds only the head, whose first op on the block stack is a `cat`, so autograd hands the stage's input gradients back as views; `PipelineStage._backward_metadata_inference` records those strides, `_create_grad_recv_info` allocates the receive buffer with `torch.empty_strided`, and c10d rejects it at the first `RECV_B` with "Tensors for P2P must be non-overlapping and dense".

Every other split passed because a later op consumed the input and autograd accumulated a dense gradient. The subclass returns dense gradients from `_compute_input_grads`; the library-side fix would be a dense `torch.empty` receive buffer and `.contiguous()` before the send.

### Transport (round 3): moved out of this PR

The two transport commits that sat on the round-2 head (`fd7ff7400`, the communicator warm-up, and `75045fed5`, the port of @elfiegg's multi-node hang fix) are no longer in this PR. The warm-up is a no-op by construction: the NCCL INIT log of a pp8 run shows the 8-rank communicator created at `init_process_group` and the sub-groups by `ncclCommSplit` at mesh build, nothing lazy during the schedule. The neighbour-group transport is opt-in for a two-node hang this PR's runtime does not cause and cannot reproduce on one node; it lives on the fork branch `k3_pp_transport` = `8126172f8` (the port alone, stacked on the review head) for the two-node evidence, and comes back as its own PR if that evidence holds. That neighbour-group work changes no tensor payload (it is about which communicator a send uses, not what is sent), so nothing in the Results section depends on it; it is unrelated to `attn_res_cache`, which chooses what a hop carries.

### Changed files

    torchtitan/config/
      configs.py                            +9/-0    module_fqns_per_model_part and pipeline_parallel_layers_per_stage are exclusive
    torchtitan/distributed/
      pipeline_parallel.py                  +64/-19  llm_split_with_pinned_modules, last-stage pinned modules; the injected split clears the knob that derived it
    torchtitan/models/kimi_k3/
      pipeline_stage.py                     +367/-0  AttnResPipelineStage and the rank-local cache (new)
      layout.py                             +185/-0  BlockLayoutTables from the split the trainer applied (new)
      parallelize.py                        +118/-3  the pipelining entry: the split, the stage rebuild, the tables, the transport switch; pipeline parallel off the unsupported list
      model.py                              +40/-23  the block stack in and out of a stage; the block's first layer joins the stack before attending
      __init__.py                           +18/-6   registers the pipelining_fn; the 33-layer flavor for the pp8 x vp4 cell
    tests/unit_tests/cpu/
      test_pipeline_llm_split.py            +145/-0  the split, the pinned modules at both ends, the pp8 x vp4 recipe's split, the shared debug model's depth (new)
      test_kimi_k3_pp_layout.py             +115/-0  the tables: uneven split, cache on and off (new)
      test_kimi_k3_stage_swap.py            +84/-0   the stage rebuild for single- and multi-stage schedules (new)
      test_kimi_k3_pp_stage.py              +79/-0   assembly, routing, the gradient split, the store (new)
      test_pipeline_parallel.py             +34/-0   the injected split clears the knob that derived it
      test_config_manager.py                +10/-0   the split-knob exclusivity
      test_integration_test_definitions.py  +2/-0    the two pipeline cells are registered in the B200 suite
    tests/integration_tests/b200.py         +12/-0   the pp8 x vp4 and pp2 x vp2 cells
    torchtitan_recipes/tests/b200.py        +42/-0   the pp8 x vp4 and pp2 x vp2 configurations; pp8 x vp4 hands core's 32-stage split to module_fqns_per_model_part

### CI/CD Coverage

Three CPU unit tests (the split, the layout tables, the stage's carrier handling) run in the default suite; two integration cells in the B200 suite, beside K3's existing cell, since Attention Gym's KDA runs only on SM100/SM103: pp2 x vp2 on two GPUs on the shared debug model, the smallest shape where a rank receives a block it already holds, and pp8 x vp4 on eight, where 32 stages put a boundary between almost every pair of layers. A plain pp2 cell would exercise none of this: with one stage per rank no rank ever receives a block twice, so the delta is the whole stack and the two transports are the same code path.

### Review round 1

- The one-line comments are applied as asked (comment revert, `first_layer_in_block`, the split function public, one integration cell rather than two).
- The even-split precondition is gone: the tables follow whatever split the trainer applied, learned by one all-gather; a 5/7/6/6 split is a unit test.
- The transport switch left the model config for the pipelining entry; the split became a pure function of the config; the stale naive-mode probe was deleted.
- The block's first layer joins the stack before its sub-layers attend ("cat at the start"); the rank store releases a micro-batch's blocks when the rank is done with them, not at step end.
- The 32-layer flavor is replaced by making the one debug model irregular (now 33 layers, the 93-layer model's partial block of 9, 35 units no pipeline shape divides) and the whole pp x vp matrix rerun on it, which is what surfaced the P2P buffer finding.
- The adapter and its wrappers were replaced by the `PipelineStage` subclass above; the reconstruction of how the adapter got there, the rejected designs, and why torch's per-stage `fwd_cache` cannot serve a non-adjacent consumer are in the logbook document linked from the top.

### Review round 3 (2026-09-11, Tianyu)

- Rebased onto today's main (`d9ca9e55a`), so CI can run it.
- The pipeline split is core's, in two halves. The vision tower rides with the embedding through #4560's public `pipeline_with_first_stage_modules`. The AttnRes aggregation has to sit next to the head, which that entry cannot express, so `_generate_llm_fqn_per_model_part` and `pipeline_llm` take `last_stage_modules` (appended after `norm, lm_head`, counted as no layer, forwarded by `pipeline_with_first_stage_modules`); with the default the generator is unchanged, checked against the previous function on 11232 stage/layer/weight shapes. The model-specific split function is gone.
- The shared `"debugmodel"` registry entry is main's again (24 layers); the 33-layer shape is a flavor of its own that only `kimi_k3_debugmodel_pp8_vp4` builds, so no other K3 cell changes depth because of this PR. A second pipeline cell, pp2 x vp2 on two GPUs, runs on the shared model: uneven over four stages, a block boundary inside a stage, and two stages per rank so the rank cache and the gradient deposits are used.
- No new config field. No `layers_per_stage` gives the pp8 x vp4 cell's 32 stages over 35 units (`ceil(35 / n)` never equals 32), so that recipe hands core's own generated split for 32 stages to `module_fqns_per_model_part`, with `vision_encoder` on stage 0 and the aggregation modules on stage 31 where K3's entry pins them; a CPU test asserts that split. The real 93-layer model reaches 32 stages through `layers_per_stage=3`. The pp2 x vp2 recipe sets nothing: looped schedules default to two stages per rank.
- Adding the exclusivity check surfaced that `pipeline_with_first_stage_modules` spells a split out while leaving `pipeline_parallel_layers_per_stage` set, so the knob is silently ignored from that point on; it now clears the field in the same `dataclasses.replace`. That site is shared by every model using the entry on main today (kimi_k2_7, muse_glimmer, qwen3_5, qwen3_6, qwen3_8).
- `ParallelismConfig` refuses `module_fqns_per_model_part` together with `pipeline_parallel_layers_per_stage` in `__post_init__`; the pipelining entry no longer rewrites the config.
- The layer-to-stage map is read off the split (a pure function of the config every rank computes), no all-gather, and the split comes from a public entry point: `llm_split_with_pinned_modules` returns the split with the caller's modules pinned to both ends, so this model computes it once, hands it to `pipeline_llm` through `parallelism.module_fqns_per_model_part` and reads the same object. `pipeline_with_first_stage_modules` shares that code rather than owning a copy and gains `last_stage_module_fqns`; `pipeline_llm` keeps only `stage_class`. No private core function is imported in the model.
- `RankStore` is `PPRankLocalCache`, one per rank shared by its stages, blocks device-resident; the block's first-layer flag is computed at init; the pipeline files type-check under the pinned pyrefly.
- Transport out (above). `stage_class` is gone from core: core builds plain `PipelineStage`s and K3 rebuilds each one from its own fields as an `AttnResPipelineStage` (`dd1c0b925`). Against the previous head, on one seed checkpoint and one inductor cache, dp1, pp2, pp2 x vp2, pp4 interleaved and the 8-GPU pp8 x vp4 cell read the same at every printed step.

### Review round 2

- `pipeline_llm` keeps only the `stage_class` parameter, no docstring (the file has none) and the pyrefly suppression the hook had dropped is back; the core diff is the seam and nothing else. (Superseded in round 3: `stage_class` is gone.)
- The pipelining entry, the split and the stage lookup moved from a `pipeline.py` into `parallelize.py`, where every model keeps its parallelism entry points; `layout.py` and `pipeline_stage.py` stay as files.
- The full-attention helper's docstring, the debug registry comment and the residual docstring are one line or upstream's own; the helper is the deduction asked for ("(3 KDA + 1 MLA) * k + remainder"), shared by the 30-layer and the 93-layer model.

--- PASTE END ---
