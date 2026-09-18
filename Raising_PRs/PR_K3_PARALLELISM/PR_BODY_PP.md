# PR title: [Kimi K3] Pipeline parallelism for the text decoder: the block attention residual crosses stages

Results updated 2026-09-13: same 4 x H100 PCIe box, `c4_test` text rows instead of the memorised debug set, one reference per stream accumulating in fp32 as the pipeline does, steps 1 / 10 / 20 / 100; logs and every step in `phase13_k3like_48b_posttrain/pp_h100x4_c4_logs_2026-09-13/`. Head for the body: `pp_review4` = `8de0c078c` (the split moved into Kimi K3); `k3_pp_text` = `8de0c078c` (fast-forward, 2026-09-13). The cached rows move further from the reference at steps 10-20 and are back within 0.10% (1024 tokens) / 2.1% (dp2, 2048) at step 100. Code review of the deposit path: every deposit is counted against the layout and a missing, extra or uncollected one raises, so the difference is the order of the bf16 additions (deposits summed, then added to the gradient that came down the chain). Measured on the 5060 box (not in these tables): pp4 x vp4 step-1 gradients, cache off 680/680 bitwise with dp1 in bf16 and fp32; cache on 346/680 in both, max relative difference 7.4e-2 in bf16 and 9.1e-6 in fp32 (`phase13_k3like_48b_posttrain/PP_CACHE_ORDER_PROBE_5060_2026-09-13.md`). Open: whether the dp2 cached cell's step-100 -2.08% sits inside the spread of order-only samples (more queued on the H100). The debug-set tables (2026-09-12) stay below the c4 ones. The numerics thread gets the c4 tables through `REPLY_4312_NUMERICS_C4_2026-09-13.md`. "identical" in the tables is the logged loss (5 decimals) and grad norm (4); `8de0c078c` is `c4fee4afd` with the commit-message trailers removed, same tree.

PR 4312. `pp_review4` = `6e1f5e41c` (2026-09-13): the one-way split check in `ParallelismConfig.__post_init__`, and the one line it needs in `pipeline_parallel.py` (no comment); `k3_pp_text` = `6e1f5e41c` (fast-forward, 2026-09-13); `k3_pp_text` = `ae2e7b9c4` (fast-forward from `8de0c078c`, 2026-09-13). `8de0c078c` since 2026-09-13 (fast-forward from `dbc425403`: the split moved into Kimi K3, `pipeline_parallel.py` +4/-1 against main). Before that `dbc425403` since 2026-09-12 (fast-forward from `dd1c0b925`: the B200 cells, the per-rank stage count removed, the comment and docstring trims; GitHub: 31 commits, 16 files, +1324/-51, still `dirty` -- `torchtitan/config/configs.py` conflicts with main `56a721b64`, 17 commits past the base). Before that it was `dd1c0b925` (moved with lease from `75045fed5`, which was 19 commits on upstream/main `6e2ac3dcd`, to `66601a7fb`, then a test commit and the stage rebuild on top); that head is `pp_review4`: the same runtime minus the two transport commits, plus round 3, rebased onto main `d9ca9e55a` (23 commits). The transport port alone is `k3_pp_transport` = `8126172f8`, stacked on it. GitHub has reported the PR unmergeable since PR 4527 landed on 2026-09-09.

Candidate `pp_review5` = `6042863a4` (2026-09-10): the same 19 commits rebased onto upstream/main `d398a8fb9`. Two files conflicted, both against PR 4527: `model.py`, where the empty stack a first stage starts from is now `h_TD.unsqueeze(1)[:, :0]` (the stack a later stage receives is unchanged), and `parallelize.py`, where the `annotate_replicated_parameters` import and the vision tower's `cpu_offload` / `dp_mesh_dims` kwargs sit beside `pp_enabled=parallel_dims.pp_enabled`. The interdiff against `75045fed5` is those two `model.py` lines; the diff against main is the same 14 files, +1693/-34. Checked on the Windows box: no conflict markers, `compileall` clean, `test_pipeline_neighbor_transport.py` and `test_integration_test_definitions.py` pass (the one failure there is upstream's `/tmp` path assertion, failing identically on main); the three K3 test files import attn-gym's CuTeDSL backend (Linux only) and run on the GPU box.

Superseded on 2026-09-11 (`pp_review4` = `66601a7fb`, not `pp_review5`, was synced onto `k3_pp_text`); the earlier plan read: before `pp_review5` is synced onto `k3_pp_text` (lease on `75045fed5`): the GPU box runs the three K3 CPU test files and the pp8 x vp4 cell on `6042863a4`, then re-pairs the Results tables on that head. Every table below was measured on the `6e2ac3dcd` base, and main since then carries PR 4484 (router backward in fp32, bf16x9 fp32 matmuls on SM 10.0+), which moves the flavor's numbers; the step-1 bitwise bar, dp1 against each pp cell, is re-read on the new head before the body is pasted. Main also carries PR 4474 (optimizer state materialized for parameters without gradients, superseding the second of Elfie's PR 4281 fixes) and PR 4527. One trap on SM 10.0+ boxes (5060 Ti, B200): `init_distributed` now calls `enable_fp32_matmul_emulation_with_bf16x9()`, which raises unless torch accepts `fp32_precision = "bfx9"`.

Paste between the markers into the PR body. Design history, the rejected designs and the per-comment answers are in `phase13_k3like_48b_posttrain/REVIEW_ANSWERS_PP_CP_2026-09-04.md` and the design note `phase13_k3like_48b_posttrain/PP_DESIGN_WORKFLOW_2026-09-07.en.md` (logbook); the body carries what the branch does and the evidence.

TODO before pasting: upload svg-1 `pp_attnres_dependencies.svg` and svg-2 `pp_stage_grid.svg` (both in this folder) by dragging them into the PR comment box, then replace the two `UPLOAD-SVG-N` placeholders below with the URLs GitHub returns. Cross-repo raw SVG links do not render (camo blocks them), so the files have to be uploaded, or the two figure lines dropped.

--- PASTE BEGIN ---

Rebase note, 2026-09-18 (not for pasting): the PR head `de6f29514` now replays onto torchtitan main `a3a819c67` as `ab8ea5f61`, 37 commits, in `/tmp/wt_pr4312_plain`. Nothing here is edited from that, because every number below is an H100 measurement and a rebase can move the reference itself. The rows that need an H100 rerun against the new base before this body is reposted are all of them, in all four tables: dp1, pp2, pp2 x vp2 naive, pp4 x vp4 naive, pp2 x vp2 cached and pp4 x vp4 cached, at steps 1, 10, 50 and 100 for both loss and grad norm, in the 09-13 c4 table, the appendix c4 table and the 09-12 debug-set table.

The two cells run on 2026-09-18 are smoke only and never go into this body: `kimi_k3_debugmodel_pp2_vp2` on 2 GPUs and `kimi_k3_debugmodel_pp8_vp4` on 8, both rc 0 over three steps on an RTX 5060 Ti box, loss 12.34963 / 10.93338 / 8.04338 and 12.59161 / 11.15744 / 8.35423. They say the pipeline runs on the new base; they say nothing about the numbers above.

### Summary

Adds pipeline parallelism to the Kimi K3 text decoder. Before this change `parallelize.py` rejects `pipeline_parallel_degree > 1`; core's `pipeline_llm` would split the model at layer boundaries and carry one hidden-state tensor per hop, which cannot express Block Attention Residuals: every later stage needs every earlier block's residual, and the final aggregation (`output_res_proj`, then `output_res_norm`) must run only on the stage that owns `lm_head`.

After it `pipeline_kimi_k3` (in `parallelize.py`) splits the model with this model's names and builds the schedule on `AttnResPipelineStage`, a `torch.distributed.pipelining.PipelineStage` subclass: a hop carries (*hidden*, *delta*), *delta* being the block residuals the receiving rank has not seen yet; each rank keeps the blocks it has seen in one store shared by its virtual stages; the backward returns every block's gradient along the same routes.

Step 1 is identical to the single-GPU reference in every cell. With the total grad norm taken in fp32 -- in bf16 the norm, and the clip factor it sets every step here, depend on how the pipeline groups the parameters (https://github.com/pytorch/pytorch/pull/194033) -- the whole-stack pipeline cells stay identical to the reference for 100 steps: the loss on every step, the grad norm on all but one (fourth decimal). The cached cells differ only in the order in which a cached block's gradient contributions are added: on step-1 gradients at pp4 x vp4, cache off is bitwise on all 680 parameters, and cache on differs in 334 (layers 0-11 and `tok_embeddings`) by up to 7.4e-2 relative in bf16 and 9.1e-6 in fp32. The same cells with the default bf16 norm are in the appendix.

### Design

What forces the protocol: every layer attends over all earlier blocks plus the running partial block, so (R1) the block stack must cross every stage boundary with the hidden state, (R2) the final aggregation runs only on the stage that owns `lm_head`, and (R3) the stack grows with depth, with a boundary inside a block putting a partial block on the wire. Sending the whole stack every hop satisfies all three and costs bytes that grow with the stage index; the delta transport sends only what the receiver lacks, which needs the routing tables, a micro-batch key that survives P2P, and a way home for a cached block's gradient.

<img width="1080" height="540" alt="pp_dual_gradient_bridge_v2" src="https://github.com/user-attachments/assets/743ff96c-a9a9-410f-9c9e-c351bde73c70" />

Why the rank store is enough: the schedule assigns stages $S = v \cdot P + R$, so a micro-batch returns to the same rank every $P$ stages and that rank already holds every block committed at stages $\le S-P$. A freshly committed block is therefore new on the wire for $P-1$ hops and no longer, the same-rank consumers of a block are exactly the stages congruent to its producer modulo $P$, and their number is what the tables expect as gradient deposits.

<img width="1100" height="408" alt="pp_stage_grid" src="https://github.com/user-attachments/assets/ea5bccc6-3be8-4a08-9278-c976848e655b" />

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
  - `attn_res_cache=False` (a `functools.partial` on the pipelining function, so every rank resolves it identically) sends the whole stack on every hop (naive); the two transports differ only in the tables, which makes them the A/B in the results. Plain `1F1B`, one stage per rank, is the naive transport by construction, so the two are bitwise there; where a rank holds more than one stage they are not, because the cached path assembles received blocks next to locally held ones and the backward sums the same contributions in a different association (step-1 gradients at pp4 x vp4: the same 334 of 680 parameters differ in bf16 and in fp32, the largest relative difference falling from 7.4e-2 to 9.1e-6, i.e. a reordered sum).
- The split (`parallelize.py`, where every model keeps its parallelism entry points): Kimi K3 builds it bottom up, the vision tower on the first stage and the AttnRes aggregation on the last, computes it once and hands the same list to `pipeline_llm` (through `module_fqns_per_model_part`) and to the layer map.
- The stages: core's `pipeline_llm` constructs plain `PipelineStage`s, as on main; K3 rebuilds each one the schedule holds as an `AttnResPipelineStage` from the constructed stage's own fields and puts it back in the schedule.
- The model (`model.py`): the first layer of a block joins the stack before its sub-layers attend, so a stage boundary at a block start needs nothing special and the stack a stage receives is exactly the stack the layers read; the head-owning stage alone runs the aggregation.
- Core: `ParallelismConfig.__post_init__` refuses an explicit `module_fqns_per_model_part` together with `pipeline_parallel_layers_per_stage`; because `dataclasses.replace` re-runs it, `pipeline_with_first_stage_modules` clears `layers_per_stage` once it spells its split out (one line; the models on that entry today: kimi_k2_7, muse_glimmer, qwen3_5, qwen3_6, qwen3_8), and Kimi K3's entry clears it the same way when it hands its own split to `pipeline_llm`.
- The shared `debugmodel` flavor is unchanged (24 layers). `debugmodel_33_layers` exists only for the pp8 x vp4 cell: 35 units that no pipeline shape divides, and no `layers_per_stage` reaches 32 stages there, so that recipe passes Kimi K3's split through `module_fqns_per_model_part`.
- What this replaced: the reviewed version carried the same protocol in a 1228-line adapter that wrapped `forward_one_chunk`, `backward_one_chunk` and `step`, kept a thread-local micro-batch id, and bridged the same-rank gradient path with a tensor grad hook and an autograd Function. The subclass implements it once, on the stage's own methods, in 367 lines.

### Results

The c4 flavors (`kimi_k3_debugmodel_c4`, `kimi_k3_debugmodel_c4_pp_naive`) and the reference / floor switches (`NOSYNC_GA`, `MB_REVERSE`) are in [this probe patch](https://github.com/QIU023/torchtitan_attention_residual/blob/38c588bcc76c7dd3d97aa99722b9d9067ef951d2/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/pp4h_probe_c4.patch), the 16-stage pp4 x vp4 switch (`PP_STAGES_PER_RANK`) in [this one](https://github.com/QIU023/torchtitan_attention_residual/blob/38c588bcc76c7dd3d97aa99722b9d9067ef951d2/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/pp_stages_per_rank.patch), the fp32 total grad norm (`GN_FP32`) in [this hack](https://github.com/QIU023/torchtitan_attention_residual/blob/38c588bcc76c7dd3d97aa99722b9d9067ef951d2/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/gn_fp32_hack.py), the whole matrix in [this script](https://github.com/QIU023/torchtitan_attention_residual/blob/38c588bcc76c7dd3d97aa99722b9d9067ef951d2/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/run_pp_c4.sh); none of them is part of this PR.

```bash
python gn_fp32_hack.py . && export GN_FP32=1   # total grad norm in fp32 (drop both for the appendix tables)
export TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor TRITON_CACHE_DIR=$PWD/cache/triton   # one compile cache for every cell, run in this order so the reference fills it first
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --checkpoint.enable --parallelism.data_parallel_shard_degree 1"
torchrun --nproc_per_node=1 $COMMON --config kimi_k3_debugmodel_c4 --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; c=$3; shift 3; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --config $c --training.steps 100 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"; IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
NOSYNC_GA=1 cell ref 1 kimi_k3_debugmodel_c4; MB_REVERSE=1 cell reversed 1 kimi_k3_debugmodel_c4; cell dp1 1 kimi_k3_debugmodel_c4
cell pp2 2 kimi_k3_debugmodel_c4 $P; cell vp2_cached 2 kimi_k3_debugmodel_c4 $P $IL; cell vp2_naive 2 kimi_k3_debugmodel_c4_pp_naive $P $IL
PP_STAGES_PER_RANK=4 cell pp4vp4_cached 4 kimi_k3_debugmodel_c4 --parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL
# dp2 table: --parallelism.data_parallel_shard_degree 2 and --training.num-tokens-per-train-step 2048, its own seed checkpoint, the same cache, the dp2 reference first
```

4 x H100 PCIe, `kimi_k3_debugmodel` (24 layers) reading `c4_test` as text-only 256-token rows, one seed checkpoint and one compile cache shared by every cell (a fresh cache per cell autotunes other kernels and moves the logged norm by itself), four 256-token micro-batches per rank, total grad norm in fp32; each cell gives the raw value and, beneath it, the change against the reference.

| cell | loss, step 1 | step 10 | step 50 | step 100 | grad norm, step 1 | step 10 | step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 ¹ | `12.609980` | `3.595760` | `2.672340` | `2.533960` | `16.9193` | `5.0429` | `1.4695` | `1.6014` |
| pp2 | `12.609980`<br>identical | `3.595760`<br>identical | `2.672340`<br>identical | `2.533960`<br>identical | `16.9193`<br>identical | `5.0429`<br>identical | `1.4695`<br>identical | `1.6014`<br>identical |
| pp2 x vp2, naive ² | `12.609980`<br>identical | `3.595760`<br>identical | `2.672340`<br>identical | `2.533960`<br>identical | `16.9193`<br>identical | `5.0429`<br>identical | `1.4695`<br>identical | `1.6014`<br>identical |
| pp4 x vp4, naive ²³ | `12.609980`<br>identical | `3.595760`<br>identical | `2.672340`<br>identical | `2.533960`<br>identical | `16.9193`<br>identical | `5.0429`<br>identical | `1.4695`<br>identical | `1.6014`<br>identical |
| pp2 x vp2, cached | `12.609980`<br>identical | `3.554240`<br>-1.15% | `2.723740`<br>+1.92% | `2.571310`<br>+1.47% | `16.9177`<br>-0.01% | `5.2631`<br>+4.37% | `1.7956`<br>+22.19% | `1.6073`<br>+0.37% |
| pp4 x vp4, cached ³ | `12.609980`<br>identical | `3.309360`<br>-7.96% | `2.678180`<br>+0.22% | `2.554590`<br>+0.81% | `16.9167`<br>-0.02% | `3.5408`<br>-29.79% | `1.6092`<br>+9.51% | `1.5763`<br>-1.57% |

- ¹ reference: the micro-batches accumulate in fp32 with the gradient sync on the last one, as the pipeline does (`NOSYNC_GA`)
- ² naive transport, the whole block stack on every hop (`attn_res_cache=False`); "cached" rows use the rank cache, the default
- ³ 16 stages, four per rank (`PP_STAGES_PER_RANK=4`)
- all 100 steps compared: pp2 matches on every step; the two naive rows match the loss on every step and the grad norm on every step but 59 (`1.6628` against `1.6629`)

dp2, 2048 tokens per step, same protocol.

| cell | loss, step 1 | step 10 | step 50 | step 100 | grad norm, step 1 | step 10 | step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 ¹ | `12.580740` | `3.576250` | `2.657010` | `2.420420` | `14.4170` | `12.4659` | `1.7446` | `1.0313` |
| dp2 x pp2 ⁴ | `12.580740`<br>identical | `3.576250`<br>identical | `2.657010`<br>identical | `2.420420`<br>identical | `14.4170`<br>identical | `12.4659`<br>identical | `1.7446`<br>identical | `1.0313`<br>identical |
| dp2 x pp2 x vp2, naive ² | `12.580740`<br>identical | `3.576250`<br>identical | `2.657010`<br>identical | `2.420420`<br>identical | `14.4170`<br>identical | `12.4659`<br>identical | `1.7446`<br>identical | `1.0313`<br>identical |
| dp2 x pp2 x vp2, cached | `12.580740`<br>identical | `3.305220`<br>-7.58% | `2.664850`<br>+0.30% | `2.434300`<br>+0.57% | `14.4191`<br>+0.01% | `4.5317`<br>-63.65% | `1.7570`<br>+0.71% | `1.0893`<br>+5.62% |

- ¹ ² as above; dp2 x pp2 and the naive row match on all 100 steps
- ⁴ run on the reference's warm compile cache; on a fresh cache of its own its logged norm differed from step 1 (+0.01%) while its dumped step-1 gradients were bitwise with the reference (680/680)

The KDA capability guard was widened locally to admit SM 9.0 for these runs; it is not part of this PR.

### A `torch.distributed.pipelining` finding

With one layer per stage the last stage holds only the head, whose first op on the block stack is a `cat`, so autograd hands the stage's input gradients back as views; `PipelineStage._backward_metadata_inference` records those strides, `_create_grad_recv_info` allocates the receive buffer with `torch.empty_strided`, and c10d rejects it at the first `RECV_B` with "Tensors for P2P must be non-overlapping and dense".

Every other split passed because a later op consumed the input and autograd accumulated a dense gradient. The subclass returns dense gradients from `_compute_input_grads`; the library-side fix would be a dense `torch.empty` receive buffer and `.contiguous()` before the send.

### Changed files

    torchtitan/config/
      configs.py                            +9/-0    module_fqns_per_model_part and pipeline_parallel_layers_per_stage are exclusive
    torchtitan/distributed/
      pipeline_parallel.py                  +3/-1    pipeline_with_first_stage_modules clears the knob that derived the split it spells out
    torchtitan/models/kimi_k3/
      pipeline_stage.py                     +367/-0  AttnResPipelineStage and the rank-local cache (new)
      layout.py                             +185/-0  BlockLayoutTables from the split the trainer applied (new)
      parallelize.py                        +200/-3  the pipelining entry: Kimi K3's own split, the stage rebuild, the tables, the transport switch; pipeline parallel off the unsupported list
      model.py                              +40/-23  the block stack in and out of a stage; the block's first layer joins the stack before attending
      __init__.py                           +18/-6   registers the pipelining_fn; the 33-layer flavor for the pp8 x vp4 cell
    tests/unit_tests/cpu/
      test_kimi_k3_pp_layout.py             +209/-0  the tables: uneven split, cache on and off; Kimi K3's split, the pp8 x vp4 recipe's split, the shared debug model's depth (new)
      test_kimi_k3_stage_swap.py            +84/-0   the stage rebuild for single- and multi-stage schedules (new)
      test_kimi_k3_pp_stage.py              +79/-0   assembly, routing, the gradient split, the store (new)
      test_pipeline_parallel.py             +34/-0   the injected split clears the knob that derived it
      test_config_manager.py                +10/-0   the split-knob exclusivity
      test_integration_test_definitions.py  +2/-0    the two pipeline cells are registered in the B200 suite
    tests/integration_tests/b200.py         +12/-0   the pp8 x vp4 and pp2 x vp2 cells
    torchtitan_recipes/tests/b200.py        +37/-0   the pp8 x vp4 and pp2 x vp2 configurations; pp8 x vp4 hands Kimi K3's 32-stage split to module_fqns_per_model_part

### CI/CD Coverage

Three CPU unit tests (the split, the layout tables, the stage's carrier handling) run in the default suite; two integration cells in the B200 suite, beside K3's existing cell, since Attention Gym's KDA runs only on SM100/SM103: pp2 x vp2 on two GPUs on the shared debug model, the smallest shape where a rank receives a block it already holds, and pp8 x vp4 on eight, where 32 stages put a boundary between almost every pair of layers. A plain pp2 cell would exercise none of this: with one stage per rank no rank ever receives a block twice, so the delta is the whole stack and the two transports are the same code path.

### Appendix: BF16 reduction numerical results (without PyTorch #194033, current HEAD)

The same cells with the default bf16 total grad norm. Step 1 is identical in every cell; later steps also carry the bf16 norm's dependence on how the parameters are grouped, which the fp32 tables above remove.

c4 (2026-09-13): 4 x H100 PCIe, `kimi_k3_debugmodel` (24 layers) reading `c4_test` as text-only 256-token rows (`kimi_k3_debugmodel_c4`, a local data flavor that is not part of this PR; the 32-sample debug set is memorised by step 20, c4 is not by step 100), one seed checkpoint, 1024 tokens per step as four 256-token micro-batches; each cell gives the raw value and, beneath it, the change against the reference.

| cell | loss, step 1 | step 10 | step 50 | step 100 | grad norm, step 1 | step 10 | step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 ¹ | `12.609980` | `3.335430` | `2.708880` | `2.568440` | `16.875` | `3.2812` | `1.7734` | `1.5547` |
| pp2 | `12.609980`<br>identical | `3.334370`<br>-0.03% | `2.711350`<br>+0.09% | `2.562870`<br>-0.22% | `16.875`<br>0% | `3.2969`<br>+0.48% | `1.6094`<br>-9.25% | `1.5312`<br>-1.51% |
| pp2 x vp2, cached | `12.609980`<br>identical | `3.310030`<br>-0.76% | `2.693930`<br>-0.55% | `2.565810`<br>-0.10% | `16.875`<br>0% | `3.2812`<br>0% | `1.6328`<br>-7.93% | `1.5781`<br>+1.51% |
| pp2 x vp2, naive ² | `12.609980`<br>identical | `3.331430`<br>-0.12% | `2.713300`<br>+0.16% | `2.571170`<br>+0.11% | `16.875`<br>0% | `3.3125`<br>+0.95% | `1.7578`<br>-0.88% | `1.5547`<br>0% |
| pp4 x vp4, cached ³ | `12.609980`<br>identical | `3.528470`<br>+5.79% | `2.722820`<br>+0.51% | `2.571010`<br>+0.10% | `16.875`<br>0% | `10`<br>+204.77% | `1.6719`<br>-5.72% | `1.6094`<br>+3.52% |
| pp4 x vp4, naive ²³ | `12.609980`<br>identical | `3.334860`<br>-0.02% | `2.704290`<br>-0.17% | `2.565350`<br>-0.12% | `16.875`<br>0% | `3.2656`<br>-0.48% | `1.7422`<br>-1.76% | `1.5391`<br>-1.00% |
| dp1 ⁴ | `12.609980`<br>identical | `3.421960`<br>+2.59% | `2.668690`<br>-1.48% | `2.536600`<br>-1.24% | `16.875`<br>0% | `5.0625`<br>+54.29% | `1.4609`<br>-17.62% | `1.5703`<br>+1.00% |
| dp1 ⁵ | `12.609980`<br>identical | `3.389450`<br>+1.62% | `2.711100`<br>+0.08% | `2.563570`<br>-0.19% | `16.875`<br>0% | `4.6875`<br>+42.86% | `1.4141`<br>-20.26% | `1.6797`<br>+8.04% |

- ¹ reference: the micro-batches accumulate in fp32 with the gradient sync on the last one, as the pipeline does (`NOSYNC_GA`)
- ² naive transport, the whole block stack on every hop (`attn_res_cache=False`); "cached" rows use the rank cache, the default
- ³ 16 stages, four per rank (`PP_STAGES_PER_RANK=4`)
- ⁴ no pipeline, default accumulation: gradient sync after every micro-batch
- ⁵ noise floor: no pipeline, micro-batch order reversed (`MB_REVERSE`)

dp2, 2048 tokens per step (four 256-token micro-batches per rank), same protocol and reference accumulation; a rerun of dp2 x pp2 matched it on all 100 steps.

| cell | loss, step 1 | step 10 | step 50 | step 100 | grad norm, step 1 | step 10 | step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 ¹ | `12.580740` | `3.663490` | `2.678790` | `2.443230` | `14.4375` | `14.1875` | `1.7656` | `1.0312` |
| dp2 x pp2 | `12.580740`<br>identical | `3.607360`<br>-1.53% | `2.677310`<br>-0.06% | `2.431600`<br>-0.48% | `14.4375`<br>0% | `13.1875`<br>-7.05% | `1.8828`<br>+6.64% | `0.9609`<br>-6.82% |
| dp2 x pp2 x vp2, cached | `12.580740`<br>identical | `3.369740`<br>-8.02% | `2.615530`<br>-2.36% | `2.392360`<br>-2.08% | `14.4375`<br>0% | `5.625`<br>-60.35% | `1.7031`<br>-3.54% | `1.0234`<br>-0.76% |
| dp2 x pp2 x vp2, naive ² | `12.580740`<br>identical | `3.687230`<br>+0.65% | `2.671240`<br>-0.28% | `2.416180`<br>-1.11% | `14.4375`<br>0% | `14.8125`<br>+4.41% | `1.8594`<br>+5.31% | `0.9961`<br>-3.40% |
| dp2 ⁴ | `12.580740`<br>identical | `3.307830`<br>-9.71% | `2.668280`<br>-0.39% | `2.426700`<br>-0.68% | `14.4375`<br>0% | `4.5312`<br>-68.06% | `1.8125`<br>+2.66% | `1.0312`<br>0% |
| dp2 ⁵ | `12.580740`<br>identical | `3.588150`<br>-2.06% | `2.681040`<br>+0.08% | `2.437380`<br>-0.24% | `14.4375`<br>0% | `12.125`<br>-14.54% | `1.8438`<br>+4.43% | `0.9766`<br>-5.29% |
| dp2 x ep2 ⁶ | `12.580740`<br>identical | `3.269320`<br>-10.76% | `2.657450`<br>-0.80% | `2.420510`<br>-0.93% | `14.4375`<br>0% | `4.4375`<br>-68.72% | `1.7422`<br>-1.33% | `1.0312`<br>0% |

- ¹ ² ⁴ ⁵ as above; ⁶ no pipeline, expert parallel 2 (another reduction order)

Debug set (2026-09-12, `--config kimi_k3_debugmodel`): 4 x H100 PCIe, `kimi_k3_debugmodel` (24 layers), one seed checkpoint, 1024 tokens per step as four 256-token micro-batches; the naive row sets `attn_res_cache=False`; each cell gives the raw value and, beneath it, the change against dp1.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `3.373330` | `18.625` | `5.4375` | `3.9844` |
| pp2 | `12.605700`<br>identical | `3.227050`<br>+3.61% | `3.288290`<br>-2.52% | `18.75`<br>+0.67% | `5.6875`<br>+4.60% | `3.7344`<br>-6.27% |
| pp2 x vp2, cached | `12.605700`<br>identical | `3.150940`<br>+1.17% | `3.349300`<br>-0.71% | `18.625`<br>0% | `3.7188`<br>-31.61% | `4.0625`<br>+1.96% |
| pp2 x vp2, naive ² | `12.605700`<br>identical | `3.514970`<br>+12.85% | `3.281700`<br>-2.72% | `18.625`<br>0% | `6.0312`<br>+10.92% | `3.6719`<br>-7.84% |
| dp1 ⁵ | `12.605700`<br>identical | `3.247610`<br>+4.27% | `3.295370`<br>-2.31% | `18.625`<br>0% | `6.6562`<br>+22.41% | `4.25`<br>+6.67% |

1024 tokens because four stages need four micro-batches and the multimodal loader needs 256 tokens per micro-batch. Steps stop at 20 because the reference memorises the 32-sample debug set after that.

dp2, 2048 tokens per step (four 256-token micro-batches per rank), same protocol; each cell gives the raw value and, beneath it, the change against dp2; the dp2 x ep2 row carries no pipeline and sizes what a change of reduction order alone does.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 | `12.521140` | `3.221120` | `2.839810` | `16.375` | `7.0625` | `2.4844` |
| dp2 x pp2 | `12.521140`<br>identical | `3.201920`<br>-0.60% | `2.846440`<br>+0.23% | `16.375`<br>0% | `5`<br>-29.20% | `2.6562`<br>+6.92% |
| dp2 x pp2 x vp2, cached | `12.521140`<br>identical | `3.205680`<br>-0.48% | `2.682970`<br>-5.52% | `16.375`<br>0% | `5.375`<br>-23.89% | `2.5156`<br>+1.26% |
| dp2 x pp2 x vp2, naive ² | `12.521140`<br>identical | `3.189240`<br>-0.99% | `2.847220`<br>+0.26% | `16.375`<br>0% | `5.5938`<br>-20.80% | `2.375`<br>-4.40% |
| dp2 x ep2 ⁶ | `12.521140`<br>identical | `3.149310`<br>-2.23% | `2.969330`<br>+4.56% | `16.375`<br>0% | `5.0625`<br>-28.32% | `2.9531`<br>+18.87% |

--- PASTE END ---
