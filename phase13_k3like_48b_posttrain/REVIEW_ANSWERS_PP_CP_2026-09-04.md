# PP PR 4312 and CP PR 4313: review answers, design notes, and the review-branch fixes (2026-09-04)

中文摘要（给自己）：

- 分支状态：所有修复都在 fork 的 review 分支上，PR 分支未动。`cp_review2` 比 PR 头多四个提交（kernel 自带集合通信的 Ulysses/KCP、SP 守卫、all-gather KV kernel、cp2 flavor 走 spmd_types）；`pp_review2` 比 PR 头多两个提交（一行类修复、ufmt）。见第 5 节。
- PP 的核心设计问题：第 2 节按 Tianyu 要的"from scratch, what's missing, so we added X" 的顺序，从 4 月 phase3 的四份 handoff 和 adapter_design.md 复原了 adapter 每个部件的来历，包括试过又放弃的四条路；第 3 节逐条回答 18 条 inline comment 和三条 high-level 意见，含 torch PP 自带 cache 能不能复用（不能，它是每个 stage 自己反向用的 per-microbatch 记录，没有跨 virtual stage 的保留和跨 stage 的梯度路由）。
- 数值：step-10 差异的主因是 bf16 的 grad-norm 归约随参数分组变化（torch 194033 在修），PR body 附录已有 fp32 下 fallback 各格两两坍缩的证据；新增的 step-1 逐参数梯度对比在第 3.3 节。
- CP：kernel 版和声明式版在同一套编译 kernel 下梯度逐位相同，此前看到的差异是 inductor autotune 在冷编译时的选择漂移；all-gather KV 从 4322 复制过来作为 MLA 的第二种 kernel。

Everything below is written for the reviewers, in English.

---

## 1. Where things stand

| branch | head | on top of the PR head | pushed to the PR? |
|---|---|---|---|
| `cp_review2` (fork) | `adc012ce4` | `f66b5de3a` CP kernels own their collectives; `3970bcd1c` SP guard in the plain-tensor splice; `adc012ce4` all-gather KV kernel; plus `84fd3ee22` cp2 flavor on spmd_types and the spmd declarations (`4b88ada6b`) merged earlier | no |
| `pp_review2` (fork) | `395fc6b30` | `e326c70a2` the one-line review fixes; `0c0e48908` ufmt on `pipeline_adapter.py`; `7dda3b847` the transport switch leaves the model config for the pipelining entry, the split becomes a pure function of the config, the stale naive-mode probe goes; `ca5f34ea8` the block layout follows the split the trainer applied (uneven stages allowed, the even-split gate gone); `eef340d25` the block's first layer joins the stack before its sub-layers attend (the cat-at-the-start refactor); `d72faf339` the rank cache releases a micro-batch's blocks when the rank is done with them; `395fc6b30` one irregular debug model (30 layers, block 12, MLA layers deduced); `c3df74847` a pipeline stage hands its inputs' gradients back dense (the torch P2P buffer finding, 3.3) | no |

The PR branches `k3_cp_text` (`b85c2a078`) and `k3_pp_text` (`087c4d177`) are untouched.

---

## 2. How the pipeline adapter got to where it is

This is the "start from scratch, what was missing, so X was added" account Tianyu asked for. It is reconstructed from the April 2026 records in `phase3_attnres_pp_integration/` (`README.md`, `adapter_design.md`, the four `handoff_status_202604*.md` session logs, `PP_Adapter_Flow.md`), the pressure-test reports of 2026-05-12 and 2026-07-22, and the current code on `k3_pp_text`. Dates are the session dates in those files.

### 2.0 The starting point: Block AttnRes under pipeline parallelism needs nothing special to be *correct*

A block attention residual attends, at every layer, over the representations of all earlier blocks (plus the running partial block). Under PP the block stack therefore has to cross every stage boundary together with the hidden state. In torchtitan a stage's `forward` may return a tuple and the next stage receives the tuple, so the first PP implementation (Phase 2 model, 2026-04-19) simply returned `(hidden, block_stack)` from every non-last stage and let `torch.distributed.pipelining` carry both. That is the **naive transport**. It is correct, it is what plain `1F1B` still runs today (one stage per rank), and it needed no adapter.

What it lacks is bounded communication: stage `S` sends every block accumulated so far, so the per-hop payload grows linearly with `S` (Comm_naive in the paper's eq. 7, `C(C-1)/2 * N_p * d` per token in total). The paper's section 4.1 describes the fix, cross-stage caching: with `V` virtual stages per rank, a rank that received blocks for its virtual stage `v` still holds them when its virtual stage `v+1` runs, so a hop only needs to carry the blocks the receiving rank has not seen. That is the whole reason the adapter exists. It is an optimisation of bytes on the wire, not a correctness component, and the naive transport remains the fallback.

### 2.1 Missing piece 1: who holds which blocks, without sending metadata

**Added: `BlockLayoutTables` (`layout.py`), 2026-04-20 morning.** Sender and receiver must agree on what a hop carries. Rather than sending indices on the wire, both sides simulate one micro-batch's forward in schedule order offline and tabulate, per stage, the blocks it commits, the blocks its rank's cache already holds at each virtual stage, and the delta the hop must carry (`delta = accumulated - receiver_cache`). The tables are a pure function of `(P, V, num_blocks, n_layers, layers_per_block)` and a layer-to-stage map. The per-hop payload is bounded by the commits of the last `P-1` stages.

The map is where the "even split" precondition comes from: under `Interleaved1F1B` a rank sees only its own stages, so the global layer-to-stage map is not available locally without a collective, and the code verifies the contiguous equal-split default instead of exchanging the real ranges (`infer_block_layout_tables_from_stages` discards the local map after checking it). `BlockLayoutTables` itself accepts an explicit map. See 3.2 for the fix.

### 2.2 Missing piece 2: a micro-batch key that survives P2P

**Added: the `forward_one_chunk` / `backward_one_chunk` patch and the thread-local, 2026-04-20 (handoff 1, issue 2).** The cache is per micro-batch, so the adapter's forward must know which micro-batch it is running. The first version keyed on `id(tensor)`; NCCL allocates fresh receive buffers, so the producer's and consumer's ids never match and every middle stage missed its cache. The schedule owns the integer `fwd_chunk_id` / `bwd_chunk_id`, but `PipelineStage` passes it to `forward_one_chunk`, not into the submodule. The adapter therefore wraps each stage's `forward_one_chunk` / `backward_one_chunk` to stash the id in a per-adapter thread-local before the submodule runs; forward and backward run synchronously on the same thread, so autograd hooks fired during backward read the same key.

This is the first of the mechanisms the reviewer flagged as "patch/thread". The first-principles version is an API on `PipelineStage` that hands the submodule its chunk id (or a stage-level context the submodule can query). We did not have that in torch, and the K3 PR keeps the wrapper because it is the only way to get the id without modifying `torch.distributed.pipelining`.

### 2.3 Missing piece 3: the backward, four rejected designs and the one that stayed

The forward delta was working on 8 GPUs on 2026-04-20 (per-hop sizes matched the tables). The gradient of a cached block has to flow from every later stage that attended over it back to the stage that committed it. What was tried, in order (handoff part 2 and 3, and the 04-21 log):

1. **Custom NCCL inside `autograd.Function.backward`** (`_SendBlockGradsBack` / `_RecvBlockGradsFromConsumers`, isend/irecv per block). Deadlocked: the autograd engine runs depth-first on one thread; a blocking `isend(...).wait()` inside backward stalls the engine while the peer rank has not reached its matching Function, and the schedule's own `SEND_B` / `RECV_B` raced on the same process group. NCCL watchdog timeout.
2. **Flush the block gradients after `backward_one_chunk`**, outside autograd. Still deadlocked: under `Interleaved1F1B` ranks reach a given micro-batch's backward at very different times, so a rank's flush posted P2P ops with no matching peer ops, and the next `SEND_B` entangled with them.
3. **One batched exchange at step end.** Fixes the deadlock (the step boundary is synchronised) but keeps every micro-batch's retained graph alive until the end of the step: memory proportional to the number of in-flight micro-batches. Rejected before running.
4. **Ride PP's own `SEND_B`**: keep received blocks autograd-attached to the tensor they arrived in, so their gradient flows back hop by hop through the schedule's existing backward P2P, with no custom collective at all. This deleted all custom gradient machinery (the file shrank from 1320 to 784 lines) and is channel A of the final design. It failed for one case: a block a rank committed at virtual stage `v` and reads back from its own cache at virtual stage `v+1`. The consumer's backward walked into the producer's forward graph and freed it; the producer's own backward, arriving later via `SEND_B`, then failed with "backward through the graph a second time".
5. **`retain_graph=True`** on every stage's backward. Correct, but +5 GiB on rank 7 for a 175M model, and it grows with `V` and the micro-batch count. Committed as a stop-gap on 2026-04-20 evening.
6. **`_LocalCacheAugment` + `_LocalCacheCapture` autograd Functions** (2026-04-20 session 3): Capture returns `None` for the tensor input, Augment adds the deposited gradient. Passed the CPU canaries, still double-backwarded on 4 GPUs: per-Function tracing showed Capture and Augment firing in the same `backward_one_chunk`, i.e. autograd still reached the producer graph from the consumer side despite the `None`.
7. **Detached cache copy + `_LocalCacheCapture` + a tensor grad hook on the producer's attached block** (2026-04-21). The rank cache stores a *detached* copy of the rank's own commit, so the consumer's Capture input has no upstream graph to walk into; Capture's backward deposits the gradient in a slot keyed `(mb, producer stage, commit index)` and returns `None`; a `register_hook` on the producer's attached block pops the slot and adds it during the producer's own backward. This is channel B of the final design. Memory returned to naive PP plus the cache footprint (7.71 vs 7.45 GiB on rank 3, PP4 x VP2, 175M). A static count of expected captures (`expected_same_rank_captures`) lets the hook detect a lost gradient and refuse the step.

So the "hook" the reviewer sees is the survivor of six attempts, each of which was either a deadlock, a memory blow-up, or an incorrect double traversal under the real schedule. The two channels are: across ranks, PP's own backward P2P (no code); on a rank, a dictionary slot bridged by one Function and one hook.

### 2.4 Missing piece 4: eviction

**Added: `_install_step_drop_patch`, `on_microbatch_end`, the VP drop-guard, 2026-04-20.** The cache is shared by a rank's virtual stages, so a micro-batch's entries cannot be dropped when the *first* virtual stage finishes its backward. The adapter marked micro-batches seen at `backward_one_chunk` exit and dropped everything after `pp_schedule.step()` returned, from the rank's last virtual stage only. This is the second wrapper. Until this week it also meant we did **not** do what the paper describes as "released as soon as the micro-batch finishes": entries lived until step end, so cache memory scaled with the number of in-flight micro-batches (the 04-21 log's envelope: `|rank_cache_at_entry| * B * T * D * 2 * M`). On the review branch (`d72faf339`) the blocks are released right after the rank's last virtual stage has run its forward for the micro-batch, which is the last read of them (the backward reads the slots and the autograd graph, not the cache); the step-end sweep remains as the safety net. The `pp_review1` integration branch attacks memory from a different side (host offload of own-rank commits, and peer offload through the Mooncake transfer engine).

### 2.5 The Kimi K3 port, August

Three things changed when the adapter moved from the `attn_res` experiment onto the K3 model:

- The block layout comes from the model config, not from marker attributes on the model (`c9b729f95`). The K3 model returns the carrier it was handed with its own commits appended, and the adapter takes the tail as this stage's commits. The old `_return_only_new_blocks` probe had become dead code: on K3 it warned "adapter will run in naive (full-stack) mode" on every run while delta mode was in fact on (`_delta_mode = layout_tables is not None`). Removed on the review branch (`7dda3b847`).
- The transport switch became the model-config field `attn_res_cache` instead of an environment variable, because a launcher exporting the variable non-uniformly gave ranks different topologies and hung a collective. Tianyu's objection (it is PP infrastructure, not model architecture) is fair: on the review branch it is an argument of `pipeline_kimi_k3` and a recipe passes `functools.partial(pipeline_kimi_k3, attn_res_cache=False)` as the `pipelining_fn` (`7dda3b847`).
- `_inject_kimi_k3_fqns` filled `parallelism.module_fqns_per_model_part` so the generic split put the AttnRes aggregation modules on the last stage. Tianyu's reading ("build the split K3 needs from the bottom up instead of injecting into the generic util") is now the code: `kimi_k3_module_fqns_per_model_part` is a pure function of the config, and the entry hands `pipeline_llm` a replaced parallelism config instead of mutating the user's (`7dda3b847`).
- The even-split gate is gone (`ca5f34ea8`): the global layer-to-stage map is one `all_gather_object` over the pipeline group of each rank's stages, validated for completeness and contiguity, and `BlockLayoutTables` builds the routing from it, so `first/last_stage_less_layers` and 93 layers over any stage count are supported. Validation runs are in 3.3.

### 2.6 What the validation history established

- 2026-04-21, PP4 x VP2, 175M, 1000 steps: adapter loss inside the naive-vs-naive band; memory back to the naive baseline plus the cache.
- 2026-07-22 reproduction on the pinned commit: adapter-vs-naive |dLoss| at or below the naive-vs-naive band on PP8 x VP2, PP4 x VP2, PP4 x VP4, 1000 steps each; per-hop bytes constant after the warm-up hops.
- 2026-09-02 (the PR body): the pp x vp matrix on the 32-layer K3 debug flavor, 2 to 32 stages, step 1 bit-identical to dp1 in every cell; the fp32 grad-norm appendix, discussed in 3.3.

---

## 3. Answers to the review

### 3.1 "Too many install / patch / hook / thread"

The inventory, what each does, and what would replace it in first-principles infrastructure:

| mechanism | what it does | why it exists | the infrastructure version |
|---|---|---|---|
| `_install_mb_index_patch` (wraps `forward_one_chunk` / `backward_one_chunk`, thread-local) | gives the adapter the schedule's micro-batch id | `PipelineStage` does not pass the chunk id to the submodule | a stage API that exposes the current chunk id to the submodule |
| `_install_step_drop_patch` (wraps `pp_schedule.step`) | sweeps the rank cache and asserts the gradient slots drained at step end | no schedule lifecycle hook for "all micro-batches done" | a schedule callback at step end. Since `d72faf339` the blocks themselves are released per micro-batch from the adapter's own forward, so this wrapper is the safety net rather than the eviction path |
| `_install_augment_hook` (tensor grad hook) + `_LocalCacheCapture` | the same-rank gradient bridge (channel B) | a stage output consumed again by a later stage on the same rank; every autograd-only variant double-traversed the producer graph (2.3) | if the schedule knew that stage `S` and `S+P` share a rank, it could deliver the activation by alias and merge the gradient itself, which is the "shared activation" edge below |
| `_keepalive_touch` | keeps the received carrier on the autograd graph of the payload a stage sends on | a stage that consumes no block from a recv tensor would leave that tensor outside the graph and the schedule's backward P2P would have nothing to send | a stage graph with explicit edges would know which recv tensors need a gradient |
| `_forward_shape_inference` | reshapes the placeholder to the delta size during `PipelineStage._shape_inference` | shape inference calls the submodule directly, bypassing the chunk-id wrapper | stage shape inference driven by declared output shapes |
| `kimi_k3_module_fqns_per_model_part` (was `_inject_kimi_k3_fqns`) | the module-to-stage split, now a pure function of the config | the generic split does not know the AttnRes aggregation modules or the tower | this is the bottom-up split; done on the review branch |
| the stale `_return_only_new_blocks` warning | nothing (dead) | left over from the experiment model | deleted on the review branch |

None of these is speculative unblocking in the sense of "we did not know how to do it properly"; each is the narrowest way to reach schedule state that `torch.distributed.pipelining` does not expose. The honest summary is: the adapter implements a **non-linear stage graph** (stage `S` produces a tensor that stages `S+1 ... S+P-1` receive through the chain and stages `S+P, S+2P, ...` read from local memory) on top of a library whose stages only know their two neighbours. Every wrapper is the seam between those two models.

### 3.2 "If skip connections across stages become common, build general infra in `torch.distributed.pipelining`"

Agreed, and the notes above say what that infra needs, in order of value:

1. **Stage outputs with multiple consumers.** Today `PipelineStage.act_send_info` / `args_recv_info` describe a chain. A block committed at stage `S` is consumed by every later stage. The chain-relay the adapter does (send the block along, cache it on the way) is one implementation of a multi-consumer edge; the library could own the routing table (`BlockLayoutTables` is that table, computed outside the library).
2. **Rank-local delivery for same-rank consumers.** Under interleaved schedules the consumer of an activation is often on the producer's rank. The library knows `stage_index_to_group_rank`; a stage output whose consumer shares the rank should be delivered by reference and its gradient merged by the schedule, which is exactly channel B done without a hook.
3. **Chunk-id and lifecycle visibility for submodules** (a context object, and callbacks at micro-batch and step end), which removes the two wrappers outright and enables per-micro-batch release.

What torch has today for reuse (the "PP already has caching" remark, 3.5) is not this.

### 3.3 Numerics

**The mechanism behind the step-10 spread.** The PR body's appendix "Numerical Correction run with unmerged upstream grad-norm precision forced to FP32" already carries the key observation: with the total gradient norm accumulated in float32 (torchtitan PR 4135, pending the upstream fix in pytorch PR 194033), the six virtual-stage cells with the transport off collapse pairwise, pp4 x vp4 and pp8 x vp4 landing on 7.27054 at step 3 and staying together through step 10. Without the fix, `torch.nn.utils.get_total_norm` computes the norm of per-tensor norms in the gradients' dtype; PP puts different parameter subsets on different ranks, so the bf16 partial sums are grouped differently per topology, the clip coefficient differs at the bf16 rounding level, and ten steps of a 24/32-layer debug model at this learning rate amplify that into the few-percent spread in the tables. The direction is not monotone in the PP degree (second table: dp1 3.62, pp8 3.42; first table: dp1 3.42, pp8 3.63), which is what a rounding-seeded divergence looks like and what a systematic bias does not. The same global batch is used in every cell (8192 tokens per step under mx3's `BATCH`, one seed checkpoint), and step 1 is identical across cells, which rules out a batch-size mismatch.

**What the fix does not remove** is the difference between the delta transport and the fallback on the same topology: the delta transport sums the block gradients arriving from several consumers in a different order (channel B adds the slot into the incoming gradient; the fallback lets `SEND_B` merge them along the chain), so the two are equal up to bf16 summation order and diverge after the first update. The right evidence for "equal up to rounding" is a gradient comparison at step 1, before anything is amplified, which is what the PR should have led with. Results from the probe runs on this box (K3 24-layer debug flavor, seed checkpoint, `--debug.deterministic`, per-parameter gradients hashed and fp32-normed before clipping, 8192 tokens per step in 8 micro-batches of 1024 for the PP cells):

| comparison (step 1) | loss / grad_norm | parameters with a different gradient (of 750) | relative difference of the per-parameter fp32 norm: median / p90 / max | where the max sits |
|---|---|---|---|---|
| dp1 vs pp2, fallback transport (`1F1B`, 12+12 layers) | identical, 12.59997 / 13.7500 vs 13.6875 (bf16 print of the total norm) | 748 | 1.3e-4 / 9.0e-4 / 8.6e-3 | `delta_attention.A_log` (16 values, norm 1e-4) |
| pp2 x vp2 transport off vs on (same topology, `Interleaved1F1B`, 4 stages of 6) | identical, 12.59997 / 13.6875 | 748 | 1.6e-4 / 1.3e-3 / 1.2e-2 | `delta_attention.A_log` |
| dp1 vs pp2 x vp2 delta transport | identical, 12.59997 / 13.6875 | 729 | 1.2e-4 / 1.0e-3 / 1.3e-2 | `delta_attention.A_log` |
| pp2 x vp2 even, review branch `ca5f34ea8` vs the PR head, same warm cache | identical | 0 | 0 / 0 / 0 | the four review fixes are numerically inert |
| uneven split (7 layers per stage, first/last stage one less: stages of 6 / 7 / 6 / 5 layers, block boundary inside stage 1), delta transport vs dp1 | identical, 12.59997 / 13.7500 | 729 | 9.7e-5 / 9.3e-4 / 2.3e-2 | `delta_attention.A_log` |
| the same uneven split, transport off vs on | identical | 748 | 1.3e-4 / 1.1e-3 / 2.0e-2 | `delta_attention.A_log` |
| the same uneven split, transport off vs dp1 | identical | 748 | 1.2e-4 / 8.8e-4 / 1.4e-2 | `delta_attention.A_log` |
| dp1, `d72faf339` (cat-at-the-start refactor + per-micro-batch release) vs `ca5f34ea8`, same warm cache | identical | 0 | 0 / 0 / 0 | the refactor is bit-exact |
| pp2 x vp2 delta transport, `d72faf339` vs `ca5f34ea8`, same warm cache | identical | 0 | 0 / 0 / 0 | the per-micro-batch release changes no gradient |
| pp2 x vp4 delta transport (8 stages of 3 layers), `d72faf339` vs the PR head `087c4d177`, same warm cache | identical | 0 | 0 / 0 / 0 | the whole review branch against the PR head, bit-exact; the same pair from two cold caches while other jobs ran differed in 694 parameters, the autotune effect of 4.2 |

**The pp x vp matrix on the irregular debug model** (`395fc6b30`: 30 layers, block size 12, MLA at every fourth layer and the last; `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 4096 tokens per step in micro-batches of 256, 8 pipeline micro-batches; `first/last_stage_less_layers` at their default 1, so every split is uneven; each cell run twice on an otherwise idle box and the second run read):

| cell | stages | ranks | layers per stage | transport | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | - | 12.44394 | 7.32431 | 3.44458 |
| pp2 | 2 | 2 | 15 / 15 | fallback (`1F1B`) | 12.44394 | 7.54290 | 3.40260 |
| pp4 | 4 | 4 | 7 / 8 / 8 / 7 | fallback | 12.44394 | 7.52203 | 3.37749 |
| pp8 | 8 | 8 | 3 / 4 ... 4 / 3 | fallback | 12.44394 | 7.47274 | 3.43359 |
| pp2 x vp2 | 4 | 2 | 7 / 8 / 8 / 7 | delta | 12.44394 | 7.38716 | 3.72131 |
| pp2 x vp2 | 4 | 2 | 7 / 8 / 8 / 7 | off (whole carrier) | 12.44394 | 7.47149 | 3.68055 |
| pp2 x vp4 | 8 | 2 | 3 / 4 ... 4 / 3 | delta | 12.44394 | 7.40650 | 3.46752 |
| pp4 x vp2 | 8 | 4 | 3 / 4 ... 4 / 3 | delta | 12.44394 | 7.49078 | 3.47512 |
| pp4 x vp4 | 16 | 4 | 1 / 2 ... 2 / 1 | delta | 12.44394 | 7.42482 | 3.40743 |
| pp4 x vp4 | 16 | 4 | 1 / 2 ... 2 / 1 | off | 12.44394 | 7.45038 | 3.39832 |
| pp8 x vp2 | 16 | 8 | 1 / 2 ... 2 / 1 | delta | 12.44394 | 7.38036 | 3.38767 |
| pp8 x vp4 (after `c3df74847`) | 32 | 8 | 0 / 1 ... 1 / 0 (embedding-only and head-only stages) | delta | 12.44394 | 7.29935 | 3.29156 |
| pp8 x vp4 (after `c3df74847`) | 32 | 8 | 0 / 1 ... 1 / 0 | off | 12.44394 | 7.45038 | (3 steps, on the delta cell's compile cache) |

The transport-off 32-stage cell is the one row that needed a second look: its matrix run, whose eight ranks autotuned their FlexAttention kernels at the same time, came out at 12.45856 at step 1; rerun on the delta cell's warm caches it is 12.44394 like every other cell. That is the compile lottery of 4.2 acting on step 1 itself (a cold compile under load can pick a kernel with different rounding), and it is why the matrix protocol reads the second run of a cell, and why no step-1 mismatch should be read as a code difference before a same-cache rerun.

Step 1 is bit-identical to dp1 in every cell that ran, uneven stages and the delta transport included; the later steps spread as 3.3 explains (bf16 total-norm grouping and the compile lottery), in both directions.

**What the 32-stage cell found: a `torch.distributed.pipelining` bug, and a K3-side boundary for it (`c3df74847`).** With one layer per stage the first stage holds only the embedding and the last only the head, and both transports died at the first backward receive with "Tensors for P2P must be non-overlapping and dense" (reproduced on 2 GPUs as pp2 x vp16, transport off). The mechanism: `PipelineStage._create_grad_recv_info` allocates a stage's gradient receive buffer with `torch.empty_strided` from the *strides* of the next stage's input gradients, which `_backward_metadata_inference` computes once with `torch.autograd.grad`. A stage whose first use of an input is a concatenation gets, as that input's gradient, a view of the concatenation's gradient (the recorded metas for the head-only stage were `(256, 1024)` with stride `(4096, 1)` and `(256, 3, 1024)` with stride `(4096, 1024, 1)`: two slices of one `[T, 4, D]` buffer), and c10d refuses a buffer built with those strides. It surfaces only when nothing else in the stage consumes the input (a later layer's use makes autograd accumulate a dense gradient), which is why every other cell in the matrix, and the 24-layer probes, passed. The fix on the review branch is model-side and forward-inert: a stage passes its inputs through an identity whose backward returns a contiguous gradient (`_DenseGradient`), so the metas and the buffers are dense; values are unchanged (same-cache gradient hashes in the table above). The torch-side fix is to allocate the receive buffer dense (`torch.empty(shape)`) and to send `.contiguous()` gradients, which is worth an upstream issue: any model whose stage begins with `cat`, `stack` or a slice of its input hits it.

Memory at this scale does not move with the per-micro-batch release: pp2 x vp2 reports 7.37 GiB reserved after step 1 and 9.00 after step 2 before and after, pp2 x vp4 7.51 and 9.14 on both codes. A cached block of the debug model is 256 tokens x 1024 x 2 bytes = 0.5 MB, so the whole rank cache is a few MB against activations of GiB; the saving the release buys is `blocks x T x D x 2 bytes` per micro-batch no longer resident, which is the 04-21 envelope's `M` term at production shapes (several GB at 48B, T = 8192), and it will need a measurement at that shape, not this one.

Reading: the delta transport is as far from the fallback as the fallback is from a single GPU, the step-1 loss is bit-identical in every cell, and the distribution of per-parameter differences is the one bf16 summation order produces (a median of 1e-4 with a tail on 16-element parameters whose norm is 1e-4). No module kind stands out: the block-residual projections and norms, which are the tensors the transport touches, sit at 4e-3 to 8e-3 in every comparison including the one with no transport at all. A systematic error in the gradient routing would show as a parameter group whose difference is orders of magnitude above the rest; there is none.

Two caveats from the probe itself. A PP cell that compiled while another job shared its GPUs came out with a different step-1 loss (12.60011); FlexAttention's autotune picks kernels by benchmark timing, which `--debug.deterministic` does not control, and this is what the PR body's "run each cell twice" rule was covering. And the gradient runs above use 32 micro-batches of 256 tokens in every cell (PP: 4 accumulation rounds of 8 pipeline micro-batches); a dp1 run with 8 micro-batches of 1024 tokens packs the data differently and is not comparable, which is also why "same global batch" alone does not make two cells comparable.

The remaining review question, whether a systematic error hides under the rounding, is answered by the same-topology transport A/B: the maximum relative difference per parameter is the size of one bf16 rounding of a sum, and no parameter group stands out.

### 3.4 The individual comments

| comment | answer |
|---|---|
| `model.py:33` why change the comment | reverted (`e326c70a2`). |
| `model.py:228` `first_layer_in_block` | applied as suggested (`e326c70a2`). |
| `model.py:228` why no `// 2` as in the paper | the paper's pseudo-code counts sub-layers (`self.layer_number % (self.block_size // 2)`, with the comment "block_size counts ATTN + MLP; each transformer layer has 2"). Our `attn_res_block_size=12` counts transformer layers, which is the paper's 24, so the `// 2` is already folded into the constant. |
| `model.py:236` cat at the start of the layer, then `_apply_attention_residual` with `prefix_sum` None | done (`eef340d25`): the block's first layer appends the incoming stream to the stack at the top of the layer, both residuals read "the stack, plus the open block's partial sum when there is one", and `_apply_attention_residual` takes that partial as Optional. The values every softmax sees are the same tensors in the same order, so the forward is unchanged bit for bit (dp1 and pp2 x vp2 gradient hashes on a shared compile cache, 3.3). |
| `model.py:272` numerics | see 3.3. |
| `model.py:274` why "even split" matters | it does not, in principle: `BlockLayoutTables` takes an explicit layer-to-stage map and the cache holds whatever blocks arrive. The gate existed because `infer_block_layout_tables_from_stages` only saw the local rank's stages and validated the equal-split default instead of exchanging the real ranges. Fixed on the review branch (`ca5f34ea8`): one all-gather of each rank's layer-to-stage entries on the PP group at setup; `first/last_stage_less_layers` and 93 layers over any stage count are supported; the gate is gone. |
| `model.py:275` `attn_res_cache` is a PP-infra flag | agreed. It moved from an environment variable to the model config to guarantee every rank resolves the same topology. Now an argument of the pipelining function, `pipeline_kimi_k3(model, ..., attn_res_cache=...)`, chosen by the recipe through `functools.partial`; the model config carries only architecture (`7dda3b847`). |
| `model.py:212` storage life cycle of the block tensors at PP=1 | the carrier is not re-saved per layer: between block boundaries the same `[T, N, D]` tensor object passes through every layer, so autograd saves references, not copies. At each block boundary `torch.cat` materialises a new `[T, k, D]` tensor (k = blocks so far), and those are what stay alive until backward: `N(N+1)/2` block-rows instead of `N` (36 vs 8 for 93 layers with `attn_res_block_size=12`). Under `SelectiveAC` the `_apply_attention_residual` ops (cat, float, rsqrt, softmax, bmm) are recomputed, only linear/mm/flex outputs are saved, so the AttnRes computation itself adds no saved activation, as the report says; the quadratic part is the cat copies. A list-of-blocks carrier (stack only at use) or a preallocated `[T, N, D]` buffer written in place would bring it to `N`. Not in this PR; noted as a follow-up with a measured number to attach. |
| `__init__.py:460/461/513/514` debugmodel defaults, irregular shape, block size 12, deducible `full_attention_layers` | done (`395fc6b30`): one debug model, `_debugmodel(attn_backend, *, num_layers)` with no defaults, the MLA layers from `kimi_k3_full_attention_layers` (every fourth layer and the last, the rule that reproduces the 93-layer model's `range(3, 92, 4) \| {92}`, now used by both), `attn_res_block_size` the model's 12. Depth 30: the last block is partial (12 + 12 + 6, as the full model's 7 x 12 + 9), the stack ends on a lone MLA layer after seven (3 KDA + 1 MLA) groups, and with the embedding and the head counted as a layer each the 32 units divide into every pipeline shape up to 32 stages with uneven stages (the first and last hold one layer fewer, and at one layer per stage none at all). The 32-layer flavor is gone; the whole pp x vp matrix is rerun on this model (3.3). |
| `features.py:314` one integration cell is enough | the pp8 x vp4 cell is removed (`e326c70a2`); pp2 stays. |
| `pipeline_adapter.py:130` how to divide 93 layers evenly | you cannot, and you should not have to; see the even-split answer. |
| `pipeline_adapter.py:140` "pytorch PP already has caching" | see 3.5. |
| `pipeline_adapter.py:140` only-incremental transfer, and release as soon as the micro-batch finishes | incremental transfer: yes, the delta is exactly `accumulated - receiver_cache` and the wire carries nothing else (per-hop sizes in the 04-21 table). Release per micro-batch: it was not (entries were dropped at step end, 2.4); done on the review branch (`d72faf339`). No forward on a rank reads the cache for a micro-batch after the rank's last virtual stage has run, and the backward never reads it (across ranks the gradient rides the autograd graph and the schedule's backward P2P, on a rank the captured-grad slots), so the adapter releases the blocks right after that forward and keeps only the slots until the producer's backward pops them; the step-end sweep stays as the safety net. It needed no new wrapper: the release point is the adapter's own forward. Gradients are unchanged bit for bit, memory numbers in 3.3. |
| `pipeline_adapter.py:1030` "inject" | agreed; replace `_inject_kimi_k3_fqns` with a K3 split function that builds `module_fqns_per_model_part` from the config and hands it to `pipeline_llm`, no kwargs patching. |
| `pipeline_adapter.py:1175` private call across files | made public (`e326c70a2`). |
| `fsdp.py:168` `add_zero_valued_dependency` (on the CP PR) | the suggestion is separate llm/vlm variants with the vlm assuming multimodal input is present. That covers "the batch has no images"; the case the helper guards is a CP sequence shard with zero image placeholders while the batch has images, which the vlm variant does not remove. We will restructure to the variants and keep the shard-local guard only where the shard can be empty. |

### 3.5 Can torch's PP cache be reused?

`torch.distributed.pipelining.PipelineStage` (torch 2.14.0.dev20260802) keeps, per stage, `fwd_cache: dict[chunk_id, (outputs, flattened inputs)]`, written at the end of `forward_one_chunk` and popped by `backward_one_chunk` to run that stage's own `stage_backward`; `bwd_cache` holds the input gradients until they are sent; `output_chunks` collects the last stage's outputs. That is the bookkeeping a stage needs for its own backward: it holds the stage's inputs and outputs for one micro-batch, on that stage, until that stage's backward. It does not keep received activations alive for a *later* stage on the same rank, it has no notion of a stage output consumed by a non-adjacent stage, and it routes gradients only to the adjacent stage. So it cannot replace `RankLocalCache`. What the adapter does share with it is the key: both are indexed by the schedule's chunk id, which is why the chunk-id wrapper exists. The reuse worth having is the other way round: an extension of `fwd_cache` semantics to multi-consumer outputs, which is 3.2.

### 3.6 The talk

Yes, we are happy to walk through the design; this document is the written version.

---

## 4. The CP PR (4313)

### 4.1 Ulysses and KCP now own their collectives (`f66b5de3a`)

Following PR 4322's shape: `ContextParallelKernel` mixin (copied, marked for deletion once 4322 lands), `UlyssesCPFlexAttention` whose forward issues the two `spmd.redistribute` all-to-alls around FlexAttention, `ContextParallelInnerKDA` which runs Attention Gym's KCP recipe with the group from the SPMD mesh, and `use_kimi_k3_cp_kernels` choosing kernels by layer kind at config time. The inner-attention boundary is an identity on the cp axis; `apply_cp_kimi_k3` only wires the vision splice.

### 4.2 Numerics of the restructure

The restructure moves the same two `spmd.redistribute` calls from the module boundary into the kernel's forward, so it should be numerically inert, and it is: with one warm inductor cache shared by both trees, every one of the 750 parameter gradients is sha1-identical between the kernel tree and the declarative tree at step 1 and at step 2 (cp2, seed checkpoint, `--debug.deterministic`, 8192 tokens per step). Four more runs from fresh caches with the GPUs otherwise idle, two per tree, are sha1-identical to each other and across the trees at both steps (loss 9.54240 at step 2 in all of them).

What first looked like a difference was not one. The 10-step matrix cells disagreed from step 2 (kernel tree 9.53739 / 9.53178 / 9.54240 across cache states; declarative tree 9.54240 every time) while the step-1 loss and total norm were identical. Every run that produced one of the two odd values had compiled while another matrix was running on the other GPUs of the box; every run that compiled alone produced 9.54240. The step-2 value depends on which FlexAttention kernel inductor's `max_autotune` picked, which is decided by benchmark timing, is outside `--debug.deterministic`, and is perturbed by load on the machine. Same-cache runs reproduce each other bitwise (three runs at 9.53178, two at 9.54240). The consequence for every table in these PRs: a step-10 loss on this box measures the autotune lottery as much as the code, and only step-1 gradients (or same-cache reruns) can support a "numerically unchanged" claim. The matrix rows below are kept for completeness, not as evidence of a difference:

| cell (cp2, spmd_types) | step 1 | step 3 | step 10 |
|---|---|---|---|
| declarative Ulysses (`84fd3ee22`), warm and cold cache alike | 12.53972 | 7.18619 | 3.00631 |
| kernel-owned Ulysses (`f66b5de3a`), one cache state | 12.53972 | 7.19535 | 3.04564 |
| tp2 x cp2 (no SP), declarative / kernel | 12.55243 / 12.55243 | 7.21316 / 7.22327 | 2.92873 / 3.02106 |
| dp2 x cp2, kernel | 12.52908 | 7.30424 | 3.16331 |

### 4.3 All-gather KV for MLA (`adc012ce4`)

`AllGatherCPFlexAttention` is copied from 4322 as the second MLA kernel: q stays token-sharded, k/v are gathered with `flex_cp_allgather`, the masks are cut along Q like every other model's. Selected by a recipe (`kimi_k3_debugmodel_cp2_allgather`) before the model's `update_from_config`, which keeps a kernel already chosen and defaults to Ulysses otherwise. The reason Ulysses stays the default for MLA: gathering the expanded K/V costs `cp` times the all-to-all volume, and the cheap alternative, gathering the compressed latent, needs the collective before `wkv_b`, outside the inner kernel where 4322 wants collectives to live.

Validation on cp2 (same seed checkpoint and batch as above): step-1 loss 12.53972 and total norm 14.0000, identical to Ulysses; per-parameter gradient norms against the Ulysses run differ with median 1.5e-4, p90 1.3e-3, max 1.5e-2 relative, the same distribution as the PP fallback-vs-delta comparison, i.e. two kernels tiling the same attention differently. The KDA layers are unchanged by the choice (they run KCP either way).

### 4.4 Other items on the CP branch

- The plain-tensor vision splice now rejects CP with sequence parallel like the DTensor branch does (`3970bcd1c`); found because `enable_sequence_parallel` defaults to True and a tp2 x cp2 cell without `--parallelism.no-enable-sequence-parallel` silently runs SP.
- `--debug.spmd_typechecking` under CP hits two pre-existing problems unrelated to the kernels: the residual-stack seed's `R -> I` redistribute over a size-1 tp group trips an `UnboundLocalError` inside spmd_types' checker, and at tp2 x cp2 the checker has no strategy for the block-residual `cat` on the cp axis. Both need attention before the CP branch can claim a typecheck-clean run; neither is touched by this PR's change.
- The declarative spmd changes (`4b88ada6b`, the K3 declarations spmd_types reads that partial_dtensor never did) are already merged into `cp_review2`; the CP PR will carry them once the stack is rebased after EP merges.
