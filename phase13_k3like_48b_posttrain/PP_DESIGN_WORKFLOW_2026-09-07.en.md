# Pipeline parallelism for Block Attention Residuals: requirements, the implementation and its bugs, what `torch.distributed.pipelining` lacks (2026-09-07)

English, condensed version of `PP_DESIGN_WORKFLOW_2026-09-07.md`. Sources: `REVIEW_ANSWERS_PP_CP_2026-09-04.md`, `PP_VP_REEXAMINATION_2026-07-30.md`, `PP_ATTNRES_ADAPTER.md`, `phase3_attnres_pp_integration/handoff_status_20260421.md`, `PR_BODY_PP.md`, `pipeline_stage.py` / `layout.py` on `k3_int_20260906`.

## 0. One page

```
Block AttnRes: every layer attends over ALL earlier blocks + the running partial block
    |
    v  split by layers into pipeline stages
R1  the block stack must cross every stage boundary with the hidden state
R2  the final aggregation (output_res_proj -> output_res_norm) runs only on the stage that owns lm_head
R3  the stack grows with depth; a boundary inside a block puts a partial block on the wire
    |
    +-- naive transport: send the whole stack every hop -> correct, bytes grow with the stage index
    +-- cross-stage cache (report 4.1): V virtual stages per rank; a hop carries only the delta the receiver lacks
            |
            v  three new dependencies
        (a) sender and receiver must agree on what a hop carries  -> routing tables (BlockLayoutTables), no metadata on the wire
        (b) the cache is per micro-batch                          -> a key that survives P2P (the schedule's chunk id)
        (c) a cached block's gradient must return from every later stage to the stage that committed it -> two gradient routes
            |
            v
        torch's PipelineStage knows one protocol: outputs to the next stage, gradients from it -> section 3
```

## 1. What PP / VP demand of the model and the micro-batches

| requirement | why | where |
| --- | --- | --- |
| stack crosses every boundary | each layer attends over all earlier blocks | hop payload `(hidden_TD, delta_TND)`; the model takes and returns the whole stack and knows nothing of transport |
| aggregation only on the head stage | one aggregation over the whole stack | `kimi_k3_module_fqns_per_model_part` keeps the res modules with `lm_head` |
| partial blocks on the wire | a block's first layer joins the stack before it attends | `first_layer_in_block`; a boundary at a block start needs nothing special |

No divisibility between block size and stages: K3 is 93 layers = 7 x 12 + 9; the debug flavor uses 33 layers so no pipeline shape divides; routing is built from the actual split (one all-gather of layer -> stage).

Two transports: naive (whole stack per hop; correct, no adapter, what plain 1F1B runs) and the cached delta (a block committed at stage $S$ is fresh on the wire for $P-1$ hops, then every receiving rank already holds it from its virtual stage $S-P$; per-hop payload bounded by the last $P-1$ stages' commits, independent of depth). Same routing table, one flag (`attn_res_cache`); the two are the A/B of the results tables.

Micro-batch dependencies: the cache key must be the schedule's integer chunk id (`id(tensor)` fails: NCCL hands out fresh receive buffers); blocks are released after the rank's LAST virtual stage forward for that micro-batch; `Interleaved1F1B` needs micro-batches >= virtual stages and virtual stages divisible by `pp`; PP is bitwise with no-PP at ONE micro-batch and differs by a bf16 accumulation-order margin at many (step 1: ~4e-4 on the loss, 1.3% on grad_norm), while VP adds nothing (VP=1 and VP=4 bitwise).

![Dependencies: the stack grows across stages, partial blocks ride the wire, only the head stage aggregates](figures/pp_attnres_dependencies.svg)

Figure 1. 12 layers, blocks of 4, 4 stages: the payload per hop, and the two transports.

## 2. From Reku's note to our implementation, and the bugs on the way

**Reku's public note (Zhihu, Kimi infra):** add an adapter after the pipeline communication that concatenates the received block with the cached ones; the backward receives all blocks' gradients, accumulates them in the adapter and sends the buffer on; under interleaved scheduling the send/recv hides in steady state; the cache changes the gradient accumulation order, so precision alignment gets harder when the PP config changes. The note is about wire bytes; for very deep models it recommends selective AC plus activation offload, not a distributed cache.

**Phase 3 adapter (2026-04), four missing pieces:** (1) who holds which blocks, without sending metadata -> `BlockLayoutTables`, both sides simulate one micro-batch offline; (2) a micro-batch key that survives P2P -> the chunk id stashed in a thread-local by wrapping `forward_one_chunk` / `backward_one_chunk` (the "patch / thread" the reviewer flagged; root cause: `PipelineStage` does not hand the chunk id to the submodule); (3) the backward, below; (4) eviction after `step()` from the rank's last virtual stage (the second wrapper).

**The backward: six attempts, one survivor.**

1. custom NCCL inside `autograd.Function.backward` -> deadlock (the engine is single-threaded and depth-first; the peer has not reached its matching Function; races with the schedule's own `SEND_B` / `RECV_B`).
2. flush block gradients after `backward_one_chunk` -> deadlock (interleaved ranks reach a micro-batch's backward at different times; P2P ops with no peer).
3. one batched exchange at step end -> keeps every micro-batch's graph alive until step end; rejected.
4. ride PP's own `SEND_B`: received blocks stay attached to the tensor they arrived in, gradients flow hop by hop with no custom collective -> **route A**. Fails for one case: a block a rank committed at virtual stage `v` and reads back from its own cache at `v+1`; the consumer's backward walked into the producer's graph and freed it: "backward through the graph a second time".
5. `retain_graph=True` -> correct, +5 GiB on rank 7 of a 175M model, grows with V and the micro-batch count; stop-gap.
6. `_LocalCacheAugment` + `_LocalCacheCapture` Functions -> still double-backward on 4 GPUs (per-Function tracing: both fired in one `backward_one_chunk`).
7. cache a DETACHED copy + Capture + a grad hook on the producer's block -> **route B**: the consumer's input has no upstream graph; Capture deposits the gradient in a slot keyed `(mb, producer stage, commit index)`; the hook adds it during the producer's own backward. Memory back to naive PP plus the cache (7.71 vs 7.45 GiB, PP4 x VP2, 175M); a static count of expected captures refuses the step on a lost gradient.

The two gradient routes: across ranks, PP's own backward P2P (no new code); on a rank, a slot. The cache stores VALUES (detached copies), never activations or graphs; every forward graph is traversed once per micro-batch. Detach is load-bearing: a `view` and a Function returning `None` were not enough (attempts 4 and 6).

![Two gradient routes: the delta window and routes A / B with their counts](../Raising_PRs/PR_K3_PARALLELISM/pp_dual_gradient_bridge_v2.svg)

Figure 2. The delta window (a block is fresh on the wire for $P-1$ hops) and the two backward routes; labels updated to the subclass version (`pp_dual_gradient_bridge.svg` kept).

**The K3 port (August) and the subclass (September, review branch).** Layout from the model config; the transport switch as a `functools.partial` argument of `pipeline_kimi_k3` instead of an environment variable (a non-uniform export once gave ranks different topologies and hung); the split built from the config; the even-split gate replaced by one `all_gather_object` of layer -> stage. `AttnResPipelineStage(PipelineStage)` (388 lines, replacing the 1228-line adapter): `forward_one_chunk` assembles the stack from the rank's `RankStore` plus the received delta, runs the stage, keeps its commits, sends only what the next rank lacks; `backward_one_chunk` reads the gradient of the assembled stack (a leaf the stage owns), returns the received columns as the delta's gradient and DEPOSITS the stored columns in the store; the stage that brought a block onto the rank collects the deposits in `_retrieve_recv_grads` before its own backward, which every schedule orders after the later stages' backward on the rank. No hook, no Function, no detach trick; the tables say how many deposits each block must have, so a lost gradient raises. One core hook: `pipeline_llm(..., stage_class=...)`. Step 1 bitwise with a single GPU on every pp x vp cell of the irregular debug model, 2 to 32 stages, both transports.

![P=2 x V=2 walk-through, subclass labels](../phase3_attnres_pp_integration/pp_adapter_flow_v2.svg)

Figure 3. Receive / assemble / emit per stage and the two ranks' stores; relabelled from phase 3's `pp_adapter_flow.svg` (kept).

![One rank's interleaved timeline: the store's put / read / release and deposit / collect](figures/pp_rank_timeline.svg)

Figure 4. One rank, virtual stages v0 / v1, micro-batch 0: put at v0's forward, read at v1's forward, release after the rank's last virtual stage forward; deposit at v1's backward, collect before v0's backward, ordered by the schedule.

**Bugs on the way, with causes.**

| symptom | cause | fix |
| --- | --- | --- |
| every middle stage missed its cache | `id(tensor)` as the key; NCCL fresh buffers | the schedule's chunk id |
| NCCL watchdog timeout | custom P2P inside autograd / in a flush | ride the schedule's `SEND_B` (route A) |
| "backward through the graph a second time" | a same-rank cached block still attached to the producer's graph | detached copies; later the store's deposit / collect |
| +5 GiB on rank 7 | `retain_graph=True` stop-gap | route B |
| ranks with different topologies hang | env-var transport switch, non-uniform export | `functools.partial` argument |
| even split only | interleaved ranks lack the global layer -> stage map | one `all_gather_object` |
| `Tensors for P2P must be non-overlapping and dense` | torch sizes the backward receive buffer from the next stage's input-gradient strides; a stage starting with cat / stack / slice yields view gradients | model-side `_DenseGradient` (identity, `.contiguous()` backward); upstream should allocate `torch.empty(shape)` and send `.contiguous()` |
| "no gradient arrived for the payload" under LoRA (`4716f8ee6`) | nothing trainable upstream of the payload (frozen embedding feeds block 0); the next stage has no gradient channel | `_retrieve_recv_grads` accepts `None`; deposits discarded |
| `KeyError: 0` in verl's forward-only log-prob pass (`cea5c4f0f`) | `schedule.eval` calls `backward_one_chunk` with backward off; the base returns, the override read the cache it never filled | honour `has_backward`, drop the chunk's bookkeeping |
| `KeyError` in the HF initial load on a stage (`7d3ba3553`) | the adapter shaped layer-0 placeholders from layer 1, absent on that stage | place only on the rank holding layer 0, shape from the config |
| `'FSDPParam' has no attribute '_unsharded_param'` on a text batch under PP (2026-09-07) | FSDP2's root final callback runs `post_backward` for every group, and the no-reduce branch (PP's non-last micro-batches) lacks the `hasattr` guard the reduce branch has; a never-forwarded FSDP unit (the vision tower on text) has no `_unsharded_param` | a text-only model has no tower (the text alias sets `vision_encoder=None`); the torch guard is an upstream issue, not a parallelism change |
| step 10 spreads by percents | Adam's first step is `lr * sign(g)`; bf16 rounding flips 0.2% of elements and the descent amplifies it; fp32 total norm does not remove it | step 1 and per-parameter gradients are the bar |

## 3. What `torch.distributed.pipelining` lacks, and the suggestions

`PipelineStage` describes a chain (`act_send_info` / `args_recv_info`: outputs to the next stage, gradients from it); its `fwd_cache` / `bwd_cache` hold one stage's own inputs and outputs for one micro-batch, for its own backward. Nothing keeps a received activation for a later stage on the same rank; nothing routes a gradient to a non-adjacent producer. AttnRes needs a multi-consumer edge. In order of value:

1. **Multi-consumer stage outputs.** The library owns the routing table (`BlockLayoutTables` is that table, computed outside it) and decides per hop what travels and what is kept; the chain relay with a rank store is one implementation, bounded to the last $P-1$ stages' commits.
2. **Rank-local delivery for same-rank consumers.** The schedule knows `stage_index_to_group_rank`; such an output can be handed over by reference and its gradient merged before the producer's backward, which the schedule already orders last on the rank (route B without a hook; the subclass proves the ordering is enough).
3. **Chunk id and lifecycle for the submodule.** A context with the current micro-batch plus callbacks at micro-batch end and step end; removes the two wrappers, enables per-micro-batch release.
4. **Dense backward P2P buffers.** `torch.empty(shape)` receive and `.contiguous()` before send; stride-sized buffers fail when a stage starts with a view op.
5. **Forward-only and no-gradient contracts.** `backward_one_chunk` under `has_backward=False` as part of the subclass contract; stage outputs with `requires_grad=False` get `None` rather than an assumed gradient.

![Protocol gap: torch's chain against the multi-consumer edge, the five suggestions in place](figures/pp_protocol_gap.svg)

Figure 5. Left: the chain protocol and its per-stage caches. Right: b0 committed at S0 read by S1 / S2 / S3, S2 on S0's rank. The five suggestions where each bites.

A paste-ready short form of section 3 is in `Raising_PRs/PR_K3_PARALLELISM/RFC_PIPELINING_MULTI_CONSUMER_DRAFT.md`.

## 4. Figures

Figures 1, 4, 5 are new (`figures/`). Figures 2 and 3 are relabelled copies of the two existing figures: the forward routing and the delta-window counts were still right; only the adapter-era backward labels (`_install_augment_hook`, `_LocalCacheCapture`, `_keepalive_touch`, `RankLocalCache`) became store deposit / `_retrieve_recv_grads` / `deposits_expected` / `RankStore`. The originals are kept next to them.
