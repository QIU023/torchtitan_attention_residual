# RFC (pytorch/pytorch issue): multi-consumer stage outputs in `torch.distributed.pipelining`

For `torch/distributed/pipelining` (owners: H-Huang, wconstab, sanketpurandare, fegin). File as an issue with the `oncall: distributed` and `module: pipelining` labels; the contiguous-wire PR (`Raising_PRs/PR_pytorch_pipelining_contiguous_wire`) is the first, mechanical piece and can go in independently. One line per paragraph.

--- PASTE BEGIN ---

### Summary

`PipelineStage` implements one protocol: a stage's outputs go to the next stage and its input gradients come back from it (`act_send_info` / `args_recv_info`; `fwd_cache` / `bwd_cache` hold one stage's own inputs and outputs for its own backward). A model whose layers read the outputs of ALL earlier layers -- Block Attention Residuals in Kimi K3 (a 2.8T open model; the report describes a cross-stage cache for exactly this), and any architecture with skip connections across the split -- needs a stage output with several consumers, some on other ranks and some on the producer's own rank under interleaved schedules, with a gradient that returns from each consumer. This RFC describes what a model-side implementation had to build outside the library, why the natural general mechanisms fail, and five additions that would let `torch.distributed.pipelining` carry such an edge; the first (dense P2P buffers) is a standalone fix.

### The requirement

![The stack grows across stages, partial blocks ride the wire, only the head stage aggregates](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_attnres_dependencies.png)

Every layer attends over the running stack of block outputs, so the stack crosses every stage boundary with the hidden state (R1), the final aggregation runs only on the head stage (R2), and a boundary inside a block puts a partial block on the wire (R3). The naive transport sends the whole stack every hop: correct, no library change, bytes grow with the stage index. The cached transport keeps a block committed at stage $S$ on the wire for $P-1$ hops, after which every receiving rank already holds it from its virtual stage $S-P$; the per-hop payload is then bounded by the last $P-1$ stages' commits. That transport needs three things the chain protocol does not have: sender and receiver agreeing on what a hop carries (a routing table, no metadata on the wire), a per-micro-batch cache keyed by something that survives P2P (tensor identity does not; the schedule's chunk id does), and the gradient of a cached block returning from every later stage to the stage that committed it.

### What a model-side implementation had to do, and what failed

![The looped stage grid: a micro-batch returns to the same rank every P stages](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_stage_grid.png)

The implementation (pytorch/torchtitan#4312, a `PipelineStage` subclass with a rank-local value store; step-1 bitwise against one GPU from 2 to 32 stages, PP8 x VP4) builds the routing table once from the actual layer-to-stage split and the schedule's stage-to-rank map, keys its store by the chunk id (`forward_one_chunk`'s argument), and routes gradients two ways: across ranks they ride the pipeline's own backward P2P; within a rank a consumer's backward deposits the gradient of a stored block in a slot and the producer's backward collects it before it runs, an ordering every schedule already provides (later virtual stages' backward first). Six general mechanisms failed first: a collective inside `autograd.Function.backward` deadlocks (single-threaded depth-first engine, the peer is elsewhere, and it races the schedule's own P2P); a per-micro-batch flush deadlocks under interleaving (ranks reach a micro-batch's backward at different times); a step-end exchange keeps every graph alive to step end; riding the schedule's P2P alone breaks for a block a rank commits at one virtual stage and reads at the next (the consumer's backward frees the producer's graph twice); `retain_graph=True` gives up the memory PP exists to save; autograd wrappers on the same-rank hand-off double-fire. The store holding VALUES (not graphs) plus the schedule's ordering is what works.

![Two gradient routes: the pipeline's own backward P2P across ranks, a store slot within a rank](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_dual_gradient_bridge_v2.png)

![One rank's interleaved timeline: put / read / release, deposit / collect](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_rank_timeline.png)

### Proposals

![torch's chain protocol against the multi-consumer edge, and where each proposal bites](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_protocol_gap.png)

1. Multi-consumer stage outputs: a stage output may name several consumer stages; the library owns the routing table (per stage: the outputs it commits, the ones its rank already holds, the ones a hop must carry) and decides per hop what travels and what is kept. The chain relay with a rank store is one implementation, bounded to the last $P-1$ stages' commits.
2. Rank-local delivery for same-rank consumers: the schedule knows `stage_index_to_group_rank`; an output whose consumer is on the producer's rank is handed over by reference, and its gradient is merged before the producer's backward, which the schedule already orders last on the rank. This is the piece that cannot live in a framework-level step-end hook.
3. Chunk id and lifecycle for the submodule: a context carrying the current micro-batch index, plus callbacks at micro-batch end (release) and step end. Frameworks are adding step-level runtime hooks (pytorch/torchtitan#4486); the micro-batch index and the micro-batch-end callback have to come from the schedule.
4. Dense P2P buffers: `_make_tensor_from_meta` allocates receive buffers with `torch.empty_strided` from the peer tensor's strides, so a stage whose forward starts with a view op (`cat` / `stack` / a slice of its input) hands back a strided input gradient and the batched P2P raises `Tensors must be contiguous`. Record contiguous strides in the metadata, allocate `torch.empty(shape)`, send `.contiguous()`. PR attached; the two-stage repro's pipelined gradient is bitwise the single-process one with it.
5. Forward-only and no-gradient contracts, stated: `backward_one_chunk` returns under `has_backward=False` (it does; subclasses must honour it, worth a sentence in the docs), and stage outputs with `requires_grad=False` receive `None` rather than an assumed gradient (a stage fed by a frozen embedding under LoRA).

### What we can bring

The routing tables from an arbitrary split, the value store with deposit slots and the expected-deposit count as the correctness check, the stage subclass as an executable spec of the ordering, an A/B against the naive transport at 2 to 32 stages on an irregular 33-layer model, and the willingness to rewrite the client on whatever shape the library settles on.

--- PASTE END ---
