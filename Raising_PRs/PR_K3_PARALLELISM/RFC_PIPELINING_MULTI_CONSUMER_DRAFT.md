# Draft comment for PR-4312: what `torch.distributed.pipelining` would need for a cross-stage residual

Paste as a comment on the PP PR when the maintainer's "build general infra" thread comes up again; one line per paragraph, English only. Source: `phase13_k3like_48b_posttrain/PP_DESIGN_WORKFLOW_2026-09-07.md` section 3.

--- PASTE BEGIN ---

The residual is a stage output with many consumers: a block committed at stage $S$ is read by every later stage. `PipelineStage` describes a chain (`act_send_info` / `args_recv_info`: outputs to the next stage, gradients from it), and its `fwd_cache` / `bwd_cache` hold one stage's own inputs and outputs for its own backward; nothing keeps a received activation for a later stage on the same rank, and nothing routes a gradient back to a non-adjacent producer. What the K3 stage subclass adds outside the library, in the order a generic version would want it:

1. A multi-consumer stage output. The library could own the routing table (`BlockLayoutTables` is that table: per stage, the blocks it commits, the blocks its rank already holds, the blocks its hop must carry) and decide per hop what travels and what is kept; the chain relay with a rank store is one implementation of that edge, bounded to the commits of the last $P-1$ stages.
2. Rank-local delivery for same-rank consumers. Under interleaved schedules the consumer of an output is often on the producer's rank; the schedule knows `stage_index_to_group_rank`, so such an output can be handed over by reference and its gradient merged by the schedule before the producer's backward, which every schedule already orders after the later stages' backward on the rank. The subclass does this with a store deposit collected in `_retrieve_recv_grads`; no hook, no autograd Function.
3. Chunk-id and lifecycle visibility for the submodule: a context carrying the current micro-batch, and callbacks at micro-batch end and step end. That removes the two wrappers the first version needed and enables per-micro-batch release.
4. Dense backward P2P buffers. `_backward_metadata_inference` records the strides of the next stage's input gradients and `_create_grad_recv_info` allocates with `torch.empty_strided`; a stage whose first op on an input is `cat` / `stack` / a slice hands back view gradients and c10d rejects the buffer (`Tensors for P2P must be non-overlapping and dense`). A dense `torch.empty(shape)` receive buffer and `.contiguous()` before send fix it; the model side carries an identity op with a `.contiguous()` backward until then.
5. A forward-only and a no-gradient contract for custom stages: `backward_one_chunk` under `has_backward=False` (what `schedule.eval` calls) as part of the subclass contract, and stage outputs with `requires_grad=False` whose gradient is `None` rather than assumed present (a frozen embedding under LoRA feeds the first block).

--- PASTE END ---
