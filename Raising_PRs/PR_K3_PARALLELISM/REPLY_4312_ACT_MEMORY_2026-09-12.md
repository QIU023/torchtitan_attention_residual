# Reply to Tianyu on PR 4312: what the pipeline stages keep, and when it is released (comment 3976976576, `model.py` line 242)

For the user to post as the reply to 3976976576. Our earlier reply (3932773900) already covered the layer / block relation; this one answers per pipeline stage, which is where the rank-local cache matters. Tables computed on 2026-09-13 with the PR head's own code (`dbc425403`): the split from `_generate_llm_fqn_per_model_part(4, 8, 1, 1, last_stage_modules=...)`, the routing from `BlockLayoutTables` (cache on and off), release / deposit points from `AttnResPipelineStage.forward_one_chunk` / `backward_one_chunk` and `PPRankLocalCache`. Entry 0 is the embedding, entry 1 block 1's result (`x4`); block 2's result is consumed by the output aggregation and never enters the stack.

--- PASTE BEGIN ---

Our earlier reply covered the layer / block relation, so here is the per-stage view, where the cache matters. Your example, 2 blocks x 4 layers, with pp2 x vp2 (Interleaved1F1B) and the split core generates:

| stage (rank, virtual) | layers | adds to the stack | received by P2P, cache off -> on | read from the rank's cache |
| --- | --- | --- | --- | --- |
| 0 (r0, v0) | emb, 0-1 | `e` (entry 0) | - | - |
| 1 (r1, v0) | 2-4 | `x4` = block 1's result (entry 1) | `[e]` -> `[e]` | - |
| 2 (r0, v1) | 5-6 | - | `[e, x4]` -> `[x4]` | `e` (stage 0 put it there) |
| 3 (r1, v1) | 7, head | - | `[e, x4]` -> nothing | `e`, `x4` (stage 1 put them there) |

Per micro-batch the hops carry 5 entries without the cache and 2 with it: a hop carries only the entries the receiving rank has not seen yet.

Forward: a stage puts each entry it adds or receives into its rank's cache (a detached view of the stack, no copy). The rank drops a micro-batch's entries right after its last stage's forward for that micro-batch (stage 2 on rank 0, stage 3 on rank 1). What backward needs is held by autograd as usual, not by the cache.

Backward: a stage that read an entry from the cache does not send that entry's gradient anywhere; it deposits it in the rank's cache (stage 3: `e` and `x4`; stage 2: `e`). The stage that put the entry on the rank adds the deposit to its own gradient before sending upstream (stage 1 collects both on rank 1, stage 0 collects `e` on rank 0). So a deposit lives from the reader's backward to the owner's backward of the same micro-batch, and the rank checks none is left after its first stage's backward.

Inside a stage, what is saved and recomputed is the same as without PP: with the default `SelectiveAC` each layer is one checkpoint region, and the residual reads are recomputed in that layer's backward.
