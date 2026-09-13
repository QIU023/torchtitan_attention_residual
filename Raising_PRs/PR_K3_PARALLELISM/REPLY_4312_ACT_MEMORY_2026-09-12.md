# Reply to Tianyu on PR 4312: what the pipeline stages keep, and when it is released (comment 3976976576, `model.py` line 242)

For the user to post as the reply to 3976976576. Our earlier reply (3932773900) already covered the layer / block relation; this one answers per pipeline stage, which is where the rank-local cache matters. Tables computed on 2026-09-13 with the PR head's own code (`dbc425403`): the split from `_generate_llm_fqn_per_model_part(4, 8, 1, 1, last_stage_modules=...)`, the routing from `BlockLayoutTables` (cache on and off), release / deposit points from `AttnResPipelineStage.forward_one_chunk` / `backward_one_chunk` and `PPRankLocalCache`. Entry 0 is the embedding, entry 1 block 1's result (`x4`); block 2's result is consumed by the output aggregation and never enters the stack.

--- PASTE BEGIN ---

Our earlier reply covered the layer / block relation, so here is the per-stage view, where the cache matters. Your example, 2 blocks x 4 layers, with pp2 x vp2 (Interleaved1F1B: global stage s runs on rank s % 2) and the split core generates. `e` is the embedding (stack entry 0), `x4` block 1's result (entry 1); block 2's result is consumed by the output aggregation and never enters the stack.

| global stage (rank, virtual) | layers | adds to the stack | received via Interleaved1F1B P2P, cache off | received via Interleaved1F1B P2P, cache on | already cached |
| --- | --- | --- | --- | --- | --- |
| 0 (r0, v0) | emb, 0-1 | `e` | - | - | - |
| 1 (r1, v0) | 2-4 | `x4` | `e` from stage 0 | `e` from stage 0 | - |
| 2 (r0, v1) | 5-6 | - | `e`, `x4` from stage 1 | `x4` from stage 1 | `e` |
| 3 (r1, v1) | 7, head | - | `e`, `x4` from stage 2 | nothing | `e`, `x4` |

Stack entries sent per micro-batch: 5 with the cache off, 2 with it on (the hidden state travels on every hop either way and is not listed).

A hop carries only the entries the receiving rank has not seen yet. The same with 4 blocks x 4 layers, pp4 x vp4 (stage s on rank s % 4; `x4`, `x8`, `x12` = the results of blocks 1-3):

| global stage (rank, virtual) | layers | adds to the stack | received via Interleaved1F1B P2P, cache off | received via Interleaved1F1B P2P, cache on | already cached |
| --- | --- | --- | --- | --- | --- |
| 0 (r0, v0) | emb, 0 | `e` | - | - | - |
| 1 (r1, v0) | 1-2 | - | `e` from stage 0 | `e` from stage 0 | - |
| 2 (r2, v0) | 3 | - | `e` from stage 1 | `e` from stage 1 | - |
| 3 (r3, v0) | 4 | `x4` | `e` from stage 2 | `e` from stage 2 | - |
| 4 (r0, v1) | 5 | - | `e`, `x4` from stage 3 | `x4` from stage 3 | `e` |
| 5 (r1, v1) | 6 | - | `e`, `x4` from stage 4 | `x4` from stage 4 | `e` |
| 6 (r2, v1) | 7 | - | `e`, `x4` from stage 5 | `x4` from stage 5 | `e` |
| 7 (r3, v1) | 8 | `x8` | `e`, `x4` from stage 6 | nothing | `e`, `x4` |
| 8 (r0, v2) | 9 | - | `e`, `x4`, `x8` from stage 7 | `x8` from stage 7 | `e`, `x4` |
| 9 (r1, v2) | 10 | - | `e`, `x4`, `x8` from stage 8 | `x8` from stage 8 | `e`, `x4` |
| 10 (r2, v2) | 11 | - | `e`, `x4`, `x8` from stage 9 | `x8` from stage 9 | `e`, `x4` |
| 11 (r3, v2) | 12 | `x12` | `e`, `x4`, `x8` from stage 10 | nothing | `e`, `x4`, `x8` |
| 12 (r0, v3) | 13 | - | `e`, `x4`, `x8`, `x12` from stage 11 | `x12` from stage 11 | `e`, `x4`, `x8` |
| 13 (r1, v3) | 14 | - | `e`, `x4`, `x8`, `x12` from stage 12 | `x12` from stage 12 | `e`, `x4`, `x8` |
| 14 (r2, v3) | 15 | - | `e`, `x4`, `x8`, `x12` from stage 13 | `x12` from stage 13 | `e`, `x4`, `x8` |
| 15 (r3, v3) | head | - | `e`, `x4`, `x8`, `x12` from stage 14 | nothing | `e`, `x4`, `x8`, `x12` |

Stack entries sent per micro-batch: 39 with the cache off, 12 with it on (the hidden state travels on every hop either way and is not listed).

Forward: a stage puts each entry it adds or receives into its rank's cache (a detached view of the stack, no copy). The rank drops a micro-batch's entries right after its last stage's forward for that micro-batch (stage 2 on rank 0, stage 3 on rank 1). What backward needs is held by autograd as usual, not by the cache.

Backward: a stage that read an entry from the cache does not send that entry's gradient anywhere; it deposits it in the rank's cache (stage 3: `e` and `x4`; stage 2: `e`). The stage that put the entry on the rank adds the deposit to its own gradient before sending upstream (stage 1 collects both on rank 1, stage 0 collects `e` on rank 0). So a deposit lives from the reader's backward to the owner's backward of the same micro-batch, and the rank checks none is left after its first stage's backward.

Inside a stage, what is saved and recomputed is the same as without PP: with the default `SelectiveAC` each layer is one checkpoint region, and the residual reads are recomputed in that layer's backward.
