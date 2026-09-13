# PP rank cache: step-1 gradients, cache on vs off (local 8 x 5060 Ti only, 2026-09-13)

Not H100 numbers; never mix into the H100 tables. Tree: `pp_fp64_probe` (`/tmp/wt_fp64`, dbc425403 + c4 probe,
NOSYNC_GA, PP_STAGES_PER_RANK, GRAD_DUMP). c4 1024 tokens (4 x 256), one seed, one shared inductor/triton cache,
one step, every parameter gradient dumped. Scripts: `matrix_scripts/tp_h100_v2/cache_order_gradprobe*.sh`,
`cache_order_compare.py`.

## bf16 (default training dtype)

| cell vs dp1 no-sync | bitwise params |
| --- | --- |
| pp4 x vp4, cache off | 680/680 (twice) |
| pp4 x vp4, cache on | 346/680 (twice, identical to each other) |

Cache on: layers 12+ bitwise; layers 0-11 and `tok_embeddings` differ, relative norm diff ~1e-2 (max 7.4e-2,
`layers.0 ffn_res_norm.weight`). Deterministic.

## Code review of the deposit path (`pipeline_stage.py` at dbc425403)

A reader stage deposits the gradient of every stack column it read from the rank cache (`split_stack_grad`,
`PPRankLocalCache.deposit`: `prior + grad_TD`, counted). The stage that brought the block onto the rank adds the
summed deposit to the gradient that came down the chain (`_collect_into`: `grad_col_TD.add_(deposit)`), after checking
the count against `layout.deposits_expected`; a missing or extra deposit raises, and so does a deposit left after the
rank's last backward. So no gradient is dropped or added twice; what changes is the association of the bf16 sum
(deposits first, then the chain), whereas cache off adds hop by hop in the order single-GPU autograd uses, which is why
cache off is bitwise with dp1.

## fp32 (`FP32_PROBE=1`: bf16 casts kept float32, experts looped in float32, KDA `impl="reference"`)

| cell vs fp32 dp1 no-sync | bitwise params |
| --- | --- |
| pp4 x vp4, cache off | 680/680 |
| pp4 x vp4, cache on | (rerun: the first dump was truncated by a full disk) |

fp32 dp1 at 1024 tokens OOMs on 16 GB in the optimizer step; its step-1 gradients are dumped before that.

## Old vs new split code (5060 only)

pp2 x vp2 cache on, 5 steps, shared cache: `dbc425403` vs `dbc425403 + c4fee4afd`, loss and grad norm identical at
steps 1-5, step-1 gradients 680/680 bitwise.
