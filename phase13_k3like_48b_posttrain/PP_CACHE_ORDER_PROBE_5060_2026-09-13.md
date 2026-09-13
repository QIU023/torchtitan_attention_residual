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
| pp4 x vp4, cache on | 346/680 |

Cache on in fp32 differs in the same 334 parameters (layers 0-11, `tok_embeddings`) as in bf16, but the largest
relative difference falls from 7.4e-2 to 9.1e-6 (`layers.0 delta_attention.dt_bias`; 3e-7 to 9e-6 per layer), i.e.
to float32 rounding: the cache-on gap is the order of the additions, not a missing or extra term. (The first fp32
cache-on dump was truncated by a full disk and rerun.)

fp32 dp1 at 1024 tokens OOMs on 16 GB in the optimizer step; its step-1 gradients are dumped before that.

## Old vs new split code (5060 only)

pp2 x vp2 cache on, 5 steps, shared cache: `dbc425403` vs `dbc425403 + c4fee4afd`, loss and grad norm identical at
steps 1-5, step-1 gradients 680/680 bitwise.

## 256 tokens (64-token rows, 4 x 64), bf16, one shared cache (5060 only)

pp2 and pp2 x vp2 naive vs dp1 no-sync: 680/680 bitwise each, step-1 grad norm 22.75 in all three. The H100 256 kit's interleaved cells read 23.125 vs 23.000 at step 1 with a cold inductor cache per cell; warm-cache reruns queued there (`run_pp_c4_256_warm.sh`). The cache-on dump here was truncated by a full disk.

## H100: dp2 x pp2 under the fp32 grad norm (2026-09-13, H100 numbers, kept here because they close the probe)

The fp32 grad-norm table had dp2 x pp2 (1F1B) differ from step 1 (norm 14.4183, a rerun on a copy of its own cache
14.4192, reference 14.4170). Step-1 gradients dumped on a shared warm cache: dp2 x pp2 and dp2 x pp2 x vp2 naive both
680/680 bitwise with the reference, norm 14.4170. Four 3-step dp2 x pp2 runs on that same cache, two with
`torch.cuda.synchronize()` before the clip and two without: all four 14.4170 / 15.2297 / 10.0644 at steps 1-3, the
reference's values. So no stream race; the gap came from each 100-step cell running on its own (cold, then copied)
compile cache, which the numerics-table rule forbids. Rerun with one warmed cache per stream: `run_pp_c4_gn32_shared.sh`.
