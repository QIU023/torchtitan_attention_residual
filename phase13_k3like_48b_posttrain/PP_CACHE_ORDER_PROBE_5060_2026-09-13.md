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

## H100: the one-step grad-norm digit (step 59 at 1024 tokens, step 27 at 256), located

Full-precision norm printed every step (`gn_repr_hack.py`, `run_pp_c4_gnrepr.sh`), fp32 grad norm, 1024 tokens, all
four cells on the warmed shared cache `jitwarm_sh1024`: the logged loss is identical to the reference on all 100 steps in
pp2, pp2 x vp2 naive and pp4 x vp4 naive; the norm differs on 48-53 of the 100 steps by about 1e-7 relative (one fp32
unit: step 1 16.919260025 against 16.919258118). The pipeline sums the squared norms per rank and all-reduces them, a
different fp32 summation order than one GPU's; the reference's step-59 norm is 1.6628501415, on the rounding boundary of
the fourth printed decimal, so that step prints 1.6629 against 1.6628. Not the compile cache, and the loss never moves.

## float64 end to end (5060, 2026-09-13)

`pp_fp64_probe` = `9a9400bfc` (patches and full diff in `matrix_scripts/tp_h100_v2/fp64_probe/`), `FP64_PROBE=1`: every
float cast to float64, KDA and the short conv on Attention Gym's eager oracles, eager flex, per-expert MoE loop, float64
loss accumulation; params, compute and reduce in float64 (grad dumps are float64). c4 64-token rows, 256 tokens per step
as 4 x 64, pp4 x vp4, one seed.

Step 1, cache off vs cache on: loss 12.6257936118563556 and grad norm 22.8332725616344483 in both, to every printed
digit. Gradients: 346/680 bitwise, the other 334 are the same parameters as in bf16 and fp32 (layers 0-11 and
`tok_embeddings`), max relative difference 1.68e-14. The cache-on difference follows the working precision -- 7.4e-2
(bf16), 9.1e-6 (fp32), 1.7e-14 (fp64) -- so it is the order of the additions and nothing else.

pp4 x vp4 in float64 runs out of memory on the rank that holds `lm_head` at the first optimizer step (16 GB cards); the
trajectories run as pp8 x vp2 (`run_fp64_pp8vp2.sh`).

### float64 trajectories, 20 steps, pp8 x vp2 (cache off vs cache on)

`run_fp64_pp8vp2.sh` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without it the lm_head rank ran out of
memory at step 4; peak 14.1 GiB of 15.5), same seed and data, 20 steps each; full-precision comparison in
`fp64_5060_logs_2026-09-13/repr_compare_pp8vp2.txt`. The loss agrees on every step to within 4e-14 relative (identical
at steps 1-3 and 8), the grad norm to within 1.3e-12. In bf16 the cached pp4 x vp4 cell was 5.8% off the naive one in
loss and 205% in grad norm at step 10 (H100 table); in float64 the same comparison stays at the 1e-14 level for all 20
steps, so the bf16 gap is rounding amplified by the flavor's early steps, not a difference in what the cache computes.

## bf16 / fp32 / fp64, pp8 x vp2, cache off vs cache on, 100 steps (5060, 2026-09-13)

`run_precision_triplet.sh` on `pp_fp64_probe` `9a9400bfc`: c4 64-token rows, 256 tokens per step as 4 x 64, one fp32
seed, per precision one compile cache warmed by a 1-step run of both cells, `KDA_NOAUTOTUNE=1`, expandable segments;
bf16 is the standard path (no probe switch; the full-precision print only logs). Logs and per-step comparisons in
`fp64_5060_logs_2026-09-13/triplet/`. Mean |relative loss difference| (cache on against cache off) per window:

| steps | bf16 | fp32 | fp64 |
| --- | --- | --- | --- |
| 1-5 | 1.4e-2 | 1.3e-7 | 7.9e-16 |
| 6-10 | 5.2e-2 | 3.9e-7 | 9.1e-15 |
| 11-20 | 1.9e-2 | 8.2e-5 | 1.9e-14 |
| 21-40 | 7.8e-3 | 1.1e-3 | 9.2e-12 |
| 41-60 | 8.1e-3 | 5.0e-3 | 1.3e-4 |
| 61-80 | 4.8e-3 | 4.3e-3 | 2.2e-3 |
| 81-100 | 4.5e-3 | 3.9e-3 | 3.9e-3 |

Step 1 is identical in all three; the first loss difference appears at step 2 (bf16, 5e-3), 3 (fp32, 1e-7), 4 (fp64,
~1e-15), i.e. at the working precision's rounding, and grows until it saturates: by steps 6-10 in bf16, 41-60 in fp32,
61-100 in fp64. The saturated band is the same in all three (about 0.4% mean at steps 81-100, sign changing), so the
precision only sets how late the two runs separate, not where they end up. Early peaks (bf16 9.5% at step 7) fall in
the phase where the loss drops from 12.6 to 3.7; the LR (warmup 2, 8e-4 until step 20, linear to 0 at step 100) stops
the growth late. Not measured: the parameter distance between the two runs.
