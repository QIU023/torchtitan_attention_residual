# Numerics note for PR 4312 (c4 re-run), to post in the numerics thread (reply to 3985333653, `parallelize.py`)

The user attaches this as the new table in the numerics comment. HOLD: the dp2 cached cell's step-100 -2.08% is read against more order-only samples queued on the H100 (`run_pp_c4_d2_floor.sh`); fill the floor numbers in before posting. H100 only; logs in `phase13_k3like_48b_posttrain/pp_h100x4_c4_logs_2026-09-13/`. Step 100 against each stream's no-sync reference: 1024 tokens pp2 -0.22%, pp2 x vp2 cached -0.10% / naive +0.11%, pp4 x vp4 cached +0.10% / naive -0.12%, floor (dp1 reversed) -0.19%; 2048 tokens dp2 x pp2 -0.48%, cached -2.08%, naive -1.11%, floor (dp2 reversed) -0.24%, dp2 x ep2 -0.93%.

--- PASTE BEGIN ---

Re-run on the same 4 x H100, now on `c4_test` as text-only rows: at 1024 tokens per step the reference memorised the 32-sample debug set within the 100 steps, so its late steps measured memorisation. Step 1 is identical in every cell (logged loss and grad norm). At step 100 the whole-stack (naive) cells sit with the no-pipeline reorderings: within 0.22% at 1024 tokens (reversed-order floor 0.19%), within 1.11% at 2048 tokens with dp2 (floor 0.24%, dp2 x ep2 0.93%).

The cached rows move further (dp2 x pp2 x vp2 cached: -8.02% at step 10, -2.08% at step 100), and that gap is located: the rank cache changes only the order in which a cached block's gradient contributions are added -- the deposits are summed first, then added to the gradient coming down the chain. Step-1 gradients on a separate box at pp4 x vp4 against the same reference: cache off is bitwise on all 680 parameters; cache on differs in 334 (layers 0-11 and `tok_embeddings`), by up to 7.4e-2 relative in bf16 and 9.1e-6 in fp32. Same parameters, the difference shrinking to fp32 rounding: a reordered sum, not a missing or extra term. Every deposit is counted against the routing tables, and a missing, extra or uncollected one raises.

c4 (2026-09-13): 4 x H100 PCIe, `kimi_k3_debugmodel` (24 layers) reading `c4_test` as text-only 256-token rows (`kimi_k3_debugmodel_c4`, a local data flavor that is not part of this PR; the 32-sample debug set is memorised by step 20, c4 is not by step 100), one seed checkpoint, 1024 tokens per step as four 256-token micro-batches; the reference accumulates the four micro-batches in fp32 with the gradient sync on the last one, as the pipeline does (a local probe switch); naive rows set `attn_res_cache=False`; pp4 x vp4 is 16 stages through a local stage-count switch; each cell gives the raw value and, beneath it, the change against the reference.

| cell | loss, step 1 | step 10 | step 20 | step 100 | grad norm, step 1 | step 10 | step 20 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1, no-sync accumulation (reference) | `12.609980` | `3.335430` | `3.007030` | `2.568440` | `16.875` | `3.2812` | `2.5312` | `1.5547` |
| pp2 | `12.609980`<br>identical | `3.334370`<br>-0.03% | `2.988620`<br>-0.61% | `2.562870`<br>-0.22% | `16.875`<br>0% | `3.2969`<br>+0.48% | `2.2188`<br>-12.34% | `1.5312`<br>-1.51% |
| pp2 x vp2, cached | `12.609980`<br>identical | `3.310030`<br>-0.76% | `3.032070`<br>+0.83% | `2.565810`<br>-0.10% | `16.875`<br>0% | `3.2812`<br>0% | `2.9375`<br>+16.05% | `1.5781`<br>+1.51% |
| pp2 x vp2, naive | `12.609980`<br>identical | `3.331430`<br>-0.12% | `2.996710`<br>-0.34% | `2.571170`<br>+0.11% | `16.875`<br>0% | `3.3125`<br>+0.95% | `2.4062`<br>-4.94% | `1.5547`<br>0% |
| pp4 x vp4, cached | `12.609980`<br>identical | `3.528470`<br>+5.79% | `3.059910`<br>+1.76% | `2.571010`<br>+0.10% | `16.875`<br>0% | `10`<br>+204.77% | `2.75`<br>+8.64% | `1.6094`<br>+3.52% |
| pp4 x vp4, naive | `12.609980`<br>identical | `3.334860`<br>-0.02% | `2.995440`<br>-0.39% | `2.565350`<br>-0.12% | `16.875`<br>0% | `3.2656`<br>-0.48% | `2.4219`<br>-4.32% | `1.5391`<br>-1.00% |
| dp1, default accumulation (no pipeline) | `12.609980`<br>identical | `3.421960`<br>+2.59% | `3.022710`<br>+0.52% | `2.536600`<br>-1.24% | `16.875`<br>0% | `5.0625`<br>+54.29% | `2.1406`<br>-15.43% | `1.5703`<br>+1.00% |
| dp1, accumulation order reversed (noise floor, no pipeline) | `12.609980`<br>identical | `3.389450`<br>+1.62% | `3.066510`<br>+1.98% | `2.563570`<br>-0.19% | `16.875`<br>0% | `4.6875`<br>+42.86% | `2.9531`<br>+16.67% | `1.6797`<br>+8.04% |

dp2, 2048 tokens per step (four 256-token micro-batches per rank), same protocol and reference accumulation; a rerun of dp2 x pp2 matched it on all 100 steps.

| cell | loss, step 1 | step 10 | step 20 | step 100 | grad norm, step 1 | step 10 | step 20 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2, no-sync accumulation (reference) | `12.580740` | `3.663490` | `2.974910` | `2.443230` | `14.4375` | `14.1875` | `2.4219` | `1.0312` |
| dp2 x pp2 | `12.580740`<br>identical | `3.607360`<br>-1.53% | `2.978480`<br>+0.12% | `2.431600`<br>-0.48% | `14.4375`<br>0% | `13.1875`<br>-7.05% | `2.4375`<br>+0.64% | `0.9609`<br>-6.82% |
| dp2 x pp2 x vp2, cached | `12.580740`<br>identical | `3.369740`<br>-8.02% | `2.922550`<br>-1.76% | `2.392360`<br>-2.08% | `14.4375`<br>0% | `5.625`<br>-60.35% | `2.2031`<br>-9.03% | `1.0234`<br>-0.76% |
| dp2 x pp2 x vp2, naive | `12.580740`<br>identical | `3.687230`<br>+0.65% | `2.973080`<br>-0.06% | `2.416180`<br>-1.11% | `14.4375`<br>0% | `14.8125`<br>+4.41% | `2.3906`<br>-1.29% | `0.9961`<br>-3.40% |
| dp2, default accumulation (no pipeline) | `12.580740`<br>identical | `3.307830`<br>-9.71% | `2.930770`<br>-1.48% | `2.426700`<br>-0.68% | `14.4375`<br>0% | `4.5312`<br>-68.06% | `2.3438`<br>-3.22% | `1.0312`<br>0% |
| dp2, accumulation order reversed (noise floor, no pipeline) | `12.580740`<br>identical | `3.588150`<br>-2.06% | `2.960280`<br>-0.49% | `2.437380`<br>-0.24% | `14.4375`<br>0% | `12.125`<br>-14.54% | `2.2656`<br>-6.45% | `0.9766`<br>-5.29% |
| dp2 x ep2 (no pipeline) | `12.580740`<br>identical | `3.269320`<br>-10.76% | `2.966810`<br>-0.27% | `2.420510`<br>-0.93% | `14.4375`<br>0% | `4.4375`<br>-68.72% | `2.25`<br>-7.10% | `1.0312`<br>0% |

--- PASTE END ---
