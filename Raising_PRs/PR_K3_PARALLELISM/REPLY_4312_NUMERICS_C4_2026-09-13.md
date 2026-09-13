# Numerics note for PR 4312 (c4 re-run), to post in the numerics thread (reply to 3985333653, `parallelize.py`)

The user attaches this as the new table in the numerics comment. H100 only; logs in `phase13_k3like_48b_posttrain/pp_h100x4_c4_logs_2026-09-13/`. Step 100 against each stream's no-sync reference: 1024 tokens pp2 -0.22%, pp2 x vp2 cached -0.10% / naive +0.11%, pp4 x vp4 cached +0.10% / naive -0.12%, floor (dp1 reversed) -0.19%; 2048 tokens dp2 x pp2 -0.48%, cached -2.08%, naive -1.11%, floor (dp2 reversed) -0.24%, dp2 x ep2 -0.93%.

Not in the paste (kept for reference): the cached rows move further than the naive ones because the rank cache sums a block's gradient contributions in another bf16 order; fp32 step-1 probe on the 5060: cache on vs off 7e-2 -> 9e-6 relative (`phase13_k3like_48b_posttrain/PP_CACHE_ORDER_PROBE_5060_2026-09-13.md`).

--- PASTE BEGIN ---

Re-run on the same 4 x H100, now on `c4_test` as text-only rows, since at 1024 tokens per step the reference memorised the original 32-sample debug set within the 100 steps. Step 1 is bitwise in every cell, and at step 100 every pipeline cell is within the range of the order-only noise-floor rows.

| cell | loss, step 1 | step 10 | step 20 | step 100 | grad norm, step 1 | step 10 | step 20 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1, no-sync accumulation (reference) | `12.609980` | `3.335430` | `3.007030` | `2.568440` | `16.875` | `3.2812` | `2.5312` | `1.5547` |
| pp2 | `12.609980`<br>bitwise | `3.334370`<br>-0.03% | `2.988620`<br>-0.61% | `2.562870`<br>-0.22% | `16.875`<br>0% | `3.2969`<br>+0.48% | `2.2188`<br>-12.34% | `1.5312`<br>-1.51% |
| pp2 x vp2, cached | `12.609980`<br>bitwise | `3.310030`<br>-0.76% | `3.032070`<br>+0.83% | `2.565810`<br>-0.10% | `16.875`<br>0% | `3.2812`<br>0% | `2.9375`<br>+16.05% | `1.5781`<br>+1.51% |
| pp2 x vp2, naive | `12.609980`<br>bitwise | `3.331430`<br>-0.12% | `2.996710`<br>-0.34% | `2.571170`<br>+0.11% | `16.875`<br>0% | `3.3125`<br>+0.95% | `2.4062`<br>-4.94% | `1.5547`<br>0% |
| pp4 x vp4, cached | `12.609980`<br>bitwise | `3.528470`<br>+5.79% | `3.059910`<br>+1.76% | `2.571010`<br>+0.10% | `16.875`<br>0% | `10`<br>+204.77% | `2.75`<br>+8.64% | `1.6094`<br>+3.52% |
| pp4 x vp4, naive | `12.609980`<br>bitwise | `3.334860`<br>-0.02% | `2.995440`<br>-0.39% | `2.565350`<br>-0.12% | `16.875`<br>0% | `3.2656`<br>-0.48% | `2.4219`<br>-4.32% | `1.5391`<br>-1.00% |
| dp1, default accumulation (no pipeline) | `12.609980`<br>bitwise | `3.421960`<br>+2.59% | `3.022710`<br>+0.52% | `2.536600`<br>-1.24% | `16.875`<br>0% | `5.0625`<br>+54.29% | `2.1406`<br>-15.43% | `1.5703`<br>+1.00% |
| dp1, accumulation order reversed (noise floor, no pipeline) | `12.609980`<br>bitwise | `3.389450`<br>+1.62% | `3.066510`<br>+1.98% | `2.563570`<br>-0.19% | `16.875`<br>0% | `4.6875`<br>+42.86% | `2.9531`<br>+16.67% | `1.6797`<br>+8.04% |

dp2, 2048 tokens per step (four 256-token micro-batches per rank), same protocol and reference accumulation; a rerun of dp2 x pp2 matched it on all 100 steps.

| cell | loss, step 1 | step 10 | step 20 | step 100 | grad norm, step 1 | step 10 | step 20 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2, no-sync accumulation (reference) | `12.580740` | `3.663490` | `2.974910` | `2.443230` | `14.4375` | `14.1875` | `2.4219` | `1.0312` |
| dp2 x pp2 | `12.580740`<br>bitwise | `3.607360`<br>-1.53% | `2.978480`<br>+0.12% | `2.431600`<br>-0.48% | `14.4375`<br>0% | `13.1875`<br>-7.05% | `2.4375`<br>+0.64% | `0.9609`<br>-6.82% |
| dp2 x pp2 x vp2, cached | `12.580740`<br>bitwise | `3.369740`<br>-8.02% | `2.922550`<br>-1.76% | `2.392360`<br>-2.08% | `14.4375`<br>0% | `5.625`<br>-60.35% | `2.2031`<br>-9.03% | `1.0234`<br>-0.76% |
| dp2 x pp2 x vp2, naive | `12.580740`<br>bitwise | `3.687230`<br>+0.65% | `2.973080`<br>-0.06% | `2.416180`<br>-1.11% | `14.4375`<br>0% | `14.8125`<br>+4.41% | `2.3906`<br>-1.29% | `0.9961`<br>-3.40% |
| dp2, default accumulation (no pipeline) | `12.580740`<br>bitwise | `3.307830`<br>-9.71% | `2.930770`<br>-1.48% | `2.426700`<br>-0.68% | `14.4375`<br>0% | `4.5312`<br>-68.06% | `2.3438`<br>-3.22% | `1.0312`<br>0% |
| dp2, accumulation order reversed (noise floor, no pipeline) | `12.580740`<br>bitwise | `3.588150`<br>-2.06% | `2.960280`<br>-0.49% | `2.437380`<br>-0.24% | `14.4375`<br>0% | `12.125`<br>-14.54% | `2.2656`<br>-6.45% | `0.9766`<br>-5.29% |
| dp2 x ep2 (no pipeline) | `12.580740`<br>bitwise | `3.269320`<br>-10.76% | `2.966810`<br>-0.27% | `2.420510`<br>-0.93% | `14.4375`<br>0% | `4.4375`<br>-68.72% | `2.25`<br>-7.10% | `1.0312`<br>0% |

--- PASTE END ---
