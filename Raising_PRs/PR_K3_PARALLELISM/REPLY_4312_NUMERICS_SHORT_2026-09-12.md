# Short reply to Tianyu's numerics question on PR 4312 (2026-09-12)

For the user to post. CLAUDE.md numerics-acceptance rule: the residual is located (layer, op, first tensor, magnitude; probe `matrix_scripts/pp_step1_0912/`, tree `pp_review4` + probe patch) but its mechanism is not explained, which the rule says blocks posting -- the user's call. Numbers: 4 x H100 PCIe, `pp_review4` = `dbc425403`, logs in `phase13_k3like_48b_posttrain/pp_h100x4_logs_2026-09-12/`; step-1 campaign on 8 x RTX 5060 Ti, `PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md`. The two no-pipeline controls use local probe switches (`MB_REVERSE`, `NOSYNC_GA`), not the PR. Long form: `REPLY_4312_NUMERICS_2026-09-11.md`.

--- PASTE BEGIN ---

Against the bar -- step-1 loss bitwise, step-1 gradients bitwise or located -- the pipeline passes: step 1 matches a single GPU in every cell below, and the one step-1 gradient difference left is located in the last layer's attention backward, before anything crosses a stage boundary.

I moved these runs to H100. KDA runs Attention Gym kernels outside the SM100/SM103 path main supports, with configurations autotuned per process, so how much this debug flavour amplifies ulp-level differences by step 10 depends on the device: the same pp2 cell reads +13.7% at step 10 on RTX 5060 Ti and +3.6% on H100.

**Accumulation order, flag off / on vs no PP.** Take one block of the residual stack under pp2 x vp2 (rank 0 runs stages 0 and 2, rank 1 stages 1 and 3). Without PP, autograd adds every layer's read of that block onto one running gradient, top-down. With the cache off, each hop hands that running gradient back and the previous stage keeps adding onto it, so the order is the same as without PP. With the cache on, stages 2 and 3 read the block from their rank's store, sum their own reads from zero and deposit the subtotal, and stages 1 and 0 add it when they collect: the same terms, grouped differently -- bitwise-different in bf16 where terms cancel, identical in float64.

**Why this is not a bug.** Comparing every parameter's step-1 gradient, with the predictions written down before the dumps were read: the cache changes only the parameters that produce a block read from a rank's store, by 2-3 bf16 ulps, and leaves everything downstream bitwise; deleting one gradient deposit, a real bug in that path, moves the same tensors about a hundred times further.

4 x H100 PCIe, one seed checkpoint, 100 steps; 1024 tokens per step because four stages need four 256-token micro-batches; steps stop at 20 because the reference memorises the debug set after that. Each cell gives the raw value and, beneath it, the change against the first row; the last row has no pipeline in it.

| cell | loss step 1 | loss step 10 | loss step 20 | grad norm step 1 | grad norm step 10 | grad norm step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `3.373330` | `18.625` | `5.4375` | `3.9844` |
| pp2 | `12.605700`<br>bitwise | `3.227050`<br>+3.61% | `3.288290`<br>-2.52% | `18.75`<br>+0.67% | `5.6875`<br>+4.60% | `3.7344`<br>-6.27% |
| pp2 x vp2, cache on | `12.605700`<br>bitwise | `3.150940`<br>+1.17% | `3.349300`<br>-0.71% | `18.625`<br>0% | `3.7188`<br>-31.61% | `4.0625`<br>+1.96% |
| pp2 x vp2, cache off | `12.605700`<br>bitwise | `3.514970`<br>+12.85% | `3.281700`<br>-2.72% | `18.625`<br>0% | `6.0312`<br>+10.92% | `3.6719`<br>-7.84% |
| dp1, accumulation order reversed | `12.605700`<br>bitwise | `3.247610`<br>+4.27% | `3.295370`<br>-2.31% | `18.625`<br>0% | `6.6562`<br>+22.41% | `4.25`<br>+6.67% |

dp2, 2048 tokens per step; raw value and, beneath it, the change against dp2; the last row has no pipeline in it.

| cell | loss step 1 | loss step 10 | loss step 20 | grad norm step 1 | grad norm step 10 | grad norm step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 | `12.521140` | `3.221120` | `2.839810` | `16.375` | `7.0625` | `2.4844` |
| dp2 x pp2 | `12.521140`<br>same | `3.201920`<br>-0.60% | `2.846440`<br>+0.23% | `16.375`<br>0% | `5`<br>-29.20% | `2.6562`<br>+6.92% |
| dp2 x pp2 x vp2, cache on | `12.521140`<br>same | `3.205680`<br>-0.48% | `2.682970`<br>-5.52% | `16.375`<br>0% | `5.375`<br>-23.89% | `2.5156`<br>+1.26% |
| dp2 x pp2 x vp2, cache off | `12.521140`<br>same | `3.189240`<br>-0.99% | `2.847220`<br>+0.26% | `16.375`<br>0% | `5.5938`<br>-20.80% | `2.375`<br>-4.40% |
| dp2 x ep2 | `12.521140`<br>same | `3.149310`<br>-2.23% | `2.969330`<br>+4.56% | `16.375`<br>0% | `5.0625`<br>-28.32% | `2.9531`<br>+18.87% |
