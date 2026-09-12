# Short reply to Tianyu's numerics question on PR 4312 (2026-09-12)

For the user to post. CLAUDE.md numerics-acceptance rule: the residual is located (layer, op, first tensor, magnitude; probe `matrix_scripts/pp_step1_0912/`, tree `pp_review4` + probe patch) but its mechanism is not explained, which the rule says blocks posting -- the user's call. Numbers: 4 x H100 PCIe, `pp_review4` = `dbc425403`, logs in `phase13_k3like_48b_posttrain/pp_h100x4_logs_2026-09-12/`; step-1 campaign on 8 x RTX 5060 Ti, `PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md`. The two no-pipeline controls use local probe switches (`MB_REVERSE`, `NOSYNC_GA`), not the PR. Long form: `REPLY_4312_NUMERICS_2026-09-11.md`.

--- PASTE BEGIN ---

Against our bar -- step-1 loss bitwise, step-1 gradients bitwise or located -- the pipeline passes: step 1 matches a single GPU in every cell below, and the one step-1 gradient difference left is located in the last layer's attention backward, before anything crosses a stage boundary.

We moved these runs to H100. KDA runs Attention Gym kernels outside the SM100/SM103 path main supports, with configurations autotuned per process, so how much this debug flavour amplifies ulp-level differences by step 10 depends on the device: the same pp2 cell reads +13.7% at step 10 on RTX 5060 Ti and +3.6% on H100.

Not a bug: comparing every parameter's step-1 gradient, the cache changes only the parameters that produce a block read from a rank's store, by 2-3 bf16 ulps, and leaves everything downstream bitwise; deleting one gradient deposit, a real bug in that path, moves the same tensors about a hundred times further.

4 x H100 PCIe, one seed checkpoint, 100 steps; 1024 tokens per step because four stages need four 256-token micro-batches; steps stop at 20 because the reference memorises the debug set after that. Percentages against the first row; the last row has no pipeline in it.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `3.373330` | `18.625` | `5.4375` | `3.9844` |
| pp2 | same | +3.61% | -2.52% | +0.67% | +4.60% | -6.27% |
| pp2 x vp2, cache on | same | +1.17% | -0.71% | same | -31.6% | +1.96% |
| pp2 x vp2, cache off | same | +12.85% | -2.72% | same | +10.9% | -7.84% |
| dp1, accumulation order reversed | same | +4.27% | -2.31% | same | +22.4% | +6.67% |

dp2, 2048 tokens per step, against dp2; the last row has no pipeline in it.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 | `12.521140` | `3.221120` | `2.839810` | `16.375` | `7.0625` | `2.4844` |
| dp2 x pp2 | same | -0.60% | +0.23% | same | -29.20% | +6.92% |
| dp2 x pp2 x vp2, cache on | same | -0.48% | -5.52% | same | -23.89% | +1.26% |
| dp2 x pp2 x vp2, cache off | same | -0.99% | +0.26% | same | -20.80% | -4.40% |
| dp2 x ep2 | same | -2.23% | +4.56% | same | -28.32% | +18.87% |
