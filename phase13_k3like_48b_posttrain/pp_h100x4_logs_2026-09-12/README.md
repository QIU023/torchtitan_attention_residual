# PR 4312 (pp_review4) on 4 x H100 PCIe, #4500's protocol (2026-09-12)

Tree: `pp_review4` = `dbc425403` (on upstream main `d9ca9e55a`) plus `matrix_scripts/tp_h100_v2/pp4h_probe.patch` (KDA guard widened, `kimi_k3_debugmodel_pp_naive`, `MB_REVERSE`, `NOSYNC_GA`; none of it in the PR). Box as `tp_h100x4_logs_2026-09-12/README.md`. Kit: `run_pp.sh`. Protocol: `seed=42`, `--debug.deterministic`, bf16, one seed checkpoint per batch shape, one inductor cache and a per-rank Triton cache per cell, 100 steps, `cc12m-test`; pp8 x vp4 left out. #4500's 256 tokens per step cannot hold a pipeline (the collator needs 256 per micro-batch, the pipeline one micro-batch per stage), so pp2 runs at 1024 (4 x 256) and 512 (2 x 256), pp2 x vp2 at 1024.

- The 1024-token cells reproduce the 2026-09-11 2 x H100 PCIe run (`h100_logs_2026-09-11/`, pp_review4 `8aea9ef03`) value for value: dp1 `12.605700` / `3.114620`, pp2 `3.227050`, pp2 x vp2 cached `3.150940` / naive `3.514970`, reversed accumulation `3.247610`.
- Step 10, 1024 tokens, against plain dp1: dp1 with the pipeline's accumulation (`NOSYNC_GA`, no pipeline) +3.85 %, reversed accumulation +4.27 %, pp2 +3.61 %, vp2 cached +1.17 %, vp2 naive +12.9 %. Against the matched-accumulation dp1: pp2 -0.24 %, vp2 cached -2.59 %, vp2 naive +8.67 %.
- 512 tokens (a two-term accumulation, exact either way): pp2 +4.71 % at step 10.
- Reportable steps (CLAUDE.md numerics-table rule, 2026-09-12): the 1024-token dp1 reference reads 3.71 / 3.77 at steps 8 / 9 (its first non-monotone step), 3.11 at 10, 3.37 at 20, 2.18 at 30 and 1.35 at 40, so steps 1 / 10 / 20 are shown and step 20 is at the edge; the 512-token reference first rises at step 11. Step 100 is never shown. The streamed-cc12m passes were dropped (the user, 2026-09-12: the original dataset only).
- Steps 1 / 10 / 20 against plain dp1 (loss): matched accumulation (no PP) 0 / +3.85 / -1.72 %, reversed accumulation 0 / +4.27 / -2.31 %, pp2 0 / +3.61 / -2.52 %, vp2 cached 0 / +1.17 / -0.71 %, vp2 naive 0 / +12.85 / -2.72 %. Against the matched-accumulation dp1: pp2 0 / -0.24 / -0.81 %, vp2 cached 0 / -2.59 / +1.03 %, vp2 naive 0 / +8.67 / -1.01 %. 512 tokens: pp2 0 / +4.71 / -0.37 %. Step-1 grad norm is bitwise except pp2 (+1 bf16 ulp).
- The matched-accumulation row is a probe. Upstream #4597 lets `training.mixed_precision_reduce` be bfloat16, which makes FSDP accumulate in the parameter dtype in both paths; pp_review4's base predates it.

## The full matrix on this box (2026-09-12), steps 1 / 10 / 20

Loss, then grad norm, as percentages against each block's reference; "same" is the same printed value. The reference loss first rises at step 9 in both the 1024- and 2048-token streams, so step 20 is at the edge of the numerics-table rule; nothing later is shown.

1024 tokens (4 x 256), reference dp1 (`12.605700` / `3.114620` / `3.373330`; grad norm `18.625` / `5.4375` / `3.9844`):

| cell | loss 1 | 10 | 20 | gnorm 1 | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1, pipeline's accumulation (no PP, probe) | same | +3.85% | -1.72% | same | -10.92% | -6.67% |
| dp1, reversed accumulation (no PP, probe) | same | +4.27% | -2.31% | same | +22.41% | +6.67% |
| pp2 | same | +3.61% | -2.52% | +0.67% | +4.60% | -6.27% |
| pp2 x vp2, cache on | same | +1.17% | -0.71% | same | -31.61% | +1.96% |
| pp2 x vp2, cache off | same | +12.85% | -2.72% | same | +10.92% | -7.84% |

1024 tokens with `training.mixed_precision_reduce=bfloat16` (#4597's option added to the probe tree), reference dp1 (`12.605700` / `3.220060` / `3.382050`); the pipeline's accumulation is then bitwise with dp1 for all 100 steps:

| cell | loss 1 | 10 | 20 | gnorm 1 | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1, pipeline's accumulation (no PP, probe) | same | same | same | same | same | same |
| pp2 | same | +6.50% | -1.97% | +0.67% | -4.74% | -14.09% |
| pp2 x vp2, cache on | same | -1.75% | +5.74% | same | -12.10% | +208.7% (one-step spike: 14.38 at 20, 3.45 at 19, 4.16 at 21) |
| pp2 x vp2, cache off | same | +0.27% | -0.01% | same | -24.21% | -8.05% |
| dp1, reversed accumulation | rerun queued (its first run died on the misplaced KDA switch) | | | | | |

512 tokens (2 x 256), reference dp1 (`12.614650` / `3.657440` / `2.948510`): pp2 same / +4.71% / -0.37%, grad norm same / +55.74% / +9.38%.

2048 tokens (dp2, 4 x 256 per rank), reference dp2 (`12.521100` / `3.221120` / `2.839810`; grad norm `16.375` / `7.0625` / `2.4844`):

| cell | loss 1 | 10 | 20 | gnorm 1 | 10 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2, pipeline's accumulation (no PP, probe) | same | +5.68% | +4.40% | same | -36.28% | +36.48% |
| dp2, reversed accumulation (no PP, probe) | +0.12% | +3.22% | +6.17% | +0.76% | -27.43% | +39.62% |
| dp2 x ep2 (no PP) | same | -2.23% | +4.56% | same | -28.32% | +18.87% |
| dp2 x pp2 | same | -0.60% | +0.23% | same | -29.20% | +6.92% |
| dp2 x pp2 x vp2, cache on | same | -0.48% | -5.52% | same | -23.89% | +1.26% |
| dp2 x pp2 x vp2, cache off | same | -0.99% | +0.26% | same | -20.80% | -4.40% |

The dp2 reversed-accumulation row moves step 1 by 0.12%, which a reordering of the same terms should not do at ulp level; not investigated. KDA-autotune-off pass (`KDA_NOAUTOTUNE=1`: `chunk_kda(..., autotune=False)`, which reaches Attention Gym's fused Triton path on this box; the kernels' own `@triton.autotune` still runs): every 1024-token cell -- dp1, the pipeline's accumulation, pp2, pp2 x vp2 cache on and off -- is identical to the autotune-on run on every printed step. On H100 the fused KDA autotuning is not a source of the pipeline-vs-dp1 difference.

## Why pp2's step-1 grad norm reads +0.67% on this box

It is one bf16 unit of the printed norm, not a gradient difference of that size. `clip_grad_norm_` computes the total norm in the gradients' dtype (bf16) and, under PP, sums the stages' squared norms in bf16. From the 8 x RTX 5060 Ti step-1 dumps of the same config (`/workspace/ppnum_0912/{dp1,pp2}` on that box): exact float64 total norms `18.68801` (dp1) and `18.68617` (pp2), 0.01% apart, both on the bf16 rounding midpoint `18.6875` between `18.625` and `18.75`; single-pass bf16 gives `18.625` for both, and pp2's per-stage bf16 norms `17.125` and `7.4062` combine to `18.625` there. Which side a cell prints is decided by bf16 rounding inside the norm; on this box pp2 lands on `18.75`. A float32 total norm (torchtitan #4135 with pytorch 194033) removes it.
