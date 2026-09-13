# PR 4312 on c4_test, 4 x H100 PCIe (2026-09-13)

Tree `pp_review4` = `dbc425403` + `pp4h_probe_c4.patch` (uncommitted on the box); kit `matrix_scripts/tp_h100_v2/run_pp_c4.sh`. torch `2.15.0.dev20260906+cu130`, Attention Gym `499404b`, spmd_types 0.2.5, torch_remat 0.2.0 (the 2026-09-12 versions). Flavor `kimi_k3_debugmodel_c4`: `c4_test` as text-only rows, each doc's first 256 tokens (2000 rows, ~0.5M tokens), no images (the vision tower does not run). One seed checkpoint per batch shape, 100 steps, seed 42, deterministic. Each cell gives the raw value and, beneath it, the change against the table's reference.

The reference does not memorise the rows: dp1's loss 1:12.6100 5:5.9629 10:3.4220 20:3.0227 30:3.0343 40:2.7667 50:2.6687 60:2.6298 70:2.7630 80:2.6357 90:2.4765 100:2.5366. No collapse toward zero (cc12m-test reached 0.19 by step 100); the small rises at 30 / 70 / 100 are step noise. Which of steps 50 / 100 enter the PR body is the user's call under the numerics-table rule.

## 1024 tokens per step (4 x 256), reference dp1

| cell | loss step 1 | loss step 10 | loss step 20 | loss step 50 | loss step 100 | grad norm step 1 | grad norm step 10 | grad norm step 20 | grad norm step 50 | grad norm step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.60998` | `3.42196` | `3.02271` | `2.66869` | `2.53660` | `16.8750` | `5.0625` | `2.1406` | `1.4609` | `1.5703` |
| dp1, accumulation reversed (noise floor, no PP) | `12.60998`<br>same | `3.38945`<br>-0.95% | `3.06651`<br>+1.45% | `2.71110`<br>+1.59% | `2.56357`<br>+1.06% | `16.8750`<br>0% | `4.6875`<br>-7.41% | `2.9531`<br>+37.96% | `1.4141`<br>-3.20% | `1.6797`<br>+6.97% |
| pp2 | `12.60998`<br>same | `3.33437`<br>-2.56% | `2.98862`<br>-1.13% | `2.71135`<br>+1.60% | `2.56287`<br>+1.04% | `16.8750`<br>0% | `3.2969`<br>-34.88% | `2.2188`<br>+3.65% | `1.6094`<br>+10.16% | `1.5312`<br>-2.49% |
| pp2 x vp2, cache on | `12.60998`<br>same | `3.31003`<br>-3.27% | `3.03207`<br>+0.31% | `2.69393`<br>+0.95% | `2.56581`<br>+1.15% | `16.8750`<br>0% | `3.2812`<br>-35.19% | `2.9375`<br>+37.23% | `1.6328`<br>+11.77% | `1.5781`<br>+0.50% |
| pp2 x vp2, cache off | `12.60998`<br>same | `3.33143`<br>-2.65% | `2.99671`<br>-0.86% | `2.71330`<br>+1.67% | `2.57117`<br>+1.36% | `16.8750`<br>0% | `3.3125`<br>-34.57% | `2.4062`<br>+12.41% | `1.7578`<br>+20.32% | `1.5547`<br>-0.99% |

## 2048 tokens per step (4 x 256 per rank), reference dp2

| cell | loss step 1 | loss step 10 | loss step 20 | loss step 50 | loss step 100 | grad norm step 1 | grad norm step 10 | grad norm step 20 | grad norm step 50 | grad norm step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 | `12.58074` | `3.30783` | `2.93077` | `2.66828` | `2.42670` | `14.4375` | `4.5312` | `2.3438` | `1.8125` | `1.0312` |
| dp2 x ep2 (noise floor, no PP) | `12.58074`<br>same | `3.26932`<br>-1.16% | `2.96681`<br>+1.23% | `2.65745`<br>-0.41% | `2.42051`<br>-0.26% | `14.4375`<br>0% | `4.4375`<br>-2.07% | `2.2500`<br>-4.00% | `1.7422`<br>-3.88% | `1.0312`<br>0% |
| dp2 x pp2 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| dp2 x pp2 x vp2, cache on | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| dp2 x pp2 x vp2, cache off | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Logs: `c4_*.log` here, `run_pp_c4.log` the driver. `c4_dp1_ns.log` (dp1 accumulating like the pipeline, NOSYNC_GA) and the dp2 pipeline cells are appended when the run finishes.
