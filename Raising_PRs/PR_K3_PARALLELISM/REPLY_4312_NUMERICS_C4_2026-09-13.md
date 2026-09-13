# Numerics note for PR 4312 (c4 re-run, fp32 grad norm), to post in the numerics thread (reply to 3985333653, `parallelize.py`)

H100 only: logs in `phase13_k3like_48b_posttrain/pp_h100x4_c4_logs_2026-09-13/gn32/`, scripts `run_pp_c4_gn32.sh` / `run_pp_c4_d2_gn32.sh`, the norm switch `gn_fp32_hack.py`. Reference = dp1 / dp2 accumulating in fp32 without a sync until the last micro-batch, as the pipeline does. The dp2 x pp2 row is `sh_d2_pp2` (100 steps on the warm cache shared with the reference-matching runs, `run_d2pp2_shared.log`); its first run on a cold cache of its own read 14.4183 at step 1, a rerun on a copy of that cache 14.4192.

--- PASTE BEGIN ---

Re-run on the same 4 x H100, now on `c4_test` as text-only rows (at 1024 tokens per step the reference memorised the original 32-sample debug set within the 100 steps), with the total grad norm taken in float32: with bf16 gradients the logged norm, and the clip factor that applies it every step here, depend on how the pipeline splits the parameters (https://github.com/pytorch/pytorch/pull/194033). Without the rank cache every pipeline cell is identical to the reference over 100 steps (the loss on every step, the logged grad norm on every step but one, where it differs in the fourth decimal); the cached cells differ because the cache adds a cached block's gradient contributions in another order.

4 x H100 PCIe, `kimi_k3_debugmodel` (24 layers), one seed checkpoint, four 256-token micro-batches per rank; the c4 flavor, the accumulation switch, the pp4 x vp4 stage count and the fp32 norm are local probe changes, not part of this PR ([c4 patch](https://github.com/QIU023/torchtitan_attention_residual/blob/74bd38d743b6f9353d4fb08e6946017db811e94e/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/pp4h_probe_c4.patch), [stage count](https://github.com/QIU023/torchtitan_attention_residual/blob/74bd38d743b6f9353d4fb08e6946017db811e94e/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/pp_stages_per_rank.patch), [fp32 norm](https://github.com/QIU023/torchtitan_attention_residual/blob/74bd38d743b6f9353d4fb08e6946017db811e94e/phase13_k3like_48b_posttrain/matrix_scripts/tp_h100_v2/gn_fp32_hack.py)).

1024 tokens per step:

| cell | loss, step 1 | step 10 | step 20 | step 100 | grad norm, step 1 | step 10 | step 20 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 (reference) | `12.609980` | `3.595760` | `2.998930` | `2.533960` | `16.9193` | `5.0429` | `2.1888` | `1.6014` |
| pp2 (all 100 steps identical) | `12.609980`<br>identical | `3.595760`<br>identical | `2.998930`<br>identical | `2.533960`<br>identical | `16.9193`<br>identical | `5.0429`<br>identical | `2.1888`<br>identical | `1.6014`<br>identical |
| pp2 x vp2, naive | `12.609980`<br>identical | `3.595760`<br>identical | `2.998930`<br>identical | `2.533960`<br>identical | `16.9193`<br>identical | `5.0429`<br>identical | `2.1888`<br>identical | `1.6014`<br>identical |
| pp4 x vp4, naive | `12.609980`<br>identical | `3.595760`<br>identical | `2.998930`<br>identical | `2.533960`<br>identical | `16.9193`<br>identical | `5.0429`<br>identical | `2.1888`<br>identical | `1.6014`<br>identical |
| pp2 x vp2, cached | `12.609980`<br>identical | `3.554240`<br>-1.15% | `3.055690`<br>+1.89% | `2.571310`<br>+1.47% | `16.9177`<br>-0.01% | `5.2631`<br>+4.37% | `2.9058`<br>+32.76% | `1.6073`<br>+0.37% |
| pp4 x vp4, cached | `12.609980`<br>identical | `3.309360`<br>-7.96% | `3.007760`<br>+0.29% | `2.554590`<br>+0.81% | `16.9167`<br>-0.02% | `3.5408`<br>-29.79% | `2.0286`<br>-7.32% | `1.5763`<br>-1.57% |

2048 tokens per step, dp2:

| cell | loss, step 1 | step 10 | step 20 | step 100 | grad norm, step 1 | step 10 | step 20 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 (reference) | `12.580740` | `3.576250` | `2.976110` | `2.420420` | `14.4170` | `12.4659` | `2.4576` | `1.0313` |
| dp2 x pp2 (all 100 steps identical) | `12.580740`<br>identical | `3.576250`<br>identical | `2.976110`<br>identical | `2.420420`<br>identical | `14.4170`<br>identical | `12.4659`<br>identical | `2.4576`<br>identical | `1.0313`<br>identical |
| dp2 x pp2 x vp2, naive (all 100 steps identical) | `12.580740`<br>identical | `3.576250`<br>identical | `2.976110`<br>identical | `2.420420`<br>identical | `14.4170`<br>identical | `12.4659`<br>identical | `2.4576`<br>identical | `1.0313`<br>identical |
| dp2 x pp2 x vp2, cached | `12.580740`<br>identical | `3.305220`<br>-7.58% | `2.949200`<br>-0.90% | `2.434300`<br>+0.57% | `14.4191`<br>+0.01% | `4.5317`<br>-63.65% | `2.6572`<br>+8.12% | `1.0893`<br>+5.62% |

--- PASTE END ---
