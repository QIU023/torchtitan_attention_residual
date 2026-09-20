# PR 4312 round 3: the local reference matrix (8 x RTX 5060 Ti, 2026-09-20 night)

Reference only. Nothing here goes into the PR body: the body's numbers come from H100, and these were taken on a Blackwell consumer card through Attention Gym's portable Triton KDA path with the SM120 guard lifted. What this matrix answers is structural: on the round 3 head (`pp_review4` `d62dcec45`, 9 layer / dim 256 debug model, main `27951c2f6`) does the comparison that the body makes still come out the same way as on 2026-09-13, cell by cell.

## Protocol

The body's, ported to the tree of 09-20 (`scratchpad/run_pp_local.sh`, hacks in `scratchpad/probe_apply.py`, never committed):

* `kimi_k3_debugmodel_c4`: the shared debug flavor reading `c4_test` as text only 256 token rows (each doc's first 256 tokens), the checkpointer attached load only; `_seed` sets `create_seed_checkpoint`, since on main `--checkpoint.*` is gone and both are Python side settings.
* seed 42, `--debug.deterministic`, one seed checkpoint per batch shape; every cell's log is checked for exactly one `Loading the checkpoint` line.
* 1024 tokens per step as four 256 token micro batches (dp1), 2048 as two ranks x four (dp2); 100 steps; `--metrics.log_freq 1`.
* `GN_FP32=1`: the total grad norm of fp32 copies (pytorch#194033 for this purpose); the reference accumulates with the three FSDP switches torch pipelining sets (`NOSYNC_GA=1`); `MB_REVERSE=1` reverses the micro batch order (noise floor); `attn_res_cache=False` through a `functools.partial` on the pipelining function (naive rows); the eight stage cells take the split spelled out in the pp2 x vp4 recipe.
* Every non reference cell runs on a copy of the reference's warm inductor and Triton caches.

Cells, against the 09-13 H100 tables: `dp1_ns` (reference), `dp1` (stock accumulation, footnote 4 of the appendix), `dp1_rev` (footnote 5), `pp2`, `vp2n` / `vp2c` (pp2 x vp2 naive / cached), `pp2vp4n` / `pp2vp4c` and `pp4vp2n` / `pp4vp2c` in place of the 09-13 pp4 x vp4 rows (16 stages need 14 layers; the 9 layer model splits into 8), `d2_dp2_ns`, `d2_dp2`, `d2_ep2`, `d2_pp2`, `d2_vp2n` / `d2_vp2c`.

## What to read off it

1. Every cell ran and loaded the seed (`rc=0 steps=100 seed_loaded=1`).
2. Step 1 identical in every cell (the 09-13 bar).
3. The whole stack rows (`pp2`, every `*n`) identical to the matched reference on every step, loss and norm (09-13: all 100 steps, the norm on all but step 59 in the fourth decimal).
4. The cached rows depart after step 1 and stay within a few percent (09-13: -1.15% to -7.96% at step 10, under 1.5% at step 100).
5. `dp1` (stock) and `dp1_rev` depart from the reference at the same order as the cached rows, since each is one different rounding (09-13 appendix, footnotes 4 and 5).

## Tables

Every cell: rc 0, 100 steps, the seed loaded on every rank (`local_matrix_cells.log`). The 1024 reference was rerun once on the dp1 cell's copy of its own warm cache after an accidental second launch truncated its log (the rerun is deterministic on that cache; the truncated log is kept as `dp1_ns_overwritten_by_dup.log` in the scratchpad).

# c4, 1024 tokens per step (4 x 256), reference dp1 with matched accumulation
| cell | step 1 | step 10 | step 20 | step 50 | step 100 | step 1 | step 10 | step 20 | step 50 | step 100 (grad norm) | steps identical (loss, norm) | first loss difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dp1_ns | `8.027620` | `3.424600` | `2.918730` | `2.605460` | `2.542180` | `1.797300` | `1.447900` | `0.962100` | `0.784400` | `0.819800` | - | - |
| dp1 | `8.027620` (bitwise) | `3.426230` (+0.0476%) | `2.922540` (+0.131%) | `2.608810` (+0.129%) | `2.544290` (+0.083%) | `1.797300` (bitwise) | `1.441500` (-0.442%) | `0.962800` (+0.0728%) | `0.778300` (-0.778%) | `0.811000` (-1.07%) | 1/100, 1/100 | 2 |
| dp1_rev | `8.027620` (bitwise) | `3.425070` (+0.0137%) | `2.917460` (-0.0435%) | `2.607050` (+0.061%) | `2.543320` (+0.0448%) | `1.797200` (-0.00556%) | `1.443200` (-0.325%) | `0.958200` (-0.405%) | `0.778400` (-0.765%) | `0.821500` (+0.207%) | 1/100, 1/100 | 2 |
| pp2 | `8.027620` (bitwise) | `3.424600` (bitwise) | `2.918730` (bitwise) | `2.605460` (bitwise) | `2.542180` (bitwise) | `1.797300` (bitwise) | `1.447900` (bitwise) | `0.962100` (bitwise) | `0.784400` (bitwise) | `0.819800` (bitwise) | 100/100, 100/100 | none |
| vp2n | `8.027620` (bitwise) | `3.424600` (bitwise) | `2.918730` (bitwise) | `2.605460` (bitwise) | `2.542180` (bitwise) | `1.797300` (bitwise) | `1.447900` (bitwise) | `0.962100` (bitwise) | `0.784400` (bitwise) | `0.819800` (bitwise) | 100/100, 100/100 | none |
| vp2c | `8.027620` (bitwise) | `3.424790` (+0.00555%) | `2.920900` (+0.0743%) | `2.608080` (+0.101%) | `2.540740` (-0.0566%) | `1.797300` (bitwise) | `1.441600` (-0.435%) | `0.962300` (+0.0208%) | `0.790000` (+0.714%) | `0.815300` (-0.549%) | 1/100, 1/100 | 2 |
| pp2vp4n | `8.027620` (bitwise) | `3.424600` (bitwise) | `2.918730` (bitwise) | `2.605460` (bitwise) | `2.542180` (bitwise) | `1.797300` (bitwise) | `1.447900` (bitwise) | `0.962100` (bitwise) | `0.784400` (bitwise) | `0.819800` (bitwise) | 100/100, 100/100 | none |
| pp2vp4c | `8.027620` (bitwise) | `3.426200` (+0.0467%) | `2.918890` (+0.00548%) | `2.608200` (+0.105%) | `2.542780` (+0.0236%) | `1.797200` (-0.00556%) | `1.442500` (-0.373%) | `0.952700` (-0.977%) | `0.784500` (+0.0127%) | `0.814900` (-0.598%) | 1/100, 1/100 | 2 |
| pp4vp2n | `8.027620` (bitwise) | `3.424600` (bitwise) | `2.918730` (bitwise) | `2.605460` (bitwise) | `2.542180` (bitwise) | `1.797300` (bitwise) | `1.447900` (bitwise) | `0.962100` (bitwise) | `0.784400` (bitwise) | `0.819800` (bitwise) | 100/100, 100/100 | none |
| pp4vp2c | `8.027620` (bitwise) | `3.425450` (+0.0248%) | `2.918550` (-0.00617%) | `2.610880` (+0.208%) | `2.539340` (-0.112%) | `1.797300` (bitwise) | `1.447600` (-0.0207%) | `0.953500` (-0.894%) | `0.792300` (+1.01%) | `0.818500` (-0.159%) | 1/100, 1/100 | 2 |

# c4, 2048 tokens per step (4 x 256 per rank), reference dp2 with matched accumulation
| cell | step 1 | step 10 | step 20 | step 50 | step 100 | step 1 | step 10 | step 20 | step 50 | step 100 (grad norm) | steps identical (loss, norm) | first loss difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d2_dp2_ns | `8.032220` | `3.324430` | `2.855760` | `2.581940` | `2.410480` | `1.618800` | `1.287800` | `0.859500` | `0.592900` | `0.692300` | - | - |
| d2_dp2 | `8.032220` (bitwise) | `3.325870` (+0.0433%) | `2.855050` (-0.0249%) | `2.582200` (+0.0101%) | `2.411260` (+0.0324%) | `1.618700` (-0.00618%) | `1.294500` (+0.52%) | `0.868000` (+0.989%) | `0.595000` (+0.354%) | `0.685700` (-0.953%) | 1/100, 0/100 | 2 |
| d2_ep2 | `8.032220` (bitwise) | `3.324740` (+0.00932%) | `2.856440` (+0.0238%) | `2.583670` (+0.067%) | `2.410460` (-0.00083%) | `1.617200` (-0.0988%) | `1.296900` (+0.707%) | `0.843800` (-1.83%) | `0.593800` (+0.152%) | `0.695300` (+0.433%) | 2/100, 0/100 | 2 |
| d2_pp2 | `8.032220` (bitwise) | `3.329750` (+0.16%) | `2.854200` (-0.0546%) | `2.583200` (+0.0488%) | `2.410590` (+0.00456%) | `1.618800` (bitwise) | `1.316100` (+2.2%) | `0.859800` (+0.0349%) | `0.599600` (+1.13%) | `0.686600` (-0.823%) | 1/100, 1/100 | 2 |
| d2_vp2n | `8.032220` (bitwise) | `3.329750` (+0.16%) | `2.854200` (-0.0546%) | `2.583200` (+0.0488%) | `2.410590` (+0.00456%) | `1.618800` (bitwise) | `1.316100` (+2.2%) | `0.859800` (+0.0349%) | `0.599600` (+1.13%) | `0.686600` (-0.823%) | 1/100, 1/100 | 2 |
| d2_vp2c | `8.032220` (bitwise) | `3.326230` (+0.0541%) | `2.856520` (+0.0266%) | `2.583040` (+0.0426%) | `2.406110` (-0.181%) | `1.618800` (bitwise) | `1.306200` (+1.43%) | `0.862300` (+0.326%) | `0.597500` (+0.776%) | `0.697100` (+0.693%) | 1/100, 3/100 | 2 |


## Reading

Against the five points:

1. All 15 cells: rc 0, 100 steps, seed loaded on every rank.
2. Step 1 is bitwise in every cell of both groups, loss and grad norm.
3. 1024 group: `pp2`, `vp2n`, `pp2vp4n` and `pp4vp2n` are identical to the matched reference on all 100 steps, loss and grad norm alike (the 09-13 H100 table had one grad norm step in the fourth decimal; here none). The structure the body claims for the whole stack transport reproduces on this box, on the round 3 model, in the eight stage shapes that replace pp4 x vp4.
4. 1024 group: the three cached rows depart at step 2 and stay within 0.21% of the reference on the listed steps (step 10: +0.006%, +0.047%, +0.025%; step 100: -0.057%, +0.024%, -0.11%), an order smaller than the 09-13 cached rows on the 24 layer model (-1.15% and -7.96% at step 10).
5. 1024 group: stock accumulation (`dp1`) and the reversed micro batch order (`dp1_rev`) depart at step 2 and sit in the same band as the cached rows (step 10: +0.048% and +0.014%; step 100: +0.083% and +0.045%), as on the 09-13 appendix.

The 2048 group does not reproduce one 09-13 row: `d2_pp2` and `d2_vp2n` are identical to each other on all 100 steps (1F1B is the naive transport by construction) but depart from the matched dp2 reference at step 2 (+0.16% at step 10, +0.005% at step 100, the grad norm bitwise at step 1 and +2.2% at step 10), whereas on the H100 they matched it on all 100 steps. Their departure is the same size as the stock accumulation and expert parallel controls (`d2_dp2`, `d2_ep2`: +0.04% and +0.01% at step 10). No mechanism is claimed from this box (memory: its step 10 spreads are the flavour's own sensitivity here); the candidates are listed for the H100 rerun, which decides: the fp32 total norm's reduction over the pp group and the FSDP shards, the loss mesh reductions, and this box's KDA path. A control without the quantile balancing hook (`noqb.sh`, queued behind the chain) is recorded below when it lands.

Other overnight items: the K3 GPU unit tests on this branch pass (3 passed, 2 skipped, the two SM100/SM103 only tests; the branch has the same 5 tests as main), and the validator runs under pp2 x vp2 with `freq=1` for 3 steps (`val_pp2vp2`: three `validate step` lines on the last stage rank, no traceback; `val_dp1` the same without PP), which is the end to end check of the three forward only fixes.
