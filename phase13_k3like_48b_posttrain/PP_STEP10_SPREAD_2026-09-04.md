# Why the PP matrix spreads at step 10, and what the float32 grad norm changes (2026-09-04)

The maintainer's question on PR 4312 is whether the step-10 spread of the PP matrix (3.38 to 3.57 across cells that start bit-identical) is a numerical problem. This note reads the per-step data of every cell that ran, states the mechanism, corrects the earlier reading of the float32 grad-norm rows, and lists the two probes queued behind the CP matrix.

## 1. The answer

No numerical problem that the evidence can find. Every cell is bit-identical at step 1 (loss 12.51030, bf16 total norm 16.1250); under a float32 total norm the cells agree to 2e-4 (dp1 16.1631, pp2 x vp4 16.1661, pp4 x vp4 16.1646, pp8 x vp4 16.1656, naive pp8 16.1649); the step-1 per-parameter comparison (REVIEW_ANSWERS 3.3, 918 parameters) has no group off (median 1.5e-4). The spread is the trajectory's sensitivity in this debug setup: the same single-GPU cell moves by 3.4% at step 10 when only the grad-norm precision and the compile cache change, dp1 vs dp2 moves by 6% in the CP matrix, and the PP cells land on both sides of dp1.

## 2. The per-step data (30-layer debugmodel, 4096 tokens per step, seed 42, `loss / grad_norm`)

bf16 total norm, the branch as it is (`d6b1ffe47` run tree, matrices of 2026-09-03):

| cell | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| dp1 | 12.51030 / 16.1250 | 10.06248 / 17.2500 | 7.39629 / 14.5625 | 6.71085 / 15.0625 | 6.03480 / 6.0000 | 5.82864 / 10.4375 | 4.31749 / 5.1250 | 4.07410 / 5.1250 | 3.62275 / 4.7500 | 3.49625 / 3.8125 |
| pp2 x vp4 | 12.51030 / 16.1250 | 10.03576 / 17.5000 | 7.45319 / 11.8125 | 6.86277 / 10.0000 | 6.27430 / 6.1562 | 5.32843 / 6.9375 | 4.19472 / 5.2812 | 3.93027 / 3.7812 | 3.43204 / 2.9531 | 3.38121 / 2.4375 |
| pp4 x vp4 | 12.51030 / 16.1250 | 10.06433 / 17.3750 | 7.44880 / 12.5000 | 7.09290 / 10.6875 | 6.10962 / 6.6875 | 5.33874 / 8.4375 | 4.33538 / 7.9688 | 4.14464 / 5.8125 | 3.69766 / 4.3438 | 3.57213 / 3.6094 |
| pp8 x vp4 | 12.51030 / 16.1250 | 10.03591 / 17.2500 | 7.40443 / 11.5000 | 6.92154 / 10.0000 | 5.88630 / 7.1875 | 5.11577 / 8.4375 | 4.31344 / 7.8438 | 4.04067 / 5.7188 | 3.57107 / 3.4062 | 3.47327 / 3.1719 |
| pp8 x vp4, naive | 12.51030 / 16.1250 | 10.06008 / 17.7500 | 7.45462 / 14.1250 | 7.07708 / 10.5625 | 6.20520 / 8.0625 | 5.02867 / 6.9688 | 4.35108 / 6.1562 | 4.06160 / 5.1875 | 3.51100 / 3.2812 | 3.45668 / 2.6719 |
| pp2 x vp4, even split | 12.51030 / 16.1250 | 10.05909 / 18.1250 | 7.51238 / 12.5625 | 7.16834 / 11.5625 | 6.26646 / 6.4375 | 5.09264 / 8.8125 | 4.27323 / 8.4375 | 4.27502 / 6.4062 | 3.68203 / 4.5000 | 3.52185 / 3.2031 |

float32 total norm (PR 4135's reduction applied to the run tree, rerun complete on the rebased head `0e7cc5ea1`, `pp3gn` runs of 2026-09-04; dp1 and pp2 x vp4 reproduce the 2026-09-04 morning runs bitwise):

| cell | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| dp1 | 12.51030 / 16.1631 | 10.05712 / 17.4504 | 7.42095 / 13.8427 | 7.05484 / 15.1281 | 6.08853 / 6.4625 | 5.80434 / 11.1456 | 4.24013 / 5.5944 | 3.89144 / 3.6862 | 3.49225 / 4.0500 | 3.37903 / 3.0276 |
| pp2 x vp4 | 12.51030 / 16.1661 | 10.06197 / 17.8635 | 7.61926 / 12.4139 | 6.97623 / 9.4817 | 6.68817 / 7.4731 | 5.54767 / 6.6999 | 4.36542 / 4.4088 | 4.03798 / 3.9907 | 3.47844 / 2.5831 | 3.47015 / 2.5834 |
| pp4 x vp4 | 12.51030 / 16.1646 | 10.06421 / 17.5054 | 7.40290 / 13.7766 | 7.02590 / 12.8781 | 6.16285 / 7.2603 | 4.97187 / 5.3227 | 4.08200 / 5.5553 | 4.05442 / 5.2507 | 3.44678 / 2.7679 | 3.46001 / 3.3025 |
| pp8 x vp4 | 12.51030 / 16.1656 | 10.05837 / 17.5317 | 7.47695 / 12.1120 | 6.62815 / 10.6156 | 6.09824 / 9.1660 | 5.40108 / 7.1759 | 4.30592 / 4.4458 | 4.08569 / 3.6582 | 3.53297 / 2.5938 | 3.44950 / 2.2857 |
| pp8 x vp4, naive | 12.51030 / 16.1649 | 10.04323 / 17.3154 | 7.41731 / 14.7474 | 6.91321 / 11.5722 | 6.00257 / 9.3402 | 5.25725 / 7.3292 | 4.36566 / 5.2428 | 4.29867 / 6.3518 | 3.73789 / 4.2400 | 3.58862 / 3.2897 |

Step-10 spread across the five cells: 3.379 to 3.589 (6.2%) under the float32 norm, 3.381 to 3.572 (5.6%) under bf16. The float32 step-1 norms of all five cells sit within 3e-4 of each other (16.1631 to 16.1661).

## 3. Reading it

- **Step 1 agrees everywhere.** The loss is the same 12.51030 in all nine runs. The bf16 total norm is the same 16.1250 in all six bf16 runs, and it cannot be otherwise: at magnitude 16 a bf16 value steps by 0.125, so the bf16 norm is quantised to 0.8% and the clip coefficient at step 1 is identical across cells. The float32 norm resolves the actual cross-cell difference: 2e-4, which is bf16 summation-order rounding of the gradients themselves.
- **Step 2 is already apart.** After a single optimizer step the losses differ by up to 2.7e-3 relative (10.036 vs 10.064). They fall into two clusters, 10.036 (pp2, pp8) and 10.06 (dp1, pp4, naive pp8, even-split pp2, and all three float32-norm runs), which follow neither the transport nor the topology. They are deterministic: on the rebased head with fresh compile caches dp1 (twice), pp2 x vp4 and pp8 x vp4 reproduce every one of the ten steps bitwise, so the clusters are two summation orders of the same gradient, not the compile lottery seen on the CP tree under multi-rank autotune (9.53739 / 9.53178 / 9.54240 at step 2 across cache states of one cell).
- **The mechanism is Adam's first step.** With bias correction the first update is $lr \cdot g / (|g| + \epsilon)$, i.e. $\pm lr$ per element regardless of magnitude. Two runs whose gradients differ by rounding agree on the sign of every element except those whose gradient sits below the rounding noise; each such element moves by $2 \cdot lr$ in opposite directions. With a flip fraction $f$ the first update differs by $2\sqrt{f}$ of its norm: $f = 10^{-3}$ already means 6% of the first update. The queued census measures $f$ directly.
- **Why this setup amplifies it.** The debug flavor trains in bf16 end to end (`dtype="bfloat16"`, no FSDP master copy at dp1, so the AdamW states are bf16 too), lr 8e-4 with 2 warm-up steps, and the loss falls from 12.5 to 3.4 in ten steps: each step moves every weight by about $lr$, 2 to 4% of a 0.02-scale weight, on a steep descent. A first-step difference of a few percent of the update does not average out in ten such steps.
- **The float32 norm does not remove the spread, and the earlier reading was wrong.** The same dp1 cell goes from 3.49625 to 3.37903 at step 10 (-3.4%) when only the norm precision changes (the compile cache is not a factor at dp1: two fresh caches are bitwise); pp2 x vp4 goes from 3.38121 to 3.47015 (+2.6%). Under the float32 norm the five cells spread 6.2% at step 10, against 5.6% under bf16. REVIEW_ANSWERS 3.3 and the PR body's "two matrices" framing credited the bf16 total-norm grouping with the spread; the grouping does change the clip coefficient from step 2 on, but it is one bf16-level perturbation among several and removing it changes nothing about the size of the spread. Under Adam the clip coefficient is nearly scale-free anyway (only its step-to-step ratio matters).
- **Controls of the same size, with no PP in them.** dp1 vs dp2 in the CP matrix (multimodal debug flavor, 8192 tokens per step): 2.98077 vs 3.15823 at step 10, 6%; dp2 is not a pure re-partition, though: the loader shards documents by data-parallel rank, so dp2 reads a different batch composition (on the 30-layer flavor dp2's step 1 is 12.49684 against dp1's 12.51030, while every dp1-based PP cell matches dp1 to the digit), which makes the dp2 rows an upper bound. The clean single-perturbation control is the float32-norm dp1 run (3.4% at step 10 from the norm precision alone); the accumulation-order control (dp1 with 512-token micro-batches, same data) is queued. One CP cell on cold vs warm caches under multi-rank autotune: apart from step 2.
- **What a real bug would look like, and does not.** A wrong or missing stage gradient shows at step 1, before any amplification: the loss would not be bit-identical, the float32 norms would not agree to 2e-4, and one parameter group would stand out in the 918-parameter comparison (the tail there sits on 16-element norm weights, the bf16 signature). A systematic error would also put the PP cells on one side of dp1; they scatter on both (pp2 and pp8 below, pp4 and even-split pp2 above, naive pp8 below).

## 4. Queued behind the CP matrix (`bridge_pp_all.sh`)

1. **Sign census** (`matrix_scripts/pp_probe_signs.sh`, `pp_step10_census.py`, `local_hacks/grad_tensor_dump_hack.py`): step-1 gradient tensors of dp1 on two fresh inductor caches, pp2 x vp4 and pp8 x vp4 on the rebased head `0e7cc5ea1`; per pair the flip fraction $f$ overall and per parameter group, where the flipped elements sit in magnitude, and $2\sqrt{f}$. The same-cell pair is the control the cross-cell pairs must not exceed by more than rounding.
2. **The two 10-step matrices on the rebased head** (`rebase_main_pp3.sh`), so the body's tables describe the branch it is on.
3. **100-step curves** (`rebase_main_pp3_curves100.sh`): dp1, pp2 x vp4, pp8 x vp4 and the naive transport at the same seed; the reading is the mean loss over steps 51 to 100 and the band the curves cross in, against the same-cell control.

## 5. What changes in the PR text

The body's Results section drops the "two matrices, the float32 one removes the spread" framing: the bf16 table stays with the step-1 identity, the float32-norm agreement (2e-4), the mechanism in two sentences, the same-cell control, and the 100-step curves when they land. The reply to the numerics question is drafted in `Raising_PRs/PR_K3_PARALLELISM/REPLY_4312_2026-09-04.md` (round 2, "Is the step-10 spread a numerical problem").

## 6. The census (2026-09-04, `ppprobe8_0904_1053`, rebased head `0e7cc5ea1`)

| pair (step 1, 1,306,058,848 elements) | sha1-identical parameters | per-parameter norm rel diff median / p90 / max | sign flips | flipped elements below 1e-2 of their tensor's rms | $2\sqrt{f}$ |
|---|---|---|---|---|---|
| dp1 vs dp1, fresh compile caches | 918 of 918 | 0 / 0 / 0 | 0 | - | 0 |
| dp1 vs pp2 x vp4 | 2 of 918 | 2.0e-4 / 1.5e-3 / 1.3e-2 | 0.223% (2,916,679) | 84.6% | 9.45% |
| dp1 vs pp8 x vp4 | 2 of 918 | 2.0e-4 / 1.6e-3 / 1.3e-2 | 0.227% (2,965,397) | 84.2% | 9.53% |

Per group (dp1 vs pp2 / dp1 vs pp8; sign flips, element-wise relative L2 difference of the gradient): attention 0.387% / 0.391%, 1.19% / 1.22%; embedding 0 / 0 (its zero rows are exact), 1.26% / 1.26%; experts 0.251% / 0.256%, 1.07% / 1.09%; head 0.400% / 0.406%, 0.55% / 0.55%; norms 0.281% / 0.303%, 1.08% / 1.10%; other 0.119% / 0.122%, 1.10% / 1.11%; router 0.245% / 0.245%, 1.07% / 1.01%. The element-wise difference is the same ~1.1% in every group while the norms agree to 2e-4: bf16 rounding of the whole gradient, no group off, and 32 stages flip no more than 8. The ten-step trajectories of all four probe runs reproduce the 2026-09-03 rows bitwise.

## 7. 100-step curves on the debug set (2026-09-04, `pp3c100`, rebased head)

The debug flavor's data is one webdataset shard of 32 samples (`tests/assets/cc12m_test`), which 4096 tokens per step cycle through every other step: the run is a memorization curve, not a training curve. The schedule stretches with `training.steps` (warm-up 2, linear decay over the last 80%), so the ten-step values differ from the matrix.

| cell | step 10 | first step below 1.0 | step 50 | mean 51-100 | sd 51-100 | max 51-100 | step 100 |
|---|---|---|---|---|---|---|---|
| dp1 | 3.30192 | 31 | 0.234 | 0.090 | 0.052 | 0.250 | 0.04273 |
| pp2 x vp4 | 3.30048 | 26 | 0.157 | 0.075 | 0.025 | 0.137 | 0.04380 |
| pp8 x vp4 | 3.62183 | 41 | 0.525 | 0.114 | 0.085 | 0.439 | 0.04678 |
| pp8 x vp4, naive | 3.30103 | 30 | 0.192 | 0.082 | 0.028 | 0.161 | 0.04558 |

Reading: all four memorize the set to the same floor. The order of descent differs by up to 15 steps at the loss-1.0 crossing, and the 32-stage delta transport is the slowest while the same 32 stages with the naive transport are as fast as dp1; that is one seed of a memorization race, which amplifies any perturbation (the step-1 census shows the delta pp8 gradients are equal to dp1's up to the same bf16 rounding as pp2's, no group off). Two runs queued to read it properly: the same four cells on streamed cc12m (`pixparse/cc12m-wds`, no repeats), and dp1 / pp2 / pp8 at seed 43 on the debug set, which gives the seed-to-seed spread of one configuration to compare the pp8 lag against.

## 8. The census on the 33-layer model (`ppprobe33_0904`, `fe34932ee`)

| pair (step 1, 1,399,095,936 elements, 1002 parameters) | sha1-identical | per-parameter norm rel diff median / p90 / max | sign flips | below 1e-2 of rms | $2\sqrt{f}$ |
|---|---|---|---|---|---|
| dp1 vs pp2 x vp4 (4/5/5/4/4/4/4/3) | 2 | 2.0e-4 / 1.7e-3 / 1.2e-2 | 0.267% (3,737,997) | 80.4% | 10.34% |
| dp1 vs pp8 x vp4 (1/2/2/1...1/0) | 2 | 2.2e-4 / 1.5e-3 / 1.4e-2 | 0.277% (3,879,046) | 79.5% | 10.53% |

Per group (pp2 / pp8; flips, element-wise rel L2): attention 0.426% / 0.441%, 1.42% / 1.45%; embedding 0 / 0, 1.48% / 1.53%; experts 0.303% / 0.316%, 1.27% / 1.30%; head 0.443% / 0.460%, 0.54% / 0.55%; norms 0.335% / 0.340%, 1.28% / 1.34%; other 0.152% / 0.154%, 1.31% / 1.36%; router 0.291% / 0.303%, 1.25% / 1.31%. Same picture as the 30-layer model on splits nothing divides: bf16 rounding of the whole gradient, no group off, 32 uneven stages no worse than 8. The census script's last line had a shell quoting slip (a `#` in a sed replacement), so the comparisons were run by hand on the dumps the script left; the trajectories of the three probe runs match the matrix rows bitwise.

## 9. Pure data parallel and data x expert parallel at 1 / 2 / 4 / 8 (float32 grad norm, 33-layer model, `ladder_gn`, 2026-09-04 late evening)

Same seed and 4096 tokens per step. The loader shards the dataset by data-parallel rank, so the pure-dp rows change the batch composition with the degree (their step 1 differs); expert parallel is read against the same-dp row, which sees the same data.

| cell | step 1 | step 3 | step 10 | step 10 vs the same-dp row |
|---|---|---|---|---|
| dp1 | 12.41967 | 7.57490 | 3.34752 | - |
| dp2 | 12.40417 | 7.37116 | 3.30122 | - |
| dp4 | 12.41166 | 8.23808 | 3.26421 | - |
| dp8 | 12.39794 | 8.13134 | 3.28591 | - |
| dp2 x ep2 | 12.40257 | 7.45076 | 3.37020 | +2.1% |
| dp4 x ep2 | 12.41024 | 8.09069 | 3.32926 | +2.0% |
| dp4 x ep4 | 12.41024 | 8.04373 | 3.20992 | -1.7% |
| dp8 x ep2 | 12.39792 | 7.95019 | 3.25586 | -0.9% |
| dp8 x ep4 | 12.39792 | 7.80701 | 3.22198 | -1.9% |
| dp8 x ep8 | 12.39792 | 7.94250 | 3.15705 | -3.9% |

Reading: with the float32 norm in place, the pure-dp ladder spreads 2.5% at step 10 (3.264 to 3.348) with the data composition changing underneath it, and expert parallel moves the same-data row by -3.9% to +2.1%, in either direction, growing loosely with the EP degree. At a fixed dp degree the EP rows share step 1 to the digit across EP degrees (dp4 x ep2 = dp4 x ep4 = 12.41024, dp8 x ep2 = ep4 = ep8 = 12.39792) and sit 2e-5 to 1.6e-3 from the no-EP row: the expert dispatch changes the grouped GEMM's summation order once, whatever the degree. These are the same few percent the PP cells show on the same tree (the dp1 group 3.30 to 3.49), with no pipeline in them.

## 10. What a real gradient difference looks like (census set 2, `ppprobe33b`, 33-layer model)

The two "no-pipeline controls" queued earlier are not controls: FSDP dp2 reads another batch (the dataset is sharded by rank), and 512-token micro-batches change the sequences the collator builds (step 1 12.34637 against 12.41967), so both compare gradients of different data. That makes them the calibration for what a genuine difference looks like next to rounding:

| pair (step 1) | per-parameter norm rel diff median / p90 / max | sign flips | flipped elements below 1e-2 of rms | $2\sqrt{f}$ |
|---|---|---|---|---|
| dp1 vs pp2 x vp4 (rounding) | 2.0e-4 / 1.7e-3 / 1.2e-2 | 0.267% | 80% | 10.3% |
| dp1 vs pp8 x vp4 (rounding) | 2.2e-4 / 1.5e-3 / 1.4e-2 | 0.277% | 80% | 10.5% |
| dp1 vs dp2 (another batch) | 3.2e-2 / 1.1e-1 / 3.5e-1 | 22.99% | 6.3% | 96% |
| dp1 vs dp1 with 512-token micro-batches (other sequences) | 1.3e-1 / 2.0e-1 / 4.8e-1 | 23.75% | 5.9% | 97% |

A gradient that is actually different flips a quarter of the signs, mostly of elements that are not small, and moves the per-parameter norms by percent; the pipeline's flips are a hundred times fewer and sit in the near-zero elements. The clean same-data controls for the step-10 spread are the grad-norm precision alone (3.2% on dp1), expert parallel at fixed dp (-3.9% to +2.1%), and delta against naive on the same split (seven pairs, -2.8% to +3.9%).

## 11. The exact test, and what it found (2026-09-04 night, 33-layer model, `--training.dtype float32`)

Every run below loads the same seed checkpoint, dumps every parameter's step-1 gradient in float32 before the optimizer step, and is compared in float64 (relative L2 difference per parameter; "identical" = bitwise). Scripts: `pp33_probe_fp32.sh`, `pp33_probe_router.sh`, `pp33_probe_fp32x.sh` (float32 experts), `pp33_probe_alloc.sh`, `pp33_probe_bisect.sh`, `pp33_probe_amp.sh`, `pp33_probe_amp2.sh`; hacks in `local_hacks/` (`grad_tensor_dump_fp32_exit_hack.py`, `router_dump_hack.py`, `experts_fp32_hack.py`).

| comparison | median / p90 / max rel L2 diff | identical params | note |
|---|---|---|---|
| float32 model (bf16 grouped GEMM kept): dp1 vs pp2 x vp4 | 1.03e-2 / 1.80e-2 / 8.43e-2 | 8 | the same as bf16 |
| float32 model: dp1 vs pp8 x vp4 | 1.06e-2 / 1.85e-2 / 8.25e-2 | 8 | |
| expert routing, dp1 vs pp2 x vp4 (top-k ids per router and micro-batch, matched by content) | 0 of 448 routings differ, 0 of 114,688 tokens | | the forward is the same function |
| float32 model + float32 experts (per-expert matmul loop): dp1 vs pp2 x vp4 | 1.01e-2 / 1.75e-2 / 7.32e-2 | 7 | head 0, `output_res_*` 1e-11 |
| dp1 vs dp1 with another allocator layout (`expandable_segments`), float32 + float32 experts | 0 / 0 / 0 | 1002 | one GPU is bitwise reproducible |
| dp1 vs pp2 one stage per rank, 1F1B (one boundary, no store) | 6.61e-3 / 1.28e-2 / 4.58e-2 | 7 | by layer: 32: 3e-6, 31: 2e-4, 30: 1e-3, ..., 0: 1e-2, no step at the boundary (16/17) |
| dp1 vs pp2 x vp2 interleaved (store and deposits in play) | 6.84e-3 / 1.49e-2 / 7.13e-2 | 7 | same profile |
| dp1 vs pp2 x vp4 naive (no delta, no store) | 6.28e-3 / 1.24e-2 / 4.23e-2 | 7 | same profile |
| dp1 vs dp1 with the expert products rounded from float64 (a float32-rounding change of the FORWARD) | 6.53e-1 / 8.91e-1 / 1.44 | 0 | every layer ~0.6: the random-init router flips experts for a large share of tokens on a 1e-7 change of its input |
| dp1 vs dp1 with the expert BACKWARD rounded from float64, forward untouched (a float32-rounding change of the backward only) | 1.08e-2 / 1.85e-2 / 7.45e-2 | 10 | by layer: 32: 2e-4, 31: 2e-3, ..., 0: 2e-2, the pipeline's profile |

Reading. The pipeline's forward is exact (the head's gradient and the routing are bitwise the single GPU's). Its backward differs from the single GPU's by a float32-rounding-sized amount at the top (3e-6 at the last layer: the block stack the stage assembles is laid out and summed in another order) and that amount grows a hundredfold through the backward Jacobians of 33 layers, to 1e-2 at layer 0. The proof that this is amplification and not a transport error is the last row: a single GPU whose only change is the rounding of the expert backward shows the same magnitude and the same layer profile, while two single-GPU runs that differ in nothing arithmetic are bitwise. The bisection agrees: one boundary without a store, the interleaved schedule with the store, and the naive transport all give the same profile, and there is no step at the stage boundary. The forward-side control (second-to-last row) is a separate fact about this debug setup: at step 1 the router's scores are near-uniform, so a 1e-7 change of the forward reroutes a large share of tokens and changes the gradient by 60 percent; every cross-configuration comparison whose step-1 loss differs (CP, TP) carries that, and it is why step-1 identity, not closeness, is the bar the pipeline is held to.
