# Why the PP matrix spreads at step 10, and what the float32 grad norm changes (2026-09-04)

The maintainer's question on PR 4312 is whether the step-10 spread of the PP matrix (3.38 to 3.57 across cells that start bit-identical) is a numerical problem. This note reads the per-step data of every cell that ran, states the mechanism, corrects the earlier reading of the float32 grad-norm rows, and lists the two probes queued behind the CP matrix.

## 1. The answer

No numerical problem that the evidence can find. Every cell is bit-identical at step 1 (loss 12.51030, bf16 total norm 16.1250); under a float32 total norm the cells agree to 2e-4 (dp1 16.1631, pp2 x vp4 16.1661, pp4 x vp4 16.1646, pp8 x vp4 16.1656, whole-stack pp8 16.1649); the step-1 per-parameter comparison (REVIEW_ANSWERS 3.3, 918 parameters) has no group off (median 1.5e-4). The spread is the trajectory's sensitivity in this debug setup: the same single-GPU cell moves by 3.4% at step 10 when only the grad-norm precision and the compile cache change, dp1 vs dp2 moves by 6% in the CP matrix, and the PP cells land on both sides of dp1.

## 2. The per-step data (30-layer debugmodel, 4096 tokens per step, seed 42, `loss / grad_norm`)

bf16 total norm, the branch as it is (`d6b1ffe47` run tree, matrices of 2026-09-03):

| cell | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| dp1 | 12.51030 / 16.1250 | 10.06248 / 17.2500 | 7.39629 / 14.5625 | 6.71085 / 15.0625 | 6.03480 / 6.0000 | 5.82864 / 10.4375 | 4.31749 / 5.1250 | 4.07410 / 5.1250 | 3.62275 / 4.7500 | 3.49625 / 3.8125 |
| pp2 x vp4 | 12.51030 / 16.1250 | 10.03576 / 17.5000 | 7.45319 / 11.8125 | 6.86277 / 10.0000 | 6.27430 / 6.1562 | 5.32843 / 6.9375 | 4.19472 / 5.2812 | 3.93027 / 3.7812 | 3.43204 / 2.9531 | 3.38121 / 2.4375 |
| pp4 x vp4 | 12.51030 / 16.1250 | 10.06433 / 17.3750 | 7.44880 / 12.5000 | 7.09290 / 10.6875 | 6.10962 / 6.6875 | 5.33874 / 8.4375 | 4.33538 / 7.9688 | 4.14464 / 5.8125 | 3.69766 / 4.3438 | 3.57213 / 3.6094 |
| pp8 x vp4 | 12.51030 / 16.1250 | 10.03591 / 17.2500 | 7.40443 / 11.5000 | 6.92154 / 10.0000 | 5.88630 / 7.1875 | 5.11577 / 8.4375 | 4.31344 / 7.8438 | 4.04067 / 5.7188 | 3.57107 / 3.4062 | 3.47327 / 3.1719 |
| pp8 x vp4, whole stack every hop | 12.51030 / 16.1250 | 10.06008 / 17.7500 | 7.45462 / 14.1250 | 7.07708 / 10.5625 | 6.20520 / 8.0625 | 5.02867 / 6.9688 | 4.35108 / 6.1562 | 4.06160 / 5.1875 | 3.51100 / 3.2812 | 3.45668 / 2.6719 |
| pp2 x vp4, even split | 12.51030 / 16.1250 | 10.05909 / 18.1250 | 7.51238 / 12.5625 | 7.16834 / 11.5625 | 6.26646 / 6.4375 | 5.09264 / 8.8125 | 4.27323 / 8.4375 | 4.27502 / 6.4062 | 3.68203 / 4.5000 | 3.52185 / 3.2031 |

float32 total norm (PR 4135's reduction applied to the run tree, rerun complete on the rebased head `0e7cc5ea1`, `pp3gn` runs of 2026-09-04; dp1 and pp2 x vp4 reproduce the 2026-09-04 morning runs bitwise):

| cell | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| dp1 | 12.51030 / 16.1631 | 10.05712 / 17.4504 | 7.42095 / 13.8427 | 7.05484 / 15.1281 | 6.08853 / 6.4625 | 5.80434 / 11.1456 | 4.24013 / 5.5944 | 3.89144 / 3.6862 | 3.49225 / 4.0500 | 3.37903 / 3.0276 |
| pp2 x vp4 | 12.51030 / 16.1661 | 10.06197 / 17.8635 | 7.61926 / 12.4139 | 6.97623 / 9.4817 | 6.68817 / 7.4731 | 5.54767 / 6.6999 | 4.36542 / 4.4088 | 4.03798 / 3.9907 | 3.47844 / 2.5831 | 3.47015 / 2.5834 |
| pp4 x vp4 | 12.51030 / 16.1646 | 10.06421 / 17.5054 | 7.40290 / 13.7766 | 7.02590 / 12.8781 | 6.16285 / 7.2603 | 4.97187 / 5.3227 | 4.08200 / 5.5553 | 4.05442 / 5.2507 | 3.44678 / 2.7679 | 3.46001 / 3.3025 |
| pp8 x vp4 | 12.51030 / 16.1656 | 10.05837 / 17.5317 | 7.47695 / 12.1120 | 6.62815 / 10.6156 | 6.09824 / 9.1660 | 5.40108 / 7.1759 | 4.30592 / 4.4458 | 4.08569 / 3.6582 | 3.53297 / 2.5938 | 3.44950 / 2.2857 |
| pp8 x vp4, whole stack every hop | 12.51030 / 16.1649 | 10.04323 / 17.3154 | 7.41731 / 14.7474 | 6.91321 / 11.5722 | 6.00257 / 9.3402 | 5.25725 / 7.3292 | 4.36566 / 5.2428 | 4.29867 / 6.3518 | 3.73789 / 4.2400 | 3.58862 / 3.2897 |

Step-10 spread across the five cells: 3.379 to 3.589 (6.2%) under the float32 norm, 3.381 to 3.572 (5.6%) under bf16. The float32 step-1 norms of all five cells sit within 3e-4 of each other (16.1631 to 16.1661).

## 3. Reading it

- **Step 1 agrees everywhere.** The loss is the same 12.51030 in all nine runs. The bf16 total norm is the same 16.1250 in all six bf16 runs, and it cannot be otherwise: at magnitude 16 a bf16 value steps by 0.125, so the bf16 norm is quantised to 0.8% and the clip coefficient at step 1 is identical across cells. The float32 norm resolves the actual cross-cell difference: 2e-4, which is bf16 summation-order rounding of the gradients themselves.
- **Step 2 is already apart.** After a single optimizer step the losses differ by up to 2.7e-3 relative (10.036 vs 10.064). They fall into two clusters, 10.036 (pp2, pp8) and 10.06 (dp1, pp4, whole-stack pp8, even-split pp2, and all three float32-norm runs), which follow neither the transport nor the topology. They are deterministic: on the rebased head with fresh compile caches dp1 (twice), pp2 x vp4 and pp8 x vp4 reproduce every one of the ten steps bitwise, so the clusters are two summation orders of the same gradient, not the compile lottery seen on the CP tree under multi-rank autotune (9.53739 / 9.53178 / 9.54240 at step 2 across cache states of one cell).
- **The mechanism is Adam's first step.** With bias correction the first update is $lr \cdot g / (|g| + \epsilon)$, i.e. $\pm lr$ per element regardless of magnitude. Two runs whose gradients differ by rounding agree on the sign of every element except those whose gradient sits below the rounding noise; each such element moves by $2 \cdot lr$ in opposite directions. With a flip fraction $f$ the first update differs by $2\sqrt{f}$ of its norm: $f = 10^{-3}$ already means 6% of the first update. The queued census measures $f$ directly.
- **Why this setup amplifies it.** The debug flavor trains in bf16 end to end (`dtype="bfloat16"`, no FSDP master copy at dp1, so the AdamW states are bf16 too), lr 8e-4 with 2 warm-up steps, and the loss falls from 12.5 to 3.4 in ten steps: each step moves every weight by about $lr$, 2 to 4% of a 0.02-scale weight, on a steep descent. A first-step difference of a few percent of the update does not average out in ten such steps.
- **The float32 norm does not remove the spread, and the earlier reading was wrong.** The same dp1 cell goes from 3.49625 to 3.37903 at step 10 (-3.4%) when only the norm precision changes (the compile cache is not a factor at dp1: two fresh caches are bitwise); pp2 x vp4 goes from 3.38121 to 3.47015 (+2.6%). Under the float32 norm the five cells spread 6.2% at step 10, against 5.6% under bf16. REVIEW_ANSWERS 3.3 and the PR body's "two matrices" framing credited the bf16 total-norm grouping with the spread; the grouping does change the clip coefficient from step 2 on, but it is one bf16-level perturbation among several and removing it changes nothing about the size of the spread. Under Adam the clip coefficient is nearly scale-free anyway (only its step-to-step ratio matters).
- **Controls of the same size, with no PP in them.** dp1 vs dp2 in the CP matrix (multimodal debug flavor, 8192 tokens per step): 2.98077 vs 3.15823 at step 10, 6%; dp2 is not a pure re-partition, though: the loader shards documents by data-parallel rank, so dp2 reads a different batch composition (on the 30-layer flavor dp2's step 1 is 12.49684 against dp1's 12.51030, while every dp1-based PP cell matches dp1 to the digit), which makes the dp2 rows an upper bound. The clean single-perturbation control is the float32-norm dp1 run (3.4% at step 10 from the norm precision alone); the accumulation-order control (dp1 with 512-token micro-batches, same data) is queued. One CP cell on cold vs warm caches under multi-rank autotune: apart from step 2.
- **What a real bug would look like, and does not.** A wrong or missing stage gradient shows at step 1, before any amplification: the loss would not be bit-identical, the float32 norms would not agree to 2e-4, and one parameter group would stand out in the 918-parameter comparison (the tail there sits on 16-element norm weights, the bf16 signature). A systematic error would also put the PP cells on one side of dp1; they scatter on both (pp2 and pp8 below, pp4 and even-split pp2 above, whole-stack pp8 below).

## 4. Queued behind the CP matrix (`bridge_pp_all.sh`)

1. **Sign census** (`matrix_scripts/pp_probe_signs.sh`, `pp_step10_census.py`, `local_hacks/grad_tensor_dump_hack.py`): step-1 gradient tensors of dp1 on two fresh inductor caches, pp2 x vp4 and pp8 x vp4 on the rebased head `0e7cc5ea1`; per pair the flip fraction $f$ overall and per parameter group, where the flipped elements sit in magnitude, and $2\sqrt{f}$. The same-cell pair is the control the cross-cell pairs must not exceed by more than rounding.
2. **The two 10-step matrices on the rebased head** (`rebase_main_pp3.sh`), so the body's tables describe the branch it is on.
3. **100-step curves** (`rebase_main_pp3_curves100.sh`): dp1, pp2 x vp4, pp8 x vp4 and the whole-stack transport at the same seed; the reading is the mean loss over steps 51 to 100 and the band the curves cross in, against the same-cell control.

## 5. What changes in the PR text

The body's Results section drops the "two matrices, the float32 one removes the spread" framing: the bf16 table stays with the step-1 identity, the float32-norm agreement (2e-4), the mechanism in two sentences, the same-cell control, and the 100-step curves when they land. The reply to the numerics question is drafted in `Raising_PRs/PR_K3_PARALLELISM/REPLY_4312_2026-09-04.md` (round 2, "Is the step-10 spread a numerical problem").

## 6. The census (2026-09-04, `ppprobe8_0904_1053`, rebased head `0e7cc5ea1`)

| pair (step 1, 1,306,058,848 elements) | sha1-identical parameters | per-parameter norm rel diff median / p90 / max | sign flips | flipped elements below 1e-2 of their tensor's rms | $2\sqrt{f}$ |
|---|---|---|---|---|---|
| dp1 vs dp1, fresh compile caches | 918 of 918 | 0 / 0 / 0 | 0 | - | 0 |
| dp1 vs pp2 x vp4 | 2 of 918 | 2.0e-4 / 1.5e-3 / 1.3e-2 | 0.223% (2,916,679) | 84.6% | 9.45% |
| dp1 vs pp8 x vp4 | 2 of 918 | 2.0e-4 / 1.6e-3 / 1.3e-2 | 0.227% (2,965,397) | 84.2% | 9.53% |

Per group (dp1 vs pp2 / dp1 vs pp8; sign flips, element-wise relative L2 difference of the gradient): attention 0.387% / 0.391%, 1.19% / 1.22%; embedding 0 / 0 (its zero rows are exact), 1.26% / 1.26%; experts 0.251% / 0.256%, 1.07% / 1.09%; head 0.400% / 0.406%, 0.55% / 0.55%; norms 0.281% / 0.303%, 1.08% / 1.10%; other 0.119% / 0.122%, 1.10% / 1.11%; router 0.245% / 0.245%, 1.07% / 1.01%. The element-wise difference is the same ~1.1% in every group while the norms agree to 2e-4: bf16 rounding of the whole gradient, no group off, and 32 stages flip no more than 8. The ten-step trajectories of all four probe runs reproduce the 2026-09-03 rows bitwise.
