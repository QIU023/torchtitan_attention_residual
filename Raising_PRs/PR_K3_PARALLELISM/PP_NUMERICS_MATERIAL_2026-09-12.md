# PP numerics, material for the reply to Tianyu's question (2026-09-12)

For the user, who replies personally. Nothing here is posted. English blocks are written to be pasted; the step-1 results are filled in from `phase13_k3like_48b_posttrain/PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md` once the dumps are compared.

The question (every round): "I'm not convinced that the numerics gap can be this large. Could you give simple examples of how accum order would change with this flag off / on vs. without PP? How do you prove it's not caused by bugs?"

What failed last time: the answer was long, argued at step 10 where the quantity is chaotic, and on the 5060 box `pp2` at step 10 sat outside the four-ordering band. The new answer is made at step 1, bitwise, with a prediction that could have failed.

## A. The simple example (runs on CPU, uses the PR's own store)

`matrix_scripts/pp_step1_0912/accum_order_example.py`. One block, read by three layers on each of four stages, laid out as pp2 x vp2 (rank 0 = stages 0 and 2, rank 1 = stages 1 and 3). With r_k the sum of stage k's reads:

| path | block gradient | bf16 vs no-PP | fp64 vs no-PP |
| --- | --- | --- | --- |
| no PP | ((r3 + r2) + r1) + r0 | reference | reference |
| PP, cache off | ((r3 + r2) + r1) + r0 | bitwise | 0 |
| PP, cache on | ((r1 + r3) + r2) + r0 | 306 / 1024 elements differ | 0 |

One element spelled out: r0..r3 = -15.8125, 13.25, -0.186523, 0.882812; the no-PP fold gives -1.8125, the cached fold -1.875. The inputs cancel (r0 + r1 is about -2.6), so one bf16 ulp of an intermediate at magnitude 16 is 32 ulps of the result at magnitude 1.8. That is how an ulp-level reordering shows up as a large relative difference in some elements.

Why the cache reorders: with the cache on, a stage that already holds a block on its rank reads it from the rank store instead of the wire. Its gradient for that block is deposited in the store and added in by the stage that brought the block onto the rank, when that stage runs its backward. Without the cache the gradient goes back hop by hop, and each hop continues autograd's top-down fold.

## B. Where the flag acts on the debug model (printed from `BlockLayoutTables`)

24 layers, blocks commit at layers 0 and 12. pp2 x vp2: stage 0 = vision, embeddings, layers 0-5 (rank 0); stage 1 = 6-12 (rank 1); stage 2 = 13-18 (rank 0); stage 3 = 19-23 + head (rank 1).

| | cache on | cache off |
| --- | --- | --- |
| stage 0 sends | block 0 | block 0 |
| stage 1 sends | block 1 | blocks 0, 1 |
| stage 2 sends | nothing | blocks 0, 1 |
| store readers of block 0 | stages 2, 3 (deposits collected by 0 and 1) | none |
| store readers of block 1 | stage 3 (collected by 1) | none |

Only block 0's sum changes association. Block 1's differs only by commutation, (r2 + r3) against (r3 + r2), which is exact. Block 0's gradient feeds only stage 0's parameters, so the prediction is that cache on and cache off agree bitwise on stages 1-3 and differ only on stage 0.

## C. The one difference every PP cell has against dp1 that is not the transport

torch pipelining turns FSDP gradient sync off for every backward but the last (`stage.py` `backward_maybe_with_nosync`, then `perform_reduce_grad`). With sync off, FSDP2 keeps the unsharded gradient in the reduce dtype and adds each micro-batch there (`_fsdp_param.py` `to_accumulated_grad_if_needed` / `accumulate_unsharded_grad_if_needed`): four bf16 micro-batch gradients summed in float32, rounded once. dp1 with gradient accumulation syncs after every micro-batch, casts each reduced gradient back to the parameter dtype and adds in bf16, rounding after every add. Two terms agree (a two-term bf16 sum is exact in float32), which is why the earlier two-micro-batch profile was bitwise at the top of the model and the four-micro-batch one was not. The reference that removes this is dp1 with sync off until the last micro-batch (`NOSYNC_GA=1`).

## D. Step-1 bitwise test (to fill)

Predictions committed before the dumps were read: logbook `08ffa11`, `phase13_k3like_48b_posttrain/PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md`. Metric per tensor: bitwise or not, fraction of differing elements, median ulps among them, max ulps among elements above the tensor's median magnitude (a near-zero sign flip is tens of thousands of ordinal steps and says nothing), and relative L2 `||a - b|| / ||a||`.

- P1 (dp1 vs dp1, same seed checkpoint, same warm cache): **bitwise, 750 / 750 tensors.**
- P3 (dp1 vs pp2): 2 / 750 bitwise; differences start at the top (lm_head, norm, output_res), median 1-2 ulps per layer, relative L2 2.6e-3 at the top rising to 2.2e-2 at layer 0, the one step at layer 12 (a block start, 2.0e-2) and none at the stage boundary between 11 and 12 on the stage-0 side (6.2e-3 at layer 11 against 7.0e-3 at 13). Consistent with the accumulation-dtype difference of section C starting at the top; whether that is the whole of it is P4's test.
- P2, P4, P5: PENDING (the pp2 x vp2 cells were killed twice by the disk watchdog pruning their Triton cache; rerun with nested caches).

## E. Last night's tables, reorganised

Same five cells on both boxes, 24-layer debug model, 1024 tokens as 4 x 256, shared step-0 checkpoint. Loss at step 10 and 20 relative to each box's dp1:

| cell | 5060 Ti step 10 | 5060 Ti step 20 | H100 step 10 | H100 step 20 |
| --- | ---: | ---: | ---: | ---: |
| dp1 (absolute) | 3.297890 | 3.337080 | 3.114620 | 3.373330 |
| pp2 | +13.7% | +4.04% | +3.61% | +2.52% |
| pp2 x vp2, cache on | -3.72% | +2.67% | +1.17% | +0.712% |
| pp2 x vp2, cache off | +0.717% | +0.739% | +12.9% | +2.72% |
| dp1, accumulation reversed (no PP) | +1.58% | +1.79% | +4.27% | +2.31% |

Four orderings of the accumulation groups on the 5060 box span -6.64% .. +1.58% at step 10 and 0 .. +4.93% at step 20. The two boxes rank the cells in opposite orders at step 10 (the largest is pp2 on one, cache-off vp2 on the other), which is what single-sample scatter looks like and not what a systematic error looks like. Step 10 is where these numbers are weakest, so the argument does not rest on them: it rests on D.

## F. Hardware

Upstream runs K3 only in the B200 lane (`tests/integration_tests/b200.py`, workflow `integration_test_b200.yaml`, runner `linux.dgx.b200.8`, nightly or `ciflow/b200/*`), and upstream KDA refuses anything but SM100 / SM103. So the hardware Tianyu can run K3 on without our guard lift is B200; H100 needs the same lift we use here. Closest: 8 x B200. Second: 8 x H100 SXM (our 2 x H100 PCIe numbers are the nearest we have). The step-1 bitwise argument in D does not depend on the box: it compares two runs on the same box, and bitwise-or-not is the claim.

## G. A CI problem in PR 4312 found on the way (not numerics)

The PR adds `kimi_k3_pp2_vp2` and `kimi_k3_pp8_vp4` to `tests/integration_tests/features.py` with `use_real_pg=True`. On every pull request the 8-GPU real-PG job runs `--test_scope=real_pg_required`, which selects exactly the `use_real_pg=True` entries, on the default CUDA runner `linux.g5.48xlarge.nvidia.gpu` (8 x A10G, SM86). Upstream KDA raises there. So both cells would fail on every PR, this one and everyone else's once merged. Upstream's own K3 cell lives in `b200.py` for this reason; the two cells belong there (the runner has 8 GPUs, so pp8 x vp4 fits). Not changed yet: it is a PR edit and the user's call.
