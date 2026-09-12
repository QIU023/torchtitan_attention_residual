# PP numerics, material for the reply to Tianyu's question (2026-09-12)

For the user, who replies personally. Nothing here is posted. English blocks are written to be pasted; the step-1 results are filled in from `phase13_k3like_48b_posttrain/PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md` once the dumps are compared.

The question (every round): "I'm not convinced that the numerics gap can be this large. Could you give simple examples of how accum order would change with this flag off / on vs. without PP? How do you prove it's not caused by bugs?"

What failed last time: the answer was long, argued at step 10 where the quantity is chaotic, and on the 5060 box `pp2` at step 10 sat outside the four-ordering band. The new answer is made at step 1, bitwise, with a prediction that could have failed.

## A. The simple example (runs on CPU, uses the PR's own store)

`matrix_scripts/pp_step1_0912/accum_order_example.py`. One block, read by three layers on each of four stages, laid out as pp2 x vp2 (rank 0 = stages 0 and 2, rank 1 = stages 1 and 3). With r_k the sum of stage k's reads:

| path | block gradient | bf16 vs no-PP | fp64 vs no-PP |
| --- | --- | --- | --- |
| no PP | one running fold over every read, top-down | reference | reference |
| PP, cache off | the same running fold, handed across each hop | bitwise | 0 |
| PP, cache on | stages 2 and 3 fold from zero; their subtotals join at stages 1 and 0 | 306 / 1024 elements differ | 0 |

One element spelled out: r0..r3 = -15.8125, 13.25, -0.186523, 0.882812; the no-PP fold gives -1.8125, the cached fold -1.875. The inputs cancel (r0 + r1 is about -2.6), so one bf16 ulp of an intermediate at magnitude 16 is 32 ulps of the result at magnitude 1.8. That is how an ulp-level reordering shows up as a large relative difference in some elements.

Paste-ready, plain torch (`matrix_scripts/pp_step1_0912/accum_order_minimal.py`):

```python
# The gradient of one block of the AttnRes stack under pp2 x vp2 (rank 0 = stages 0, 2; rank 1 =
# stages 1, 3). a[k] are the gradients of stage k's reads of the block, in backward order.
# Autograd adds each read onto whatever gradient already arrived: a running fold, top-down.
import torch

torch.manual_seed(0)
a = [[torch.randn(4096).mul(10.0 ** torch.randint(-2, 2, (4096,))).bfloat16() for _ in range(3)]
     for _ in range(4)]

def fold(start, reads):          # ((start + r0) + r1) + r2, the order autograd accumulates in
    g = start
    for r in reads:
        g = r.clone() if g is None else g + r
    return g

no_pp = fold(None, a[3] + a[2] + a[1] + a[0])            # one graph: every read, top-down

g = fold(None, a[3])                                      # cache off: each hop hands the running
for k in (2, 1, 0):                                       # sum back and the next stage keeps folding
    g = fold(g, a[k])
cache_off = g

d3, d2 = fold(None, a[3]), fold(None, a[2])               # cache on: stages 2, 3 read the block from
g1 = fold(None, a[1]) + d3                                # their rank's store and fold from zero; stage 1
cache_on = fold(g1 + d2, a[0])                            # adds rank 1's deposit, stage 0 rank 0's, then folds

print("cache off == no PP:", torch.equal(cache_off, no_pp))
print("cache on  == no PP:", torch.equal(cache_on, no_pp), "| elements that differ:",
      int((cache_on != no_pp).sum()), "of", no_pp.numel())
a64 = [[r.double() for r in s] for s in a]
exact = sum(r for s in a64 for r in s)
print("max |fold - exact sum| in float64 terms: no PP %.2e, cache on %.2e"
      % ((no_pp.double() - exact).abs().max(), (cache_on.double() - exact).abs().max()))
```

Output: cache off == no PP `True`; cache on == no PP `False`, 1568 of 4096 elements differ; the largest distance of either bf16 fold from the exact (float64) sum is 0.53 for no PP and 0.44 for cache on -- the cached sum is a different rounding of the same exact sum, not a worse one.

The thing the snippet gets right, and an earlier version of this note got wrong: autograd does not add up per-stage subtotals. It adds each read onto whatever gradient has already arrived, so the gradient coming back from later stages is where a stage's fold starts. Cache off keeps that running sum intact across every hop. Cache on lets a stage fold its own reads from zero and adds its subtotal later, at the stage that collects the deposit.

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

Both blocks change association: block 0 (read from the store by stages 2 and 3) and block 1 (read from the store by stage 3). A block's gradient reaches every parameter that produced it, and block 1 is layer 11's output, so the parameters that can move are layers 0-11, the embeddings and the vision encoder; layers 12-23 and the head cannot. (The first version of this prediction, committed in `08ffa11`, treated block 1's change as a commutation and predicted layers 6-11 bitwise; the dumps refuted it, below.)

## C. The one difference every PP cell has against dp1 that is not the transport

torch pipelining turns FSDP gradient sync off for every backward but the last (`stage.py` `backward_maybe_with_nosync`, then `perform_reduce_grad`). With sync off, FSDP2 keeps the unsharded gradient in the reduce dtype and adds each micro-batch there (`_fsdp_param.py` `to_accumulated_grad_if_needed` / `accumulate_unsharded_grad_if_needed`): four bf16 micro-batch gradients summed in float32, rounded once. dp1 with gradient accumulation syncs after every micro-batch, casts each reduced gradient back to the parameter dtype and adds in bf16, rounding after every add. Two terms agree (a two-term bf16 sum is exact in float32), which is why the earlier two-micro-batch profile was bitwise at the top of the model and the four-micro-batch one was not. The reference that removes this is dp1 with sync off until the last micro-batch (`NOSYNC_GA=1`).

## D. Step-1 bitwise test (to fill)

Predictions committed before the dumps were read: logbook `08ffa11`, `phase13_k3like_48b_posttrain/PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md`. Metric per tensor: bitwise or not, fraction of differing elements, median ulps among them, max ulps among elements above the tensor's median magnitude (a near-zero sign flip is tens of thousands of ordinal steps and says nothing), and relative L2 `||a - b|| / ||a||`.

- P1 (dp1 vs dp1, same seed checkpoint, same warm cache): **bitwise, 750 / 750 tensors.**
- P3 (dp1 vs pp2): 2 / 750 bitwise; differences start at the top (lm_head, norm, output_res), median 1-2 ulps per layer, relative L2 2.6e-3 at the top rising to 2.2e-2 at layer 0, the one step at layer 12 (a block start, 2.0e-2) and none at the stage boundary between 11 and 12 on the stage-0 side (6.2e-3 at layer 11 against 7.0e-3 at 13). Consistent with the accumulation-dtype difference of section C starting at the top; whether that is the whole of it is P4's test.
- P2 (cache off vs cache on, pp2 x vp2), as committed: stages 1-3 bitwise, only stage 0 differs. **Partly refuted.** Measured: layers 12-23 and the whole head (`lm_head`, `norm`, `output_res_norm`, `output_res_proj`) bitwise -- 346 of 750 tensors; every tensor of layers 0-11, the embeddings and the vision encoder differs, median 2-3 ulps, relative L2 6.4e-3 at layer 11 growing to 3.0e-2 at layer 0. The committed prediction had stage 1's layers 6-11 bitwise, on the argument that block 1's two sums differ only by commutation; that argument modelled a stage's reads as a pre-summed subtotal, which is not how autograd accumulates (section A). Corrected, the boundary of what moves is block 1's commit: everything that produces a block whose gradient passes through a deposit moves, and nothing downstream of it does. That is where the line falls in the dump, tensor for tensor.
- dp1 vs cache off and dp1 vs cache on: both 2 / 750 bitwise, the same top-down profile as dp1 vs pp2 (relative L2 2.6e-3 at the head, about 2.5e-2 at layer 0); the cached cell is not further from dp1 than the naive one (layer 0: 2.9e-2 against 2.4e-2).
- dp1 vs dp1 with matched accumulation (`NOSYNC_GA=1`, no pipeline anywhere): 2 / 750 bitwise, about 31% of every tensor's elements one ulp apart, relative L2 2.8e-3 flat across all layers -- one final rounding, float32-then-bf16 once against bf16 after every micro-batch, with nothing compounding. That is section C measured, and it is the same size as the whole dp1-vs-pipeline difference at the top of the model.
- P4 (pipeline cells vs dp1 with matched accumulation): the head becomes bitwise in pp2, cache off and cache on. A residual remains (20 / 750 bitwise), and it starts inside layer 23 -- the last layer, an MLA layer, on the last stage, whose backward runs before any gradient has crossed a stage boundary -- in the attention's tensors (`wq_a` 2,151 of 524,288 elements, `wq_b`, `wkv_a/b`, `q_norm`, `attention_norm`, `attention_res_*`), while layer 23's MoE and FFN, earlier in the backward, are bitwise. The residual is identical, number for number, in all three pipeline cells: it does not move with the stage boundaries or with the flag. `FlexAttention` compiles with `max_autotune` and `coordinate_descent_tuning`, so each process benchmarks and picks its own kernel config; P6 (running) pins it.
- P5 (one deposit dropped vs cache on): layers 12-23 and head bitwise; layers 0-11, embeddings and vision at relative L2 0.68-0.84, median about 190 ulps. The cache on/off reordering moves the same tensors by 6e-3 .. 3e-2, median 2-3 ulps. A real bug in the deposit path is two orders of magnitude above the reordering, in exactly the tensors the routing table says it can reach.
- P6 (flex compiled without autotune in every process): **refuted.** pp2 vs matched dp1 is 20 / 750 bitwise with the same profile, number for number, as the unpinned run, so flex's kernel choice is not the source. Narrowed: layer 23's `attention.wo` is bitwise in every pipeline cell, so flex's forward output and the gradient into it are identical, and the difference is born in flex's backward outputs (dq, dk, dv).
- P7 (activation checkpointing off in both, flex pinned, matched accumulation): **refuted.** pp2 vs dp1 is again 20 / 750 bitwise with the same numbers.

What the campaign establishes, and what it does not:
- Established: the cache flag moves exactly the parameters that produce a block whose gradient passes through a deposit (layers 0-11, embeddings, vision), leaves everything downstream bitwise, and moves them at the ulp scale of a reordered sum; a real bug in that path moves the same tensors two orders of magnitude further. The whole dp1-vs-pipeline difference at the head is the FSDP micro-batch accumulation dtype.
- Established: after matching the accumulation, one residual remains, identical in pp2, cache off and cache on, invariant to flex autotune and activation checkpointing, starting inside the last layer's attention backward on the last stage before any gradient crosses a stage boundary. It is not the transport.
- Not established: cache-off PP bitwise with dp1. The residual's own source is not identified.

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

The PR adds `kimi_k3_pp2_vp2` and `kimi_k3_pp8_vp4` to `tests/integration_tests/features.py` with `use_real_pg=True`. On every pull request the 8-GPU real-PG job runs `--test_scope=real_pg_required`, which selects exactly the `use_real_pg=True` entries, on the default CUDA runner `linux.g5.48xlarge.nvidia.gpu` (8 x A10G, SM86). Upstream KDA raises there. So both cells would fail on every PR, this one and everyone else's once merged. Upstream's own K3 cell lives in `b200.py` for this reason; the two cells belong there (the runner has 8 GPUs, so pp8 x vp4 fits). Fixed on `pp_review4` = `59f73634d` (2026-09-12, the two cells and their recipes moved to `b200.py`; definitions test and the split test's import updated; 22 CPU tests pass, pyrefly no delta, flake8 and ufmt clean). Not yet on the PR branch `k3_pp_text`.
