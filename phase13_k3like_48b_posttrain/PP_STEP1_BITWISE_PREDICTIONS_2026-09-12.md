# PP step-1 bitwise test: predictions, written before the dumps are read (2026-09-12 05:20)

Tree: `pp_review4` = `9f984333e` (upstream main `d9ca9e55a` + PR 4312), worktree `/tmp/wt_ppnum` with three probe hacks (KDA SM120 guard lift, `kimi_k3_debugmodel_pp_naive` alias, `GRAD_DUMP` of every parameter's step-1 gradient in its own dtype). Mutation and matched-accumulation hacks live in `/tmp/wt_ppmut` (`MUTATE=drop_deposit`, `NOSYNC_GA=1`). Patches and drivers: `matrix_scripts/pp_step1_0912/`.

Protocol: 8 x RTX 5060 Ti, main's 24-layer `debugmodel`, 1024 tokens per step as 4 x 256, seed 42, `--debug.deterministic`, one shared step-0 seed checkpoint (`.mx3_seeds_pp100/kimi_k3_debugmodel_fbfadaabdd4f`, the key of the 2026-09-11 H100 runs), every cell on a copy of dp1's warm inductor cache, per-rank Triton cache.

Splits (from the logs): pp2 = layers 0-11 | 12-23 + head. pp2 x vp2 = stage 0 (rank 0): vision, embeddings, layers 0-5; stage 1 (rank 1): 6-12; stage 2 (rank 0): 13-18; stage 3 (rank 1): 19-23 + head. Blocks commit at layers 0 and 12.

Routing (printed from `BlockLayoutTables`): cache on, block 0 is read from the rank store by stages 2 and 3 (one deposit each, collected by stage 0 and stage 1), block 1 by stage 3 (collected by stage 1); stage 3 receives no block on the wire. Cache off: no store, no deposits, every hop carries both blocks.

What changes the order of a sum, with r_k = what stage k's layers contribute to a block's gradient:
- dp1 and cache off: block 0's gradient is ((r3 + r2) + r1) + r0 -- autograd folds the reads top-down, and the wire hands over the partial fold unchanged.
- cache on: ((r1 + r3) + r2) + r0 for block 0. Block 1 is (r2 + r3) + r1 against (r3 + r2) + r1, equal because IEEE addition is commutative.
- micro-batches: torch pipelining turns FSDP gradient sync off for every backward but the last, so FSDP2 accumulates the four micro-batch gradients in the reduce dtype (float32) and rounds once; dp1 with gradient accumulation syncs after every micro-batch and accumulates in the parameter dtype (bf16), rounding after each add. Two terms agree (a two-term bf16 sum is exact in float32); three or more need not.

Predictions:
- P1. dp1 vs a second dp1: bitwise on all 750 tensors. (Observed before this file was written: 750/750.)
- P2. cache on vs cache off at pp2 x vp2: stages 1, 2 and 3 bitwise; differences only on stage 0 (vision encoder, embeddings, layers 0-5).
- P3. pp2 and cache-off pp2 x vp2 vs dp1: differences from the top of the model down (lm_head, norm, output_res), from the micro-batch accumulation dtype, not from the transport.
- P4. pp2 and cache-off pp2 x vp2 vs dp1 with `NOSYNC_GA=1` (dp1 accumulating like the pipeline): bitwise on all 750 tensors. A failure here is the one result that leaves an unexplained difference, and it is localised by the first differing tensor in backward order.
- P5. cache on with one deposit dropped vs cache on: stages 2 and 3 bitwise, stages 0 and 1 far outside the ulp scale of P2.

## Results (appended after reading the dumps; the predictions above are unedited)

- P1: held. 750 / 750 bitwise.
- P2: partly refuted. Layers 12-23 and the head bitwise (346 / 750); layers 0-11, embeddings and vision differ. The prediction had layers 6-11 bitwise because block 1's two sums were argued to differ only by commutation. Wrong model: autograd folds each read onto the gradient that already arrived, so the incoming gradient starts a stage's fold; with the cache on a stage folds from zero and its subtotal joins later. Block 1's association changes too, and block 1 is layer 11's output, so the line of what moves is block 1's commit -- which is where the dump draws it.
- P3: held in shape (differences from the head down, 1-2 ulps median); whether the accumulation dtype is all of it is P4.
- P4: refuted in its strict form, and located. Against dp1 with matched accumulation the head is now bitwise in every pipeline cell (it was not against plain dp1), so the head-level difference was the accumulation dtype. A residual remains: 20 / 750 bitwise, starting inside layer 23 (the last layer, an MLA layer) in the attention's own tensors (`wq_a` 2,151 elements, `wq_b`, `wkv_a/b`, `q_norm`, `attention_norm`, `attention_res_*`) while layer 23's MoE and FFN, earlier in the backward, are bitwise. It is identical, number for number, in pp2, pp2 x vp2 cache off and cache on, so it does not depend on where the stage boundaries fall or on the flag; layer 23's backward runs before any gradient has crossed a wire. Suspect: `FlexAttention` compiles with `max_autotune` and `coordinate_descent_tuning`, so each process benchmarks its own kernel config.
- P5: held. Dropped deposit vs cache on: layers 12-23 and head bitwise; layers 0-11, embeddings and vision at relative L2 0.68-0.84 (median about 190 ulps), against 6e-3 .. 3e-2 (median 2-3 ulps) for the cache on/off reordering in the same tensors.

## P6 (written before the run)

With `FLEX_NOAUTOTUNE=1` (flex compiled without max_autotune and coordinate descent, so the kernel is a fixed function of shape, dtype and device in every process), from `/tmp/wt_ppmut`, same seed checkpoint:
- P6a. pp2 vs dp1 with matched accumulation: bitwise on all 750 tensors.
- P6b. pp2 x vp2 cache off vs the same dp1: bitwise on all 750.
- P6c. pp2 x vp2 cache on vs the same dp1: layers 12-23 and head bitwise; layers 0-11, embeddings and vision differ at the ulp scale of P2.
If P6a fails, the first differing tensor in backward order names the next source (a KDA kernel with its own autotune is the next candidate).
- P6a: refuted. With flex compiled without autotune in both processes, pp2 vs matched dp1 is 20 / 750 bitwise with the same profile, number for number, as the unpinned run. Flex kernel selection is not the source. Narrowed further: layer 23's `attention.wo` is bitwise in every pipeline cell, so flex's forward output and the gradient into it are identical; the difference is born in flex's backward outputs (dq, dk, dv).

## P7 (written before the run)

Last week's forward dumps counted 380 attention-residual calls per step on one GPU against 452 under the pipeline: the two paths do not recompute the same regions under activation checkpointing, and flex's backward reads what the recomputed forward saved. With activation checkpointing off in both cells (flex pinned, matched accumulation):
- P7a. pp2 vs dp1: bitwise on all 750 tensors.
If it fails, the first differing tensor in backward order is again the pointer.
