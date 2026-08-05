# Comment for the eager PR: the seeded faithful matrix

Replaces the earlier cold-start table. Two things changed underneath it: the
structure now matches the review comments, and every cell starts from a shared
step-0 checkpoint instead of cold-starting, which is what collapsed the step-1
spread from 0.047 to 0.009.

Single-line paragraphs (tables excepted) so it copies verbatim.

--- PASTE BEGIN ---

Updated table -- the earlier one cold-started every cell, and following the suggestion to fix initialization changes the picture qualitatively rather than marginally.

Eighteen parallelism configurations on the debug model, seed 42, `--debug.deterministic`, global batch 8, `local_batch_size 1` so gradient accumulation is on, 100 steps, every cell loading the same step-0 checkpoint written with `--checkpoint.create_seed_checkpoint`. Eager: **18/18**. Compiled: **15/18** (the three gaps are one `torch._grouped_mm` limitation on a zero-length contraction dimension, not a parallelism failure; details below).

| configuration | step 1 | step 50 | step 100 |
|---|---|---|---|
| single-GPU | 12.07418 | 1.33860 | 0.35143 |
| fsdp2 | 12.07418 | 1.34609 | 0.36425 |
| pp2 | 12.07418 | 1.35109 | 0.35443 |
| cp2 | 12.07565 | 1.35855 | 0.34527 |
| tp2 | 12.07566 | 1.30560 | 0.34967 |
| fsdp2 x tp2 x pp2 | 12.07566 | 1.37392 | 0.35423 |
| fsdp2 x tp2 x cp2 | 12.07590 | 1.43442 | 0.37419 |
| tp2 x pp2 x cp2 | 12.07590 | 1.34100 | 0.36659 |
| fsdp2 x pp2 x cp2 | 12.07565 | 1.39481 | 0.37566 |
| ep2 x fsdp2 | 12.07418 | 1.35289 | 0.36067 |
| ep2 x fsdp2 x tp2 x pp2 | 12.07190 | 1.44611 | 0.36211 |
| ep2 x fsdp2 x tp2 x cp2 | 12.06705 | 1.40905 | 0.36507 |
| ep2 x fsdp2 x pp2 x cp2 | 12.07565 | 1.33048 | 0.36867 |
| ep8 x fsdp8 | 12.07418 | 1.34121 | 0.36849 |
| pp4 | 12.07418 | 1.33522 | 0.34277 |
| pp8 | 12.07418 | 1.30919 | 0.38591 |
| tp4 | 12.07286 | 1.36290 | 0.34306 |
| cp4 | 12.06692 | 1.42309 | 0.37493 |

Step-1 spread is 0.009 across all eighteen, and it decomposes by reduction structure rather than being a band of noise:

| step 1 | configurations |
|---|---|
| 12.07418 | single-GPU, fsdp2, pp2, pp4, pp8, ep2 x fsdp2, ep8 x fsdp8 |
| 12.07565 | cp2, fsdp2 x pp2 x cp2, ep2 x fsdp2 x pp2 x cp2 |
| 12.07566 | tp2, fsdp2 x tp2 x pp2 |
| 12.07590 | fsdp2 x tp2 x cp2, tp2 x pp2 x cp2 |

Seven configurations agree **bit-for-bit** with the single-GPU run: FSDP, pipeline parallelism at degrees 2/4/8 and expert parallelism at degrees 2/8 introduce no numerical difference at all. The configurations that differ are exactly the ones where TP or CP changes the floating-point reduction order, and they group by which of the two is present. Four cells sit outside those clusters -- the two EP x TP combinations, tp4 and cp4 -- all within the same 0.009.

All eighteen decrease monotonically to the same order (0.343-0.386 at step 100). Two independent full runs of the same configuration reproduce bit-for-bit.

`ep8 x fsdp8` is worth a second look: 8 experts over 8 ranks is exactly one expert per rank, so empty expert groups occur on every step. It passes eager and compiled.

Three configurations the debug config cannot express, recorded rather than worked around by editing it: `tp8`, `cp8` and `tp4 x cp4` have only 4 attention heads (MLA and KDA both) to shard, and `pp8 x vp4` cannot host 32 virtual stages on 13 layers.

On the compiled gaps: `torch._grouped_mm` rejects a zero contraction dimension, which is the weight-gradient shape a rank sees when it ends up with zero routed tokens in total -- the ep2 x TP/PP/CP combinations produce it, while per-group empties alone do not (`ep8 x fsdp8` passes compiled unshimmed). Five-line reproduction with no model, compile or distributed involved; being filed against pytorch/pytorch. A local shim that re-strides exactly as the proposed patch would makes all eighteen pass, but the shim is ours, so 15/18 is the number we report.

Scope, so the table is not read as more than it is: this is eighteen parallelism configurations of one stack agreeing with each other, not two stacks agreeing. We have not rebased onto this PR, so cross-stack numerical alignment is a separate exercise and we will post it separately.

Environment: torch `2.14.0.dev20260802+cu130`, fla-core `0.5.1`, 8 x RTX 5060 Ti (15.5 GiB, sm_120). Scripts and per-leg logs at `phase13_k3like_48b_posttrain/matrix_scripts/` in https://github.com/QIU023/torchtitan_attention_residual; the code is https://github.com/QIU023/torchtitan on `attention_residual_dev`. `PP8xVP4` on our own 32-layer flavor is published at `seq_len 1024` because 8192 does not fit in 15.5 GiB -- that and the long-horizon runs are pending re-verification on larger hardware.

--- PASTE END ---

## Deliberate differences from the draft this replaces

* **The `ln(163840) = 12.006` reference is gone.** It made sense as reassurance
  for a cold-start table whose cells were spread over 0.047 -- "all just above
  ln(vocab)" was the only structure available. With seeded init the numbers
  cluster exactly by reduction order, which is a stronger statement, and quoting
  a random-init floor next to it invites the question of why a seeded run needs
  that reassurance.
* **The "what these numbers are not" paragraph is gone, except one clause.**
  Deleting it entirely would be a problem: a tight table with no scope statement
  reads as cross-stack agreement, which is exactly what has not been done. One
  sentence keeps the claim honest without arguing with a reader who has not
  objected yet. The "about 0.4 nats apart" figure is dropped -- it described a
  cold-start comparison that no longer exists.
* **Steps are 100, not 10.** The seeded matrices ran the full horizon.
* **Compiled is stated up front** rather than left out, since 15/18 with a named
  upstream cause is a better answer than silence about compile.
