# PR-C: pipeline parallel for Kimi K3 with Block Attention Residuals

## What

Pipeline parallelism for K3, including Block AttnRes, whose residual stack is a
second value carried alongside the hidden state across stage boundaries.

**BLOCKED on an interface.** #4025's decoder loop iterates all layers from a
single entry point. PP needs it enterable at layer `i` with
`(x, block_residual_TND)` and exitable at `j` returning the same pair. Without
that seam this PR duplicates the loop body, which is not a reviewable state.
The ask is filed on #4025; this branch is ready to rebase the moment it is
answered.

## Design

The cross-stage adapter is **private to the model folder's parallelize** -- the
generic mechanism was proposed upstream before and declined, and we are not
re-proposing it.

Communication is cache-based and incremental: each hop ships only the blocks
this stage newly committed, the receiver rebuilds the full stack from its cached
prefix plus the delta, and entries are released when the microbatch finishes.
That is the same shape the report describes for Block AttnRes under PP.
Non-Interleaved1F1B schedules fall back to the plain path, correct without the
cache saving.

## Multimodal

The PP split cannot see through a multimodal wrapper: core's `_split_module`
walks only top-level `named_children()`, so neither flat nor dotted FQNs reach
the text stack and every stage ends up with zero parameters. This PR splits the
text model and rebuilds the wrapper around the chunk that kept `embed_tokens` --
vision features are spliced into the embeddings, so nothing vision-side crosses
a stage boundary. That matches the report's DEP framing, where the tower is not
sharded across stages.

## Verification

PP8 x VP4 (32 layers, `layers_per_stage 1`): |Dloss| 0.0018 vs the no-PP
reference at step 1. Text legs `pp2`, `fsdp2_tp2_pp2`, `tp2_pp2_cp2`,
`fsdp2_pp2_cp2`, `ep2_fsdp2_tp2_pp2`, `ep2_fsdp2_pp2_cp2`; multimodal
`mm_fsdp2_pp2`, `mm_fsdp2_tp2_pp2`, `mm_fsdp2_pp2_cp2`, `mm_ep2_fsdp2_pp2`,
`mm_ep2_fsdp2_tp2_pp2`, `mm_ep2_fsdp2_pp2_cp2`.

## Relationship to #4025

Built on #4025's `torchtitan/models/kimi_k3/` layout. Does not change the eager
forward path. Rebase onto that PR's landing before review.

## Verification

Seed 42, `--debug.deterministic`, 10 steps unless stated. Full matrices and the
per-defect history are in the RFC and in
`phase13_k3like_48b_posttrain/`.

## Honesty

2.8T has never been run by us; verified on 48B-real-weight shapes and
K3-faithful downscales, scale-out is config-level.
