# PR-B: expert parallel + grouped-GEMM for Kimi K3

## What

Expert parallelism over K3's Stable LatentMoE (896 experts, top-16, 2 shared at
full scale), with the grouped-GEMM path and the all-to-all dispatch.

## The state-dict question

The released checkpoint stores experts per-expert; grouped-GEMM wants them
stacked. Today that mapping is fixed. This PR carries the conversion, but it
belongs behind a hook in #4025's `state_dict_adapter.py` -- raised as an ask in
that PR's review -- so one adapter serves both layouts in both directions.

## Matrix

`ep2_fsdp2`, `ep2_fsdp2_tp2_pp2`, `ep2_fsdp2_tp2_cp2`, `ep2_fsdp2_pp2_cp2` --
text; `mm_ep2_fsdp2`, `mm_ep2_fsdp2_pp2`, `mm_ep2_fsdp2_tp2_pp2`,
`mm_ep2_fsdp2_tp2_cp2`, `mm_ep2_fsdp2_pp2_cp2` -- multimodal.

EP-off legs are covered too: the `in_grad_placements` defect above only appears
with EP disabled, which is why the matrix runs both.

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
