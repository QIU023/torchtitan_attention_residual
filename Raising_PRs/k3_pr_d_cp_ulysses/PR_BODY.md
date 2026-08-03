# PR-D: context parallel for Kimi K3 (Ulysses for MLA, KCP for KDA)

## What

Context parallelism for K3's hybrid attention, which #4025 currently rejects.

- **Gated MLA**: Ulysses head-sharding (all-to-all between sequence and head
  axes inside the attention module). Composes with TP -- the head axis is
  already tp-sharded and cp splits the local heads further.
- **KDA**: fla's merged KCP (fla-org/flash-linear-attention#691, the
  implementation the K3 report cites), via `chunk_kda(cp_context=...)`, plus its
  `causal_conv1d_cp` for the short-convolution halo. We deliberately do not
  carry our own recurrence: plain state summation is wrong for KDA because the
  delta rule applies a token-dependent transition to the incoming state.

The load balancer must stay off (`context_parallel_load_balancer=None`): K3's CP
reassembles contiguous rank-ordered shards, and a permuting balancer silently
breaks causal order. `parallelize` raises if it is set.

## A gradient defect worth naming

An earlier hand-rolled short-conv halo fetched the neighbour's state with
`dist.all_gather`, which is not autograd-aware, so the gradient each rank owed
its left neighbour's tail was silently dropped -- rank 0's last `W-1` tokens
came out ~60% wrong while every interior token was exact. Forward was bit-exact
either way. An error confined to `W-1` boundary tokens per rank does not move a
loss curve; only per-position gradient comparison finds it.

## Multimodal

`prepare_context_parallel_input` shards inputs, labels and positions but not
`pixel_values`, so each CP rank holds a slice of the vision sentinels while
receiving every image. Each rank keeps only the features belonging to its shard,
located by one all_reduce of per-rank sentinel counts.

A rank's shard can legitimately hold zero sentinels. Returning early there
deadlocks every other CP rank inside that all_reduce, so forward encodes and
selects before deciding whether it has anything to splice. This is the same
hazard class as #4025's `add_zero_valued_dependency`, reached via CP rather than
via an image-free batch.

The encoder still runs redundantly per CP rank. Sharding it along the patch
dimension with gather-KV is the report's sec 5.2.3 dynamic CP, and is out of
scope here.

## Matrix

`cp2`, `fsdp2_tp2_cp2`, `tp2_pp2_cp2`, `fsdp2_pp2_cp2`, `ep2_fsdp2_tp2_cp2`,
`ep2_fsdp2_pp2_cp2` -- text; `mm_fsdp2_cp2`, `mm_fsdp2_pp2_cp2`,
`mm_fsdp2_tp2_cp2`, `mm_ep2_fsdp2_tp2_cp2`, `mm_ep2_fsdp2_pp2_cp2` --
multimodal.

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
