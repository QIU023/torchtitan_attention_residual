# PR-A: tensor parallel for Kimi K3 (KDA + Gated MLA + LatentMoE)

## What

Tensor parallelism for the K3 architecture, which #4025 currently rejects
(`("tensor parallel", parallel_dims.tp_enabled)` in its guard list).

Covers the three pieces that are not DSv3-shaped:

- **KDA** is `NoParallel` under TP -- replicated, because fla's kernels are not
  DTensor-aware. The DTensor shims are bound per instance rather than patched
  onto the class, so two models in one process cannot corrupt each other.
- **Gated MLA** follows the DSv3 pattern for q/kv compression, with the
  full-rank output gate sharded alongside the heads.
- **Stable LatentMoE** shards `down -> experts -> RMSNorm -> up`; the latent
  input's TP gradient is declared `Partial`, not `Replicate`.

## Why the gradient placements are the substance

Two of the defects fixed on the way here were pure `grad_placements` bugs that
no forward test could see:

- `moe_sharding` computed `in_grad_placements` and then passed it only on the
  EP branch, dropping it without EP. This reproduces on unmodified `deepseek_v3`
  (0.478 -> 0.0025 per-parameter gradient error) and is filed separately.
- Block AttnRes declared a `Partial()` TP gradient placement that over-reduced
  by exactly `1/tp`.

Both were found by per-parameter gradient attribution against a shared seed
checkpoint, not by loss curves.

## Matrix

`tp2`, `fsdp2_tp2_pp2`, `fsdp2_tp2_cp2`, `tp2_pp2_cp2`, `ep2_fsdp2_tp2_pp2`,
`ep2_fsdp2_tp2_cp2` -- text; `mm_fsdp2_tp2`, `mm_fsdp2_tp2_pp2`,
`mm_fsdp2_tp2_cp2`, `mm_ep2_fsdp2_tp2_cp2`, `mm_ep2_fsdp2_tp2_pp2` --
multimodal. All pass, all monotone.

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
