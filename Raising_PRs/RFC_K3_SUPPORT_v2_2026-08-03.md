# Update for RFC #3029: Kimi K3 parallelism support

**Not a new RFC.** This is the text to post as an update on
pytorch/torchtitan#3029 (the Block Attention Residuals RFC), which stays the
single thread. Filing a second RFC would split the discussion and force the
maintainers to reconcile two; #3029 already carries the AttnRes history and its
adoption gate is now satisfied, so the K3 parallelism work belongs as its
continuation.

Supersedes our unposted pre-release draft
(`phase13_k3like_48b_posttrain/RFC_K3_SUPPORT_DRAFT.md`), written before the
2026-07-27 weights + report drop and before pytorch/torchtitan#4025 existed.
Three things changed and all three matter:

1. the tech report and official `config.json` are public, so the architecture is
   no longer provisional,
2. K3 is confirmed natively multimodal and the report documents the vision
   tower, so "text-only until the report" is no longer the right scope, and
3. **#4025 has claimed `torchtitan/models/kimi_k3/`** with an eager FSDP-only
   reference. This RFC is now about what goes *on top of* that, not about
   whether the folder should exist.

## Position

**We are not proposing a competing model definition.** #4025 should land. What
we are staking out here is the parallelism layer, which #4025 explicitly scopes
out -- its `parallelize.py` raises on tensor parallel, context parallel,
activation checkpointing, CPU offload, and any non-default SPMD backend.

All of it is implemented and verified in our fork today. We would like to land
it as four follow-up PRs against #4025's layout rather than carry it
out-of-tree, and we are asking for the seams that make that a patch rather than
a fork.

## What is ready

Every number below is from the current fork head, seed 42, `--debug.deterministic`.

### Text: 13-leg parallelism matrix

`kimi_k3_mini_diag_4l_moe_depth`, global batch 8, 10 steps. All 13 legs pass and
all 13 are monotone.

    dp1  fsdp2  pp2  cp2  tp2
    fsdp2_tp2_pp2  fsdp2_tp2_cp2  tp2_pp2_cp2  fsdp2_pp2_cp2
    ep2_fsdp2  ep2_fsdp2_tp2_pp2  ep2_fsdp2_tp2_cp2  ep2_fsdp2_pp2_cp2

Cross-leg spread peaks at 6.03% of loss at step 4 and closes to 2.70% by step
10 -- it converges rather than diverging, which is the property that matters.

### Multimodal: 12-leg parallelism matrix

`kimi_k3_mini_vl` (MoonViT-V2 shrunk to 4 layers / hidden 256, every structural
feature kept), real images from the bundled `cc12m-test` shard, 10 steps. All 12
pass, all 12 monotone, spread peaks at 2.55% and closes to 1.94%.

    mm_fsdp2  mm_fsdp2_tp2  mm_fsdp2_cp2  mm_fsdp2_pp2
    mm_fsdp2_tp2_pp2  mm_fsdp2_pp2_cp2  mm_fsdp2_tp2_cp2
    mm_ep2_fsdp2  mm_ep2_fsdp2_tp2_cp2  mm_ep2_fsdp2_pp2
    mm_ep2_fsdp2_tp2_pp2  mm_ep2_fsdp2_pp2_cp2

Each leg is gated on a liveness probe that asserts the tower actually executed
and received gradients, not merely that the loss fell. That check exists because
a whole earlier matrix passed while the vision path was silently inert.

### Pipeline parallel

PP8 x VP4 (32 layers, `layers_per_stage 1`, Interleaved1F1B): |Dloss| 0.0018 vs
the no-PP reference at step 1. The cross-stage adapter is cache-based and
incremental -- only newly committed blocks cross a hop, and they are released
when the microbatch finishes -- which is the same shape the report describes for
Block AttnRes under PP.

### Context parallel

Ulysses head-sharding for MLA; for KDA we use fla's merged KCP
(fla-org/flash-linear-attention#691, the implementation the report cites),
including its `causal_conv1d_cp` for the short-conv halo. We do not carry our
own recurrence.

## What we are asking #4025 for

Detailed in the review; summarized here because they are the RFC's real content.

1. **Factor the decoder loop over a layer range.** Block AttnRes threads a
   second carried value (the committed block-residual stack) alongside the
   hidden state. PP needs the loop enterable at layer `i` with
   `(x, block_residual_TND)` and exitable at `j` returning the same pair. This
   one decides whether our PP is a patch or a duplicate loop body.
2. **A layout hook in the expert state-dict path**, so per-expert (checkpoint)
   and grouped (grouped-GEMM) layouts are a backend choice rather than baked in.
3. **Per-feature unsupported-parallelism guards**, so support can land
   piecewise without editing one list every time.

## Why this is an update to #3029 rather than a new RFC

#3029 proposed Block Attention Residuals and set an adoption gate: adopt if a
production model ships it. K3 shipping Block AttnRes satisfies that gate, and
the report cites Moonshot's own AttnRes preprint for the cache-based pipeline
communication we independently arrived at. The parallelism work here is that
RFC's continuation, not a separate proposal, so it stays in the same thread.

## Scope

**In:** the parallelism layer for K3 (TP, CP, PP, EP) over #4025's model,
text and multimodal.

**Out:** inference/serving (Moonshot's own vLLM contribution covers it); the
generic cross-stage PP mechanism (proposed upstream before and declined -- the
adapter stays private inside the model folder); the report's sec 5.2.3 encoder
optimizations (dynamic CP along the patch dimension, and DEP's scheduling of ViT
compute into pipeline bubbles). The last one needs an upstream interface that
does not exist: `_ComputationType` is a closed enum and
`_step_microbatches` raises on unknown actions, so there is no way to place
user compute in a bubble without adding an action type. We would raise that
separately if there is appetite.

## Honesty notes

- 2.8T has never been run by us. The claim is: verified on 48B-real-weight
  shapes and K3-faithful topology; scale-out is config-level.
- The report's "<2% overhead" is algorithm FLOPs. Our "+2.7% step-time" is
  PP-adapter communication on PCIe. Different quantities; we do not conflate
  them.
- The vision tower here is deliberately shrunk. Nothing in this RFC claims a
  trained VLM.
