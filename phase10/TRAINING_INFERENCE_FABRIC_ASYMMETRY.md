# Phase 10 — Training ↔ Inference Fabric Asymmetry

This document compares the NCCL fabric pattern of **kimi_linear Block AttnRes**
under training (v11 production, 4-axis 4D mesh) and forward-only inference
(Stage D, 3-axis mesh). The asymmetry between the two regimes is itself the
main finding — there is no third party comparison reference because Block
AttnRes inference does not exist in vLLM / TensorRT-LLM / SGLang upstream.

## Mesh comparison

| Axis | Training (v11) | Inference (Stage D) | Notes |
|---|---|---|---|
| FSDP (dp_shard) | 2 | 4 | Inference uses larger FSDP because no PP/CP frees ranks |
| PP | 2 | **1** | Inference forward is single-call; PP scheduler complexity dropped |
| TP | 2 | 2 | Same (intra-node, NVLink) |
| EP | 2 | 2 | Same (overlays dp_shard) |
| CP | 1 | 1 | Both: KDA/fla-core blocks ring-recurrence |
| **fabric axes ≥ 2** | **4** | **3** (PP dropped) | |
| world | 8 | 8 | |

## Per-collective signature

50-step trace baselines, kimi_linear_436m_block_attn_res_n4 ckpt.

| Collective | Training (v11, fwd+bwd) | Inference (Stage D, fwd only) | Asymmetry source |
|---|---|---|---|
| `AllGather 256MB+ nranks=2` | yes (FSDP unshard) | **400 ops** (init only) | Inference frontloads weight unshard once vs every microbatch in training |
| `AllGather 16-256MB nranks=4` | n/a (different mesh) | 8 ops/step | FSDP=4 layer-wise unshard during forward |
| `ReduceScatter` (FSDP grad) | yes (every microbatch backward) | **0 ops** | Inference has no backward |
| `Send/Recv 12-24MB nranks=2` | yes (PP fwd + bwd) | 26 ops total | Inference has no PP (these 26 are init path) |
| `Send/Recv (EP all-to-all)` | yes (per MoE layer fwd + bwd) | yes (per MoE layer fwd) | Inference has half the EP volume (no bwd combine) |
| `AllReduce nranks=2` (TP) | yes (per attention + per ffn fwd + bwd) | yes (per attention + per ffn fwd) | Same shape, half the count |
| `Broadcast 1-16MB nranks=2` | (DCP save) | 776 ops (DCP load) | Inference reads ckpt once at init; training reads + writes |

## Volume comparison (50-step traces, post-warmup)

| Axis | Training (v11) | Inference (Stage D) | Ratio (inf/train) |
|---|---|---|---|
| pp+ep (heuristic combined) | 2566 TB | 928 GB | ~0.0004x (training has 5000 steps; per-step rate inference ≈ 0.5x training fwd) |
| fsdp | 329 TB | 156 GB | ~0.0005x |
| dp | 4 KB | 112 B | similar |

Per-step normalization:

| Per-step bytes | Training | Inference (fwd only) | Ratio |
|---|---|---|---|
| pp+ep | 513 GB/step (5000 step) | 18.6 GB/step (50 step) | ~0.04x — but training has 4 active axes with PP and inference has 0 PP, so the EP-only volume is much smaller |
| fsdp | 65.8 GB/step | 3.1 GB/step | ~0.05x — inference avoids RS (backward) and uses smaller per-layer messages because no microbatch chunking |

**Caveat**: heuristic axis labels conflate PP and EP at `nranks=2` Send/Recv
(see `phase7/expand_to_flows.py:_classify`). To split EP from PP cleanly, a
trainer-side commId-to-axis dump is required (future work, identified in
`phase7/FINAL_CATALOG.md`). For inference at PP=1 the entire `pp+ep`
heuristic bucket is pure EP all-to-all, so the comparison is sound *within
inference* but the *training* `pp+ep` is genuinely mixed.

## Inference-specific patterns absent from training

1. **Startup weight unshard** (400 × AllGather at 256MB+ nranks=2) — once-per-process
   reconstitution path. Not present in training because FSDP allgather there is
   per-microbatch overlapping with compute. Suggests inference deployments
   benefit from `cudaMallocAsync` pool warming + persistent-kernel modes that
   training doesn't use.
2. **DCP load broadcast** (776 + 504 Broadcasts at 64KB–16MB nranks=2) — checkpoint
   read fan-out across ranks. In training this is a once-per-resume cost; in
   inference it's once-per-process and dominates the early-second timeline.

## Training-specific patterns absent from inference

1. **FSDP ReduceScatter** (per-layer per-microbatch grad reduce-scatter) —
   completely zero in inference. ~half of training FSDP fabric volume.
2. **PP fwd + bwd Send/Recv** (per-microbatch in both directions) — zero in
   inference (PP=1) and would be half-volume in PP-enabled inference (no bwd).
3. **Optimizer step AllReduce** (rare, low-volume) — zero in inference.

## Block AttnRes-specific delta

Block AttnRes adds a `block_attn_res(blocks, partial_block, proj, norm)`
aggregation **twice per layer** (pre-attn + pre-ffn). The aggregation is
local compute; it produces no cross-rank fabric. The only fabric impact
is via the `attn_res_proj.weight` (1×D vector) and `attn_res_norm.weight`
(D vector) — both ReplicatedLinear / RMSNorm under TP, ~2.4K params each
× 16 layers × 2 (attn+ffn) = ~80K parameters total. FSDP unshard volume
is dominated by attention QKV and FFN projections; AttnRes is < 0.1% of
that.

The blog (referenced by user) describes a **two-phase computation** where
Phase 2's online-softmax merge naturally embeds in the TP all-reduce path
as `reduce-scatter → local merge → RMSNorm → all-gather`. This would
**replace** the standard single TP `AllReduce` (1 op per attention) with
**2 ops** (`ReduceScatter` + `AllGather`). Volume-wise the bytes are
identical (RS+AG = AR), but the **fabric pattern signature** differs:
two adjacent collectives instead of one. This is **not implemented** in
our Stage D inference — current behavior is naive standard TP AllReduce.
Implementing two-phase would change Stage D inference to:

* `2 × AllReduce → 1 × ReduceScatter + 1 × AllGather` per attention
  layer (16 layers × 2 = 32 swaps)
* No volume change (identical bytes), but **fabric pattern shape**
  changes: collective count doubles, per-call message size halves
  (because RS sends only the rank's stripe).

This is the unique research contribution available — IXIA can model both
patterns and compare wire-level packet distributions. Implementation of
two-phase is left as Stage F-stretch (requires modifying
`KimiAttnResDecoderLayer.forward` to compute Phase 1 once per block then
fold Phase 2 into the post-attention TP collective path).

## Key takeaway for IXIA

For an inference deployment of Block AttnRes the fabric profile to model
in IXIA is approximately **(training fabric) × (0.4–0.5)** with these
specific deletions:

* All ReduceScatter ops dropped
* All "backward" Send/Recv on the PP axis dropped (here PP=1 so already 0)
* All optimizer-step AllReduce dropped
* Plus a one-time burst of large AllGather + Broadcast at process start

…and these specific additions:

* Persistent low-rate fwd-only EP all-to-all (= half training EP volume)
* Per-layer fwd-only TP AllReduce (= half training TP volume; or, if
  two-phase is enabled, RS+AG pair instead)
* No ReduceScatter at all anywhere

Inference fabric is **lighter and more predictable** than training fabric
on the same mesh — but Block AttnRes's two-phase optimization, if
deployed, fundamentally changes the per-attention TP collective pattern
(AllReduce → RS+AG), which is the unique fabric signature this codebase
profiles that no inference framework upstream can today.
