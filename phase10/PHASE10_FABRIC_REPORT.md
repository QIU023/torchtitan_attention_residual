# Phase 10 — Block AttnRes Fabric Profiling: Production-Grade Catalog

This document aggregates the fabric-trace findings from Phase 7 (training),
Phase 9-B (toy PPO), and Phase 10 (inference + real PPO) into a single
end-to-end Block AttnRes fabric reference covering pretrain, post-train,
inference, and RLHF regimes.

## Scope and contributions

* **Training fabric** (4D): v11 4D pretrain, v12 4D EP-replace-TP, SFT 4D
  post-train (Phase 7).
* **Inference fabric** (3D, fwd-only): Phase 10 Stage D, FSDP=4 × TP=2 ×
  EP=2 on phase4 step-8000.
* **Toy cross-mesh RLHF** (Phase 9-B): random-init MLP actor + ref on
  disjoint sub-meshes, captures the cross-mesh KL signature.
* **Real RLHF fabric** (Phase 10 Stage F): kimi_linear AttnRes actor +
  frozen ref co-located on FSDP=4 × TP=2 × EP=2; production-shape
  per-step fabric.

The unique research contribution: **no inference framework upstream
(vLLM / SGLang / TensorRT-LLM) implements Block AttnRes**; the RS+AG
collective pattern from the paper's two-phase optimization is therefore
exclusive to this codebase, and the per-regime fabric profiles below
are the project's deliverable for IXIA modeling.

## Mesh comparison

| Regime | Mesh | World | Forward | Backward |
|---|---|---|---|---|
| v11 training | FSDP=2×PP=2×TP=2×EP=2 | 8 | yes | yes |
| v12 training | FSDP=2×dpRep=2×PP=2×EP=2 | 8 | yes | yes |
| SFT post-train | same as v11 | 8 | yes | yes |
| Stage D inference | FSDP=4×TP=2×EP=2 (+EP overlay) | 8 | yes | no |
| Stage F PPO real | FSDP=4×TP=2×EP=2 (+EP overlay) | 8 | yes (actor + ref) | yes (actor only) |
| Toy PPO 9-B | actor 0–3 / ref 4–7 disjoint | 8 | yes | actor only |

## Per-collective per-regime signature (50-step traces)

Numbers below are total ops over the trace window; `nranks` indicates
the size of the participating process group (axis identity).

| Collective | v11 train | Stage D infer | Stage F PPO | Toy PPO 9-B | Notes |
|---|---|---|---|---|---|
| `AllGather 256MB+ nranks=2` | yes | **400** (init burst) | yes (init) | none | Inference / PPO-init weight unshard |
| `AllGather 16-256MB nranks=4` | n/a | **8** | **408** | none | FSDP=4 layer-wise unshard during fwd. PPO has 2× (actor + ref) |
| `ReduceScatter 16-256MB nranks=4` | yes | **0** | **400** | none | Inference drops RS. PPO has actor backward (1× per step) |
| `ReduceScatter 1-64KB nranks=4` | yes | 0 | **400** | small | FSDP grad scatter on small params (AttnRes pseudo-queries etc.) |
| `Send/Recv 16-256MB nranks=2` | PP fwd+bwd | 26 (init) | **752** (376 dispatch + 376 combine) | none | EP all-to-all logged as Send/Recv pairs at nranks=2 |
| `AllReduce <1KB nranks=8` | optimizer step | 8 | 8 | 8 | Cross-mesh barrier / MoE bias reduce |
| `Broadcast <1KB nranks=8` | n/a | 16 | 24 | **800** (cross-mesh KL) | Toy PPO's distinctive cross-mesh KL signature |
| `Broadcast 1-16MB nranks=2` | DCP save | 776 | many | none | Per-process DCP load fan-out |

Key observations:
1. **Stage F PPO real has both AllGather and ReduceScatter** at 16-256MB
   nranks=4 — the diagnostic split between training and inference. PPO
   shows training fabric for actor + inference fabric for ref simultaneously.
2. **Toy PPO 9-B's nranks=8 Broadcast** at <1 KB (800 ops) is the unique
   cross-mesh signature; absent from all other regimes (which run on a
   single coupled mesh).
3. **Stage D inference has zero ReduceScatter** at all sizes —
   distinctive forward-only signature.

## Fabric volume comparison (cumulative bytes, 50-step windows)

| Regime | pp+ep volume | fsdp volume | Effective per-step rate |
|---|---|---|---|
| v11 5000-step train | 2566 TB | 329 TB | 513 GB/step pp+ep |
| Stage D infer 50-step | 928 GB | 156 GB | 18.6 GB/step pp+ep, 3.1 GB/step fsdp |
| Stage F PPO real 50-step | 710 GB | 478 GB | 14.2 GB/step pp+ep, 9.6 GB/step fsdp |
| Toy PPO 9-B 50-step | 0 (toy MLP) | 0 (toy MLP) | only KL exchange |

PPO real's `fsdp` volume (9.6 GB/step) is **3× inference** (3.1 GB/step) —
because PPO has both actor's allgather (1×) AND ref's allgather (1×) per
step plus the reduce-scatter on actor backward. Roughly tracks (2 fwd + 1 bwd) /
1 fwd = 3.

## Block AttnRes-specific fabric pattern

Block AttnRes contributes:
1. Two extra `block_attn_res(...)` aggregations per layer — local compute,
   **zero fabric** (the aggregation itself never goes cross-rank in our
   current naive impl).
2. Four extra parameters per layer: `attn_res_proj`, `attn_res_norm`,
   `mlp_res_proj`, `mlp_res_norm`. Each is replicated across TP and
   FSDP-sharded across the dp_shard mesh. Their AllGather/ReduceScatter
   is included in the small-msg counters above (~400 RS ops at 1-64KB
   nranks=4 in Stage F is dominated by these).
3. Two top-level final aggregations: `final_attn_res_proj` /
   `final_attn_res_norm` (last PP stage / last layer). Same fabric
   pattern as per-layer params, just one extra set.

The blog-described **two-phase computation** (paper-aligned production
inference path) would convert each attention-layer's TP path from
`AllReduce` to `ReduceScatter + AllGather` (same total bytes, different
collective shapes). In the current codebase this is **not implemented**;
implementing it would change Stage D / Stage F fabric as follows:

| Pattern | Current (naive) | With two-phase |
|---|---|---|
| Per-attention TP collective | 1 × AllReduce nranks=2 | 1 × ReduceScatter + 1 × AllGather nranks=2 |
| Per-step total per actor | 16 × AllReduce | 16 × RS + 16 × AG |
| Per-step bytes | 16 × 12 MB = 192 MB | 16 × 6 MB RS + 16 × 6 MB AG = 192 MB |
| Fabric pattern shape | 16 events at full size | 32 events at half size (2× the count, half the per-call) |

This is the unique research artifact for IXIA modeling — the only
deployment of Block AttnRes inference anywhere is in this codebase, and
the two-phase optimization changes the fabric pattern signature in a
quantifiable way.

## IXIA configs catalog

Generated `ixia_config.json` files (post-Phase 10):

| Path | Size | Source |
|---|---|---|
| `phase5/runs/v11_4d_*/tier_b_trace/ixia_config.json` | 30 KB | v11 5000-step training |
| `phase5/runs/v12_trace_50step/tier_b_trace/ixia_config.json` | 24 KB | v12 50-step post-hoc |
| `phase5/runs/sft_v11_*/tier_b_trace/ixia_config.json` | 30 KB | SFT 490-step |
| `phase5/runs/5d_mode_b_*/tier_b_trace/ixia_config.json` | 29 KB | llama3 PP=2×FSDP=2×CP=2 |
| `phase5/runs/ppo_smoke_no_vllm/tier_b_trace/ixia_config.json` | 6.3 KB | toy PPO cross-mesh |
| `phase5/runs/inference_torchtitan_phase4_step8000/tier_b_trace/ixia_config.json` | 29 KB | **Stage D inference** |
| `phase5/runs/ppo_real_torchtitan/tier_b_trace/ixia_config.json` | ~30 KB | **Stage F real PPO** |

Plus older alignment-matrix traces from Phase 6 (8gpu_a2/a3/b0 — see
`phase7/FINAL_CATALOG.md`).

## Outstanding work

1. **Two-phase computation impl**: would close the gap between paper-
   described production inference and our captured naive-AllReduce
   pattern. ~1-day engineering work to implement in
   `KimiAttnResDecoderLayer.forward` + apply_tp_kimi_linear.
2. **Cross-system PPO** (separate actor-mesh + ref-mesh + reward-mesh):
   blocked by sgl_kernel cu130/py314/sm120 wheel availability. Once
   SGLang serves on this env, the `kimi_block_attn_res.py` upstream
   model file (already on `attention_residual_inference` branch) +
   `dcp_to_hf_kimi_attn_res.py` ckpt converter close the loop.
3. **commId-aware axis labels**: heuristic conflates PP and EP at
   nranks=2 Send/Recv. A trainer-side commId dump joined post-hoc
   with `comm_axis_map.csv` resolves the ambiguity (carry-over from
   `phase7/FINAL_CATALOG.md`).
4. **CP / fla-core ring-recurrence**: KDA blocks CP support; would
   require fla-core upstream PR for `chunk_kda` to support ring
   updates.
