# K3 reproduction stack — component status map (2026-07-21)

Every component the K3 training/post-training stack needs, mapped to its
implementation state on this fork + the evidence. "Provisional" = built
against blog-only info, exact form reconciles when the 7.27 config +
report drop. Fork: QIU023/torchtitan @attention_residual_dev.

## Architecture (K3 family)

| Component | Status | Evidence |
|---|---|---|
| KDA (linear attn) | DONE | fla-core kernels; kimi_linear layers; suite |
| MLA (NoPE) | DONE | KimiMLAAttention; DSv3-aligned |
| **Gated MLA** (K3 delta) | PROVISIONAL | near-identity sigmoid gate, graft-viable (max rel dlogit 6.8e-4 at init); test_gated_mla; exact form -> 7.27 |
| MoE (sigmoid grouped-topk) | DONE | KimiMoE over common MoE; router+shared+routed |
| Block AttnRes | DONE | attn_res.py + attn_res_model.py; the headline |
| **AttnRes graft gate (alpha)** | DONE | bit-exact step-0 identity; 48B real-weight anchor max|dlogit|=0.0 |
| SiTU activation | SKIP | weights trained for SwiGLU; non-graftable (report) |
| **Quantile Balancing routing** | DONE (provisional) | quantile_balance.py: CDF-position bias vs DSv3 sign rule; drop-in hook; tested. Exact K3 rule -> 7.27 |
| **Per-Head Muon optimizer** | DONE (base) | muon.py: NS-orthogonalized Muon + per-head hook + AdamW fallback; tested. Exact K3 per-head variant -> 7.27 |
| Stable LatentMoE | 7.27 | framework-level; wait for report |

## Parallelism (5D)

| Axis | Status | Evidence |
|---|---|---|
| FSDP2 / HSDP | DONE | all post-training demos; 48B sharded load |
| TP | DONE | module-internal MoE migration; TP=2 parity vs FSDP (step-1 exact) |
| EP | DONE | TP/EP migration; **EP@896** real-mesh smoke (896 experts, 112/rank) |
| PP (cross-stage adapter) | DONE | |dLoss|<=0.0057 to PP8xVP4; 2026-07-19 re-verification report |
| CP | DONE | correctness-first seq all-gather (KDA + MLA), composes with FSDP/PP/EP, cp=4 fwd close to cp=1 (6e-4); memory-optimal KDA Ulysses head-shard validated bit-exact (CP_ULYSSES_DESIGN.md) |
| Full-5D simultaneous | PARTIAL | 4D (FSDP+TP+EP+PP) + CP-composed variants (CP+FSDP/PP/EP) validated at debug scale; simultaneous 5-axis @ 48B = H200 (PLAN caveat) |

## Quantization

| Format | Status | Evidence |
|---|---|---|
| fp8 rowwise (SM89+) | DONE | KimiLinearFloat8Spec; 5-step smoke on SM12.0 |
| NF4 QLoRA (customer option) | DONE | FSDP2 composition; QLoRA SFT loop; dim-alignment guard. NOT K3's format |
| **MXFP4+MXFP8 QAT** (K3-faithful) | DONE | apply_mxfp4_qat STE fake-quant; torchao mx; 2 tests. Emulated (any GPU) |
| packed-MXFP4 import | GUARDED | state_dict_adapter rejects packed weights loudly; real unpack -> 7.27 |

## Post-training (veRL x torchtitan)

| Path | Status | Evidence |
|---|---|---|
| veRL torchtitan engine | INTEGRATED | patch v3 (mapping/namespace/flavor/gloo); shims; official-48B flavor resolves |
| Full-param SFT | DONE | 194m 40 steps (loss 11.49->11.26); config-scalable to 48B/H200 |
| LoRA SFT | DONE | 194m graft + 48B real-weight (bf16 base, AttnRes graft), end-to-end; loss 0.254->0.133 |
| LoRA save/export | DONE | merge_lora_state_dict folds adapter -> HF (to_hf drops lora keys otherwise); graft+LoRA compose+merge+export unit-tested |
| QLoRA SFT | DONE (mechanism) | standalone 194m loop + post-load quantize_lora_bases hook (correct meta-first order) + adapter-dtype fix, unit-tested. 48B-via-veRL QLoRA (NF4 base under titan shard-then-load DCP) = not yet run |
| LoRA P0 trio | DONE | step-0 identity / grad routing / LoRA-only payload; HF-export leg added via merge; tested |
| LoRA x parallelism | DONE | bf16-LoRA trains under FSDP/TP/EP/PP/PP+EP/wide-EP, incl. AttnRes-LoRA x PP skip-edge grad routing; TP was the last axis, fixed 07-21 |
| MXFP4-base LoRA x parallelism | DONE | shards + trains under FSDP2 (EP-aware apply_fsdp); FSDP+EP directly verified; NF4/MXFP4 x EP root-caused as orthogonal subsystems (debug-scale plumbing edges, not fundamental) |
| GRPO (standalone on K3) | DONE | grpo_titan_standalone.py: full-recompute rollout + group-adv + PG update, titan-native, 6 steps |
| GRPO (veRL-native) | BLOCKED | veRL RL rollout = inference-server-only. sglang overlay now PROVEN serving Kimi-Linear on 5090 (fork e45f675, 2 real bugs fixed); remaining = veRL imports sglang in-process -> torch 2.11 vs 2.12 venv split (titan FSDP uses 2.12 DataParallelMeshDims). Strategy: rollout side is covered by official K3 vllm on 7.27; our niche is the titan-actor + weight-sync wiring. GRPO_STATUS |

## Checkpoint / weights

| Item | Status | Evidence |
|---|---|---|
| HF<->tt state_dict_adapter | DONE | official 48B: 603/603 keys, 4/4 bit-exact sharded load |
| 48B graft anchor | DONE | gated graft vs base: max|dlogit|=0.0, top-1 100% |
| Artifact-discovery (7.27) | READY | 34 official key patterns catalogued; runbook |
| DCP cross-degree reshard | DONE | dp8->dp4 resume verified |

## Scale / flavors

| Flavor | Status |
|---|---|
| 194m..528m scaling-law + 447m_aligned | DONE (parameterized generator) |
| 48B-A3B (real weights) | DONE (loaded, graft-anchored) |
| **2.8T-A50B provisional** (896/16) | DONE meta-build + EP@896 mesh; dims placeholder -> 7.27 |

## Usable through the CLI

`kimi_linear_debugmodel_k3faithful` (`--module kimi_k3 --config`)
turns on the K3 architecture deltas (Gated MLA + alpha graft) and
trains through the real trainer; MXFP4 QAT + Per-Head Muon apply
via mxfp4_qat.py / muon.py hooks. Suite: 69 tests stable green.

## Honest blockers (not hidden)

1. 48B step-time on this box: all-gather bound (no P2P, 3.84 GB/s), ~5
   min/step; QLoRA cuts gather ~4x but only helps at 48B scale; H200
   NVLink unaffected. See PERF_48B_ALLGATHER_INVESTIGATION.
2. GRPO needs an inference server (sglang overlay / 7.27 vllm).
3. Full-5D simultaneous + full-param 48B = H200 (PLAN 3c).
4. Non-standard invented parts (alpha gate / module-LoRA / NF4-experts):
   see INVENTED_PARTS_REVIEW for risk + upstream posture.
5. QLoRA/MXFP4-QLoRA trainer flow (quantize-then-shard ordering) is the
   next H200 open item -- see H200_HANDOFF_2026-07-21.

## What 7.27 closes

Exact: AttnRes N + placement, KDA:MLA ratio, Gated-MLA form, Quantile
Balancing, Per-Head Muon, MXFP4-QAT recipe, official weight keys, 2.8T
dims. The provisional pieces above have extension points ready; the
adapter's packed-MXFP4 guard flips to a real unpack.
