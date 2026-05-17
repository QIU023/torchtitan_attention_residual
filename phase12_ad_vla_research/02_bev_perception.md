# 02 — BEV perception

> **Scope**: bird's-eye-view (top-down ego-centred) representation as the perception modality entering the LLM. Output is text/QA (action is in [`03_vla_planning.md`](03_vla_planning.md); when BEV is fused with future-state prediction see [`04_world_models.md`](04_world_models.md)).

## Why BEV at all

Concatenating tokens from 6–8 surround cameras does **not** model spatial consistency — the LLM has to re-derive "where things are" from view-tagged 2D tokens. A BEV is **ego-centred, metric, and spatially consistent by construction**: object location, scale, and distance are explicit. It is also the native interface to HD maps and to most existing AD perception stacks.

Direct evidence: BEV-InMLLM reports **+4.1 pt on spatial tasks** from BEV injection alone over a multi-view-only baseline.

## Three integration patterns

### (A) BEV-feature → projector → tokens (the dominant pattern)

A BEV encoder (BEVFormer / LSS / InterFuser) produces a dense BEV feature map. BEVFormer's default is **200×200** over a 102.4 m × 102.4 m ego area (0.512 m/cell). Raw is 40k cells — too large to feed. Every BEV-VLA work compresses before the LLM:

- **Adaptive pool + per-type MLP** (OpenDriveVLA): pool to `<SCENE>` token; separate `<TRACK>` (agents) and `<MAP>` tokens; each type 2-layer GeLU MLP into language space. Result: O(10²) structured tokens.
- **Q-Former** (VLA-MP, BEVDriver, BEV-InMLLM): BLIP-2-style with a small fixed set of **learnable BEV queries** cross-attending the BEV feature map. All three converge on **32 query tokens** (BEV-InMLLM ablation: 64 → diminishing returns). Optionally instruction-conditioned.

### (B) BEV-raster-as-image

Render the BEV semantic/occupancy/HD-map raster as an image and feed through the existing vision encoder. Zero new encoder, but throws away metric precision and the encoder must relearn top-down semantics. Cheap baseline; not what strong works do.

### (C) BEV injection / residual fusion (BEV-InMLLM)

Keep the multi-view token stream as primary; use BEV Q-Former output to **residually fuse** spatial awareness back. Plug-and-play; gives spatial benefit without replacing the vision encoder. Natural "video+BEV hybrid" — relevant when not committing to one modality (see [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md)).

## Token-budget arithmetic

| Modality | Tokens into LLM | vs single-image |
|---|---:|---:|
| Current single-image VLM | ~196 | 1× |
| BEV Q-Former (VLA-MP/BEVDriver/BEV-InMLLM) | **~32** | ~0.2× |
| BEV pooled structured tokens (OpenDriveVLA: scene+track+map) | **~64–256** | ~0.3–1.3× |
| Multi-cam × temporal raw (see [`01_video_vlm.md`](01_video_vlm.md)) | ~5k–12k | ~25–60× |

**Implication**: BEV is **1–2 orders of magnitude cheaper** than raw surround-video; **fixed token count** (does not grow with cams/frames — temporal context is folded into the BEV encoder, e.g. BEVFormer's recurrent BEV). The price: heavier separately-trained BEV encoder + BEV quality is a hard dependency.

## Landscape (representative BEV-VLA works)

| Work | BEV encoder → LLM | Tokens | Action | Notes |
|---|---|---:|---|---|
| **OpenDriveVLA** | per-view BEV + adaptive pool + per-type MLP, Qwen2.5-Instruct full-param | 64–256 | trajectory tokens AR | hierarchical scene+track+map tokens; nuScenes SoTA-class |
| **VLA-MP** | RGB+LiDAR fusion → Q-Former → LLM, GRU bicycle dynamics head | 32 | physics-constrained trajectory | 3-stage training (perception → BEV-LM align → joint); CARLA LangAuto DS 44.3/63.5/78.4 |
| **BEVDriver** | InterFuser BEV (5.37M, 256-d) → 32-query Q-Former 768-d → LLaMA LoRA r16 → GRU waypoints | 32 | waypoints | closed-loop CARLA LangAuto long-route DS 48.9 / RC 59.7 |
| **BEV-InMLLM** | frozen LSS/BEVFormer → instruction-aware BEV Q-Former → residual fuse into multi-view MLLM | 32 | QA | +4.1 pt spatial tasks, +2.1 pt holistic |
| **Talk2BEV** | BEV map from cams+LiDAR, per-object VL features | object-level | QA | ICRA'24 |

## How Kimi-Linear / AttnRes fits BEV

**Honestly, not great**. The win conditions for KDA and AttnRes are *long sequences*; at 32–256 BEV tokens, the attention term is negligible either way and linear attention's O(N) advantage shrinks to noise. BEV is **structurally better suited to Qwen2.5-VL + the DriveLM-Qwen compression toolkit** (see [`06_strategy_workplan.md`](06_strategy_workplan.md)).

What *does* transfer from our stack to a BEV track:
- **SATS-CRP-style region-pooled attention distillation** maps cleanly to BEV: regions = BEV grid cells / map elements. Distill from a larger teacher (Qwen2.5-VL-32B or a strong BEV-VLA) into the 3B student.
- The orchestration / GRPO pipeline (if combining with [`03_vla_planning.md`](03_vla_planning.md)).
- The inference / serving overlay.

## When to pick BEV over video ([`01_video_vlm.md`](01_video_vlm.md))

- ✅ Pick BEV when **metric/spatial reasoning** is the headline (planning, distance, geometry).
- ✅ Pick BEV when **token budget** must be tiny and fixed.
- ✅ Pick BEV when the project is built around an **existing AD perception stack** (BEVFormer/LSS).
- ❌ Skip BEV when **appearance detail** matters (signs, lights, text).
- ❌ Skip BEV when you want to exercise linear attention — BEV is the wrong regime.

## Limitations / open questions

- BEV quality is a hard floor: a weak frozen BEV encoder caps everything downstream.
- Camera-only BEV (no LiDAR) is competitive but more fragile (VLA-MP/BEVDriver use cam+LiDAR).
- The 3-stage training schedule (perception pretrain → BEV-LM align → joint) is non-trivial to orchestrate.

See [references.md → BEV-VLA](references.md#bev-vla-track) for citations.
