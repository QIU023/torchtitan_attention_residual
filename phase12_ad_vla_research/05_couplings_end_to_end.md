# 05 — Couplings: when topics merge end-to-end

> **Scope**: configurations where 2+ of [`01_video_vlm.md`](01_video_vlm.md), [`02_bev_perception.md`](02_bev_perception.md), [`03_vla_planning.md`](03_vla_planning.md), [`04_world_models.md`](04_world_models.md) become inseparable. End-to-end driving systems live here.

## When topics couple

In principle the four topics are orthogonal: pick a perception modality (video XOR BEV), optionally add an action head (VLA), optionally add a future-state head (WM). In practice the strong AD-VLA / world-model works couple them tightly because:

- **Closed-loop sim training requires WM** — you can't get useful planning gradients without future-state prediction (either learned or via a sim wrapper).
- **BEV + video together (BEV-injection)** is empirically stronger than either alone — but the integration becomes architecture-specific (BEV-InMLLM's residual fusion is not a clean drop-in).
- **Multi-task heads** (perception QA + trajectory + occupancy forecast) share the encoder backbone — separating them costs accuracy.

## Reference end-to-end systems

### AutoVLA (NeurIPS 2025) — video + VLA + (optional) closed-loop
- Vision: multi-view frames
- Action: trajectory tokens + dual fast/slow "thinking"
- Training: SFT → **GRPO** RFT
- Eval: nuPlan + nuScenes (10k→185k); also CARLA closed-loop
- **Coupling**: video × VLA. WM is implicit in the closed-loop sim (CARLA), not in the model itself.
- **Why it matters to us**: this is the direct precedent for our SFT→GRPO recipe applied to driving. Our existing infrastructure maps 1:1.

### GAIA-2 (Wayve, 2024) — video + WM + (latent) BEV + action conditioning
- Architecture: latent video diffusion WM, conditioned on past frames + ego action + (optionally) text instruction + (optionally) HD map.
- Pretrain: ~50k hours of driving video from Wayve fleet.
- **Coupling**: video × WM × BEV × VLA — all four. Drives a full closed-loop sim.
- **Why it matters to us**: the architectural target if we ever scale up. SANA-WM is the open-source spiritual successor with linear attention.
- **Why it's out of scope**: see Tier C in [`04_world_models.md`](04_world_models.md).

### DriveDreamer-2 (2024) — video + WM + VLA
- Architecture: video diffusion WM + LLM-based action proposer; action proposer queries WM for futures, picks the best.
- **Coupling**: WM × VLA × video. Model-based planning.
- **Why it matters to us**: shows the "WM as imagination engine for VLA planning" pattern. Tier A WM in our [`04_world_models.md`](04_world_models.md) is a much smaller version of the same idea (latent next-state prediction informing planning).

### DriveVLM / EMMA / DriveMLM / OpenEMMA — video + VLA (no WM)
- Vision: multi-view + (sometimes) BEV.
- Action: trajectory.
- Training: SFT only (no GRPO).
- **Coupling**: video × VLA, sometimes + BEV-injection.
- **Why it matters to us**: the dominant pattern in industry. AutoVLA's RFT layer is what differentiates from these.

### BEV-InMLLM — video + BEV (no action)
- Vision: multi-view tokens (primary) + BEV Q-Former tokens (residually fused).
- Output: QA only — no action.
- **Coupling**: video × BEV. Perception only.
- **Why it matters to us**: the cleanest example of "you don't have to pick one perception modality." If we end up in a video+BEV hybrid, this is the architecture template.

### OpenDriveVLA / VLA-MP / BEVDriver — BEV + VLA
- Vision: BEV-only.
- Action: trajectory or waypoints.
- **Coupling**: BEV × VLA. No WM in the model.
- **Why it matters to us**: the cheapest end-to-end AD-VLA recipe. If we go BEV in [`02_bev_perception.md`](02_bev_perception.md), the path to [`03_vla_planning.md`](03_vla_planning.md) is well-trodden.

## What couples vs what stays modular

| Combination | Genuinely coupled? | Reason |
|---|---|---|
| video + BEV (BEV-injection) | ⚠️ partial — the residual fusion module is architecture-specific | the rest of the stack (LLM, output head) is unchanged |
| video + VLA | ❌ no — just swap LM head for trajectory head | label / objective change only |
| BEV + VLA | ❌ no — same | trajectory head sits on BEV-token LLM output |
| video + WM | ✅ yes — WM head must align with vision encoder representation | encoder + WM head co-evolve |
| BEV + WM (next-step BEV) | ✅ yes — BEV encoder + WM head co-evolve | same reason |
| VLA + WM (model-based planning) | ✅ yes — WM is consumed by the action head | DriveDreamer-2 style |
| **video + BEV + VLA + WM** (full E2E) | ✅ heavily coupled — GAIA-2 / SANA-WM territory | every component co-trained |

**Practical read**: as long as WM is not in the picture (Tiers A–C in [`03_vla_planning.md`](03_vla_planning.md)), the perception/action split is clean and components can be swapped. Adding WM means co-training the encoder + WM head, which couples the perception and WM tracks.

## Pragmatic stack choices for end-to-end

If the target is **a single end-to-end demo**, three coupled stack choices:

### Stack 1 — light coupling: video + VLA, no WM (AutoVLA pattern)
- Modules: video encoder + projector + LLM + trajectory head + GRPO loop
- Coupling: minimal — each module swappable
- Our existing pipeline: ~80% reusable
- Effort: ~2 weeks (Tier A+B in [`03_vla_planning.md`](03_vla_planning.md))

### Stack 2 — medium coupling: video + VLA + Tier A WM (auxiliary)
- Modules: above + latent dynamics head
- Coupling: encoder shared between VLA and WM head
- Effort: +1 week on Stack 1
- **Best ROI** for showing world-model awareness without scaling up

### Stack 3 — heavy coupling: BEV + VLA + BEV-WM (occupancy forecasting)
- Modules: BEV encoder + Q-Former + LLM + trajectory + BEV occupancy forecasting head
- Coupling: BEV encoder shared by all heads
- Effort: 4-6 weeks (Tier B of [`02_bev_perception.md`](02_bev_perception.md) + Tier A of [`04_world_models.md`](04_world_models.md))
- **Most novel** without going to SANA-WM scale

## Limitations / open questions

- Co-training multiple heads (multi-task) loses ~1–3% on each individual task vs single-task baseline; needs careful loss weighting.
- BEV-injection (BEV-InMLLM) requires architecture surgery; not a clean LoRA addition.
- WM-in-the-loop GRPO is theoretically nice but empirically fragile; "reward = low next-state surprise" is hackable.

See [`06_strategy_workplan.md`](06_strategy_workplan.md) for our specific base + work plan.
