# Phase 12 — Autonomous-Driving VLA: Research & Strategy

Status: scoping doc. Created 2026-05-14.
Question: can the current multimodal Kimi-Linear + SigLIP stack (SFT→GRPO pipeline)
extend into a Vision-Language-Action (VLA) model for autonomous-driving video?
And: is it more natural to extend from the existing **DriveLM Qwen2.5-VL LoRA**
project, or from **this** (Kimi-Linear) project?

---

## 1. Bottom line

- **Feasible and well-trodden.** VLA-for-AD (VLA4AD) is an active field with a
  dedicated survey + "awesome" repos. The VLM→VLA extension is standard.
- **The SFT→GRPO recipe maps almost 1:1.** AutoVLA (NeurIPS 2025) is an end-to-end
  driving VLA trained with exactly SFT + GRPO reinforcement fine-tuning. RFT gave
  it +10.6% PDMS (NAVSIM) and −66.8% runtime. So the GRPO half of our pipeline is
  not a "nice to have" — it is a meaningful, published capability for AD-VLA.
- **Recommendation (see §3): start the near-term prototype from the DriveLM
  Qwen2.5-VL base; keep Kimi-Linear as the architecture research track.**
  Rationale: the DriveLM-Qwen base is already in-domain (driving data + a
  video-capable, dynamic-resolution VLM with native sglang support), and the
  GRPO pipeline is *portable*. The Kimi-Linear linear-attention advantage is real
  but only pays off at the long multi-cam-video context scale — a later concern.

---

## 2. VLA4AD landscape (2023–2026)

The architecture pattern is consistent: **VLM backbone + action/trajectory head**,
optionally + RL fine-tuning in (open- or closed-loop) sim.

| Work | Backbone / vision | Action output | RL? | Notes |
|------|-------------------|---------------|-----|-------|
| **AutoVLA** (NeurIPS 2025, UCLA) | VLM | trajectory tokens; dual fast/slow "thinking" | **GRPO RFT** | nuPlan+nuScenes (10k→185k samples), open- & closed-loop (CARLA). Directly validates SFT→GRPO. |
| **OpenDriveVLA** | pretrained vision enc., multi-view + ego state + language cmd | driving trajectory | — | 3D VLA, hierarchical visual input |
| **VLA-MP** | BEV (RGB+LiDAR) + cross-modal projector + LLM | physics-constrained trajectory (GRU-bicycle dynamics adapter) | — | emphasizes physically-consistent action |
| RT-2 / OpenVLA (robotics, not AD) | ViT/DINOv2/SigLIP + LLM | discretized action tokens (reuse LM head) | — | the canonical "VLM→VLA via action tokenization" pattern |

Data scale takeaway: AutoVLA used **10k–185k** driving samples. Our current SFT
uses **558k** image-caption pairs. **Same order of magnitude** — driving-data
scale is *not* the bottleneck for a research prototype.

---

## 3. Two candidate bases — strategic comparison

### Base A — DriveLM Qwen2.5-VL LoRA (the other project)

| | |
|---|---|
| **Domain** | ✅ already a *driving* VL fine-tune (DriveLM = driving QA/perception dataset) |
| **Vision** | ✅ Qwen2.5-VL: native dynamic resolution + **native video support** + multi-image — far closer to multi-cam/temporal driving input out of the box |
| **Serving** | ✅ sglang has **first-class Qwen2.5-VL support** — "no sglang pipeline" is cheaply fixable, not a real blocker |
| **Post-training** | ❌ no GRPO pipeline today — but GRPO infra is **portable** (the actor/rollout/reward loop in `phase11/rlhf` is largely model-agnostic; the hard part — the driving env wrapper — is needed either way) |
| **Adaptation depth** | ⚠️ LoRA only — limits how much the backbone can reshape for an action head; may need to widen to full-param or larger LoRA rank for the action modality |
| **Attention** | ⚠️ standard quadratic attention — costly for long multi-cam-video token sequences |

### Base B — current Kimi-Linear + SigLIP (this repo)

| | |
|---|---|
| **Domain** | ❌ never seen driving data; SigLIP is single-image 224² — multi-cam/temporal encoder must be built from scratch |
| **Post-training** | ✅ SFT→GRPO pipeline + sglang overlay already built and (as of phase 11/this work) debugged: 4D-mesh TP fix, val-loss wiring, checkpoint-save fix all landed |
| **Architecture fit** | ✅ KDA **linear attention is O(N)** — the long multi-cam-video token sequence (5k–12k vision tokens) that kills quadratic VLMs is exactly where this wins; AttnRes carrier also aids long-context aggregation |
| **Training** | ✅ full-param; ⚠️ 447M active is small for competitive AD (fine for a prototype) |
| **Maturity** | ⚠️ the model itself is still in stage-0 pretrain (~step 3.4k/12.75k) |

### Recommendation

**Near-term AD-VLA prototype → start from Base A (DriveLM Qwen2.5-VL).**
It is already in-domain, video-capable, and sglang-served; porting GRPO is
tractable engineering, not research risk. Fastest path to a *working* open-loop
driving VLA.

**Keep Base B (Kimi-Linear) as the architecture research track.** Linear
attention is the correct long-term bet once the token budget explodes
(surround-cam × temporal). Revisit as the primary base when (a) stage-0 +
follow-on pretrain produce a strong enough backbone, and (b) the prototype on
Base A has proven the data/reward/eval loop.

Key reframe: **the GRPO+sglang pipeline is portable; the linear-attention
architecture is not.** Don't let "Base A lacks the pipeline" drive the decision —
that gap closes with engineering. Let *domain readiness + vision maturity* drive
the near-term choice, and *architecture* drive the long-term one.

What we'd need to know to firm this up: current state of the DriveLM-Qwen
project (LoRA rank, which Qwen2.5-VL size — 3B/7B/72B, training data volume,
eval harness), and whether closed-loop sim (CARLA/nuPlan) is in scope.

---

## 4. Architecture mapping: VLM → AD-VLA (applies to either base)

| Component | Current (image-caption VLM) | AD-VLA target |
|-----------|------------------------------|---------------|
| Vision | single image → encoder → projector → tokens | **multi-view (6–8 surround cams) + temporal (4–8 frames)**; per-view encode+concat, or BEV projection (VLA-MP / OpenDriveVLA style) |
| Extra inputs | — | ego state (speed/accel/yaw), navigation command, optionally HD-map / BEV raster |
| Backbone | LM | unchanged (this is the point — VLM backbone is reused) |
| Output head | LM head → text | **trajectory decoder**: (a) discretized action tokens reusing the LM head (RT-2 / AutoVLA style — minimal change), (b) continuous waypoint regression, or (c) physics-constrained head (bicycle-dynamics adapter) |
| Post-train | SFT on captions → GRPO on text reward | SFT on (obs→trajectory) → GRPO with **driving reward** (safety / comfort / progress / rule-compliance), open-loop first, then closed-loop sim in the rollout |

---

## 5. Compute estimates

- **Data scale**: not the bottleneck — AD-VLA papers use 10k–185k driving
  samples vs our 558k captions (same OOM).
- **Dominant multiplier = vision tokens per sample.** Multi-cam (6–8) ×
  temporal (4–8 frames) × ~196 tok ≈ **5k–12k vision tokens** vs ~196 now →
  ~25–60× tokens/sample.
  - With **linear attention (Kimi-Linear)**: FLOPs scale ~linearly → **~25–60×**
    the current SFT per-step compute.
  - With **quadratic attention (Qwen2.5-VL)**: attention term scales ~quadratically
    in sequence length → **~600–3600×** in the attention blocks (mitigable with
    windowed/sparse attention, token merging, or fewer frames). This is the
    concrete cost of choosing Base A — manageable for a prototype with frame
    subsampling + token merging, but it is the reason Base B wins at scale.
- **Backbone size**: AutoVLA-class systems are typically 3–7B. 447M–1B is fine
  for a research prototype; competitive systems want 3–7B (+~10–15×).
- **Verdict**:
  - Research prototype (≤1B backbone, nuScenes-scale, open-loop trajectory):
    **feasible on the current 8× RTX 5090**, days-to-weeks of training.
  - Closed-loop GRPO (CARLA/nuPlan in the rollout loop): adds significant
    orchestration + compute; AutoVLA shows it is doable.
  - Competitive system (3–7B, large-scale logs, closed-loop): needs a cluster,
    ~100–1000× current.

---

## 6. Work breakdown (phased)

1. **Decide base** (need DriveLM-Qwen project state — see §3).
2. **Driving data pipeline** — nuScenes / nuPlan / Waymo are open; build a
   driving-data variant of the multimodal dataset loader (obs + ego + cmd →
   trajectory label).
3. **Multi-view temporal vision encoder** — the biggest new component.
4. **Action head + tokenization scheme** — start with discretized action tokens
   (minimal change, reuse LM head).
5. **SFT** on (observation → trajectory) — open-loop, supervised.
6. **GRPO** — port the `phase11/rlhf` actor/rollout/reward loop; design the
   driving reward; open-loop reward first, then closed-loop sim wrapper.
7. **Eval harness** — open-loop (ADE/FDE, PDMS-style) then closed-loop (CARLA/nuPlan).

---

## 7. Open questions

- DriveLM-Qwen project: LoRA rank, Qwen2.5-VL size, data volume, eval harness state?
- Closed-loop sim in scope, or open-loop trajectory prediction only for v1?
- Target: research demo, or a path toward a deployable system?
- If Base B (Kimi-Linear): wait for a stronger pretrained backbone, or prototype
  on the current stage-0 checkpoint?

---

## References

- A Survey on Vision-Language-Action Models for Autonomous Driving (ICCVW 2025) — arxiv.org/pdf/2506.24044
- AutoVLA (NeurIPS 2025) — arxiv.org/abs/2506.13757 ; autovla.github.io
- OpenDriveVLA — arxiv.org/html/2503.23463v2
- VLA-MP — mdpi.com/1424-8220/25/19/6163
- Awesome-VLA4AD — github.com/JohnsonJiang1996/Awesome-VLA4AD
- VLA Models for Autonomous Driving: Past, Present, and Future — arxiv.org/html/2512.16760v2
