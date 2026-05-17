# 06 — Strategy, base choice, asset inventory, work plan

> **Scope**: synthesis of the topic-specific docs into a concrete plan. Base choice between Kimi-Linear (this repo) vs DriveLM-Qwen (the other project). Asset inventory across both repos. Tiered work plan + interview narrative.

## Bottom line

For a **near-term AD-perception prototype** (1–2 weeks): start from **DriveLM-Qwen2.5-VL** in a **BEV-token configuration** with the existing compression toolkit. See [`02_bev_perception.md`](02_bev_perception.md).

For a **research / interview architectural bet** (4–6 weeks): **Kimi-Linear + Spatio-Temporal AttnRes** on the long multi-cam-video regime, with a Tier A WM head on top. See [`01_video_vlm.md`](01_video_vlm.md) Tier C + [`04_world_models.md`](04_world_models.md) Tier A.

The two are **not in tension** — they are two independently-shippable deliverables targeting different audiences (production-pragmatic vs research-bet).

## Two candidate bases

### Base A — DriveLM Qwen2.5-VL LoRA (the other project)

| | |
|---|---|
| Domain fit | ✅ already a driving VL fine-tune (DriveLM-nuScenes) |
| Vision | ✅ Qwen2.5-VL native dynamic resolution + native video + multi-image |
| Serving | ✅ first-class sglang support |
| Post-training | ⚠️ no GRPO yet (but infrastructure is portable from phase 11) |
| Compression toolkit | ✅ **decisive** — 4-method × 4-16× toolkit lives here |
| SATS-CRP distillation | ✅ user's own method, Qwen-native |
| GGUF quant | ✅ deployable artefact |
| Attention | ⚠️ quadratic — costly for long video, mitigated by compression |

### Base B — Kimi-Linear + SigLIP (this repo)

| | |
|---|---|
| Domain fit | ❌ never seen driving data |
| Vision | ❌ SigLIP 224² single image — multi-cam/temporal/BEV adapter required |
| Post-training | ✅ SFT→GRPO pipeline + sglang overlay debugged |
| Architecture fit | ✅ KDA O(N) wins on **long multi-cam video** specifically |
| AttnRes carrier | ✅ unique mechanism; long-context aggregation |
| Compression toolkit | ❌ none (and lower marginal value on linear attention — token compression and linear attention are partial substitutes) |
| Maturity | ⚠️ stage-0 still in progress; stage 2 SFT in-flight (2026-05-17) |

### Recommendation

| Track | Base | Why |
|---|---|---|
| **Near-term BEV prototype** ([`02_bev_perception.md`](02_bev_perception.md) Tier B) | **A** (DriveLM-Qwen) | In-domain, sglang-served, fits BEV-token budget, compression toolkit transfers cleanly |
| **Near-term video prototype** ([`01_video_vlm.md`](01_video_vlm.md) Tier A/B) | **A** (DriveLM-Qwen) | Quadratic attention pain mitigated by user's 4×–16× compression; multi-image / native video already wired |
| **VLA action extension** ([`03_vla_planning.md`](03_vla_planning.md) Tier A/B) | **A** (DriveLM-Qwen, then port phase 11 GRPO) | Tier A is a label swap; Tier B GRPO port is portable infra work |
| **Architecture research bet** ([`01_video_vlm.md`](01_video_vlm.md) Tier C + [`04_world_models.md`](04_world_models.md) Tier A) | **B** (Kimi-Linear) | Linear attention + AttnRes generalized to spatio-temporal is unique to Base B |

**Reframe**: GRPO+sglang is portable; the linear-attention architecture is not. The compression/distillation stack is also not portable. Two non-portable asset piles sit on opposite bases — the BEV prototype gets more value from the compression pile, the long-video architecture bet gets more value from the linear-attention pile.

## Asset inventory

### From `torchtitan_attention_residual` (this repo)

| Asset | Value |
|---|---|
| Block AttnRes (paper-faithful impl) | depth-wise aggregation → generalizable to temporal-wise ([`01_video_vlm.md`](01_video_vlm.md) Tier C, [`04_world_models.md`](04_world_models.md) Tier A) |
| KDA linear attention O(N) | the natural win for long video sequences |
| AttnRes SGLang inference overlay | deployment story for any AttnRes-based model |
| 4D parallelism (FSDP / TP / PP / EP) + AC | scaling story for long-context video pretrain |
| GRPO pipeline (phase 11) | portable to any base for VLA RFT |
| Stage 0 LM @ val 2.88 ↓ | a Kimi-Linear LM ready as VLM backbone |
| NCCL trace catalog + ixia | infra story (separate from perception) |

### From `DriveLM_VLM_Project` (the other repo)

| Asset | Value |
|---|---|
| Qwen2.5-VL-3B LoRA r16/α32 on DriveLM v1.1 | in-domain driving VLM baseline |
| 4 visual-token compression methods | applies to surround-cam × temporal token explosion |
| **SATS-CRP** (user's own method) | region-aware self-attention distillation; novel for BEV grid cells |
| GGUF Q4_K_M quant + llama.cpp deploy | deployable artefact |
| DriveLM v1.1 data pipeline (377k QA, 696 scenes) | in-domain QA SFT data |
| nuScenes raw 24k images, 6 cams | multi-view raw material |
| YAML-driven train framework | reusable for ablations |
| Hardware-portable configs (B200/GH200/4070Ti) | the 3B LoRA fits on 4070Ti |
| Continual-learning roadmap | distillation/KD narrative depth |

### Where they complement (the integration story)

- **DriveLM has** the domain data + the efficient-VLM craft + the deployment path.
- **This repo has** the long-context architecture + the inference infra + the parallelism plumbing.
- **The gap on both sides is video / BEV input.** Neither has it wired yet. Closing this gap is the phase-12 deliverable.

## Compute estimates (8× RTX 5090)

| Configuration | Tokens/sample | Per-step compute | Verdict |
|---|---:|---:|---|
| BEV track (Tier B [`02_bev_perception.md`](02_bev_perception.md)) | ~32–256 | ~0.2–1.5× current SFT | ✅ cheap; days of training |
| Video track Base A + 4× compression | ~2.3k | ~40–230× attention (∝ N²) | ✅ feasible with compression |
| Video track Base A no compression | ~9k | ~600–3600× attention | ❌ infeasible single-node |
| Video track Base B (linear attn) | ~9k | ~25–60× (∝ N) | ✅ feasible, this is where linear attn pays off |
| Closed-loop GRPO in sim | + sim wall-time | significant orchestration | ⚠️ months not weeks |
| Competitive 3–7B system, large-scale logs, closed-loop | — | ~100–1000× current | ❌ needs a cluster |

Backbone size: AutoVLA-class are 3–7B; OpenDriveVLA/VLA-MP/BEVDriver are in that range. DriveLM-Qwen 3B sits in the sweet spot. 447M (Base B) is fine for a research prototype but not competitive.

## Phased work plan

| Phase | Topic | Doc | Est |
|---|---|---|---|
| 1 | Decide modality + base | this file + topic files | 1 day |
| 2 | Tier A video temporal SFT (or BEV Tier B) | [`01_video_vlm.md`](01_video_vlm.md) / [`02_bev_perception.md`](02_bev_perception.md) | 1–2 weeks |
| 3 | Tier A VLA action head | [`03_vla_planning.md`](03_vla_planning.md) | 1 week |
| 4 | Tier B VLA open-loop GRPO | [`03_vla_planning.md`](03_vla_planning.md) | 2 weeks |
| 5 (optional) | Tier A WM aux head | [`04_world_models.md`](04_world_models.md) | 1 week |
| 6 (optional) | SATS-CRP distillation on BEV grid cells | [`02_bev_perception.md`](02_bev_perception.md) | 1 week |
| 7 (optional) | Tier C Spatio-Temporal AttnRes on Base B | [`01_video_vlm.md`](01_video_vlm.md) | 4–6 weeks |
| 8 (optional) | GGUF quant deployable artefact | [`06_strategy_workplan.md`](06_strategy_workplan.md) | 3 days |
| 9 (very optional) | Closed-loop GRPO in CARLA/nuPlan | [`03_vla_planning.md`](03_vla_planning.md) | 4–6 weeks |

## Interview narrative

### The 90-second pitch

> Built a multimodal SFT + GRPO pipeline on a custom Kimi-Linear (KDA + AttnRes + MoE) backbone. Stage 0 LM pretrain → Stage 1 projector alignment → Stage 2 visual instruction tuning, all the way to multi-modal GRPO with SGLang rollouts. Then asked: can this extend to driving? Audited the field (AutoVLA NeurIPS 2025 validates exactly SFT+GRPO for AD-VLA; OpenDriveVLA/VLA-MP/BEVDriver validate BEV-tokens), inventoried our two-repo asset stack, and identified Spatio-Temporal AttnRes as the novel architectural bet that uniquely uses the linear-attention backbone in the long-video regime. Wrote a phased plan: BEV prototype on the user's DriveLM-Qwen base (fast, in-domain) plus a research-track Temporal-AttnRes prototype on the Kimi-Linear base (architecture-novelty).

### Structural understanding talking points

- **Why BEV gets only 32 tokens** (Q-Former vs raw 200×200=40k cells); the three integration patterns.
- **Why linear attention and token compression are partial substitutes** (they attack the same long-sequence cost; stacking yields less than the sum).
- **Why SATS-CRP transfers to BEV** (region-pooled attention distillation; regions = BEV grid cells).
- **Why GRPO is the correct post-training for AD-VLA** (AutoVLA's +10.6% PDMS, sparse rewards from sim, our infra is portable).
- **Why depth-AttnRes generalizes to time-AttnRes** (the loop axis is the only thing that changes).
- **Why most "world models" are three different architectures** (latent dynamics vs AR video tokens vs latent diffusion — see [`04_world_models.md`](04_world_models.md)).

### The honest weakness

- **No closed-loop sim** in v1 (open-loop trajectory only). Closed-loop is a month+ of sim integration.
- **447M Kimi-Linear backbone is small** for competitive AD; we'd need a follow-on pretrain or a larger base.
- **SANA-WM / GAIA-2 / LeWorldModel-scale WM is out of scope** for single-node. Document as a limitation, not a hidden gap.

## Open questions

- **Modality**: BEV-token, multi-cam video, or hybrid for v1?
- **DriveLM-Qwen state**: eval harness ready? Is 3B the intended size? Are compression methods wired for video/multi-image or single-image only?
- **Closed-loop sim**: in scope or open-loop only?
- **BEV encoder**: train our own or adopt frozen pretrained?
- **Sensor suite**: camera-only or camera+LiDAR?
- **Target**: research demo or path to deployment?
- **If Base B**: wait for a stronger pretrained backbone, or prototype on stage-0/stage-2 ckpts? (Only worth it for the video modality where linear attention wins.)
