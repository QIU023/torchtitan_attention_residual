# 03 — VLA (Vision-Language-Action) planning

> **Scope**: extending a VLM to output **action / trajectory** (not just text). Pairs with either [`01_video_vlm.md`](01_video_vlm.md) or [`02_bev_perception.md`](02_bev_perception.md) on the input side. Future-state predictive variants are in [`04_world_models.md`](04_world_models.md).

## Definition

A VLA = VLM backbone + an action-output head. Architectural recipes:

| Action representation | Head | Examples |
|---|---|---|
| **Discretized action tokens** reusing the LM head | text-token vocabulary extended with action tokens | RT-2 / OpenVLA / AutoVLA |
| **Continuous waypoint regression** | small MLP / GRU on the LLM's last hidden state | BEVDriver, VLA-MP |
| **Physics-consistent dynamics head** | bicycle / kinematic model parametrized by LLM output | VLA-MP |
| **Diffusion planner** | latent diffusion conditioned on LLM hidden state | DiffVLA |

## SFT → GRPO maps almost 1:1 to AD-VLA

**AutoVLA** (NeurIPS 2025) is an end-to-end driving VLA trained with exactly SFT + GRPO reinforcement fine-tuning. Reported: GRPO RFT gave **+10.6% PDMS (NAVSIM) and −66.8% runtime**. This is the strongest available evidence that our existing SFT→GRPO infrastructure (phase 5 + phase 11) maps directly onto a driving VLA — the GRPO half is not a "nice to have", it is a published, meaningful AD-VLA capability.

Reward shaping for driving:
- **Open-loop**: trajectory MSE against expert + safety (collision/off-road) penalties + comfort (jerk, lateral accel).
- **Closed-loop** (CARLA / nuPlan / NAVSIM): Driving Score (DS), Route Completion (RC), Infraction Score (IS), PDMS.
- **Rule compliance**: structured penalty for traffic rule violations (red lights, lane changes without indicator).

## Landscape

| Work | Perception | Action | RL? | Bench |
|---|---|---|---|---|
| **AutoVLA** | multi-view frames | trajectory tokens + dual fast/slow | **GRPO RFT** | nuPlan + nuScenes (10k → 185k); CARLA closed-loop |
| **RT-2 / OpenVLA** (robotics) | ViT/DINO/SigLIP + LLM | discretized action tokens | — | canonical VLM→VLA via token reuse |
| **OpenDriveVLA** | BEV (see [`02_bev_perception.md`](02_bev_perception.md)) | trajectory AR | — | nuScenes SoTA-class |
| **VLA-MP** | BEV + Q-Former | GRU bicycle-dynamics head | — | CARLA LangAuto DS 44/64/78 |
| **BEVDriver** | BEV InterFuser | waypoints | — | CARLA LangAuto long-route DS 48.9 |
| **DriveLM-Qwen** (our other project) | single CAM_FRONT image | QA only — no action yet | — | DriveLM-nuScenes QA |
| **DiffVLA** | vision + language | diffusion planner | — | — |

Data scale: AutoVLA used 10k–185k driving samples. OpenDriveVLA/VLA-MP/BEVDriver are nuScenes-/LMDrive-scale (~100k). Our SFT uses 558k (Pretrain) + 665k (Instruct) image–caption pairs. **Same order of magnitude** — driving-data scale is not the bottleneck for a research prototype.

## How our SFT→GRPO infra maps

| Phase 11 component | AD-VLA repurpose |
|---|---|
| `phase5_vlm_multimodal_sft/train_mm.py` | swap caption labels → trajectory tokens; reuse projector + vision encoder |
| `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_caption.py` | swap LlavaCaptionTask → DrivingPlanTask (obs → trajectory, reward = open-loop or closed-loop sim score) |
| `phase11_rlhf_grpo_infra/rlhf/llava_caption_task.py` | template for the driving reward grader (rule-based or sim-based) |
| SGLang multi-modal overlay | inference path for the VLA; unchanged structurally |
| FSDP/PP/TP plumbing | unchanged |

The piece that **isn't portable** is the **driving environment** — the reward source. Open-loop is easy (compare to expert trajectory). Closed-loop needs a sim wrapper (CARLA leaderboard 2.0, nuPlan, NAVSIM, or LMDrive's LangAuto). This is the biggest engineering unknown.

## Tiered deliverables

### Tier A — open-loop trajectory SFT (1 week)
- nuScenes scenes → (obs, ego state, nav cmd) → expert trajectory.
- Tokenize trajectory as N waypoints; train SFT to predict.
- Eval: ADE/FDE vs expert.
- **Headline**: VLM → VLA via action-token SFT; reuses phase 5 stack with only a label swap.

### Tier B — open-loop GRPO RFT (2 weeks)
- On top of Tier A: GRPO with reward = `−trajectory_MSE + collision_penalty + comfort_bonus`.
- Reuses `phase11_rlhf_grpo_infra/rlhf/` with a `DrivingPlanTask` swap.
- Eval: PDMS-style aggregate.
- **Headline**: SFT→GRPO recipe directly reproduces AutoVLA-style RFT on a smaller (3B or 447M) backbone.

### Tier C — closed-loop GRPO RFT in sim (4–6 weeks)
- Wrap CARLA-LangAuto or nuPlan sim into a Generator that yields trajectories conditioned on rollout state.
- GRPO with closed-loop DS/RC/IS as reward.
- Eval: CARLA leaderboard 2.0 or nuPlan val14 score.
- **Risk**: sim integration is the big lift; the model side is incremental.

## Pairing with perception modality

| If perception is... | VLA recipe |
|---|---|
| [`01_video_vlm.md`](01_video_vlm.md) (video) | AutoVLA pattern; trajectory tokens AR; compression mandatory |
| [`02_bev_perception.md`](02_bev_perception.md) (BEV) | OpenDriveVLA / VLA-MP / BEVDriver pattern; small fixed token budget; GRU/bicycle dynamics head pairs naturally |
| [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md) (BEV-injection hybrid) | BEV-InMLLM + AutoVLA head |

## Limitations / open questions

- Closed-loop sim is the real engineering lift; open-loop is 1-week scope, closed-loop is 1-month+.
- Reward design is non-trivial: open-loop MSE alone over-rewards copying expert; need safety/comfort terms.
- Discretized action tokens are simpler but lose precision; physics-constrained heads (VLA-MP) are more research-y.

See [references.md → VLA](references.md#vla4ad--general--video-track) for citations.
