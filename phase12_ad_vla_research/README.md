# Phase 12 — AD perception / VLA / WM research

Last revised: 2026-05-17 (restructured into per-topic files).

## What this folder is

A scoping doc set for extending the project beyond pure VLM SFT/GRPO into
autonomous-driving-relevant capabilities. Four independent topics, plus
a section on where they couple end-to-end.

## Topic files

| File | Topic | When to read |
|---|---|---|
| [`01_video_vlm.md`](01_video_vlm.md) | **Video VLM** — multi-cam + temporal frame input | Considering long-context vision input that exercises Kimi-Linear's KDA |
| [`02_bev_perception.md`](02_bev_perception.md) | **BEV perception** — bird's-eye-view tokens into the LLM | Considering metric/spatial grounding with a small fixed token budget |
| [`03_vla_planning.md`](03_vla_planning.md) | **VLA planning** — language → trajectory / action output | Considering closing the loop with planning, e.g. AutoVLA-style SFT+GRPO |
| [`04_world_models.md`](04_world_models.md) | **World models** — predict future state/frame (NEW 2026-05-17) | Considering future-state prediction as an auxiliary head or replacement objective |
| [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md) | **Couplings** — when 2+ of the above tie together as end-to-end systems | E.g. DriveDreamer-2 (VLA+WM), GAIA-2 (video+WM+BEV), AutoVLA (video+VLA) |
| [`06_strategy_workplan.md`](06_strategy_workplan.md) | **Strategy + workplan** — base choice, asset inventory, phased plan, interview pitch | Deciding what to actually build first |
| [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md) | **Time-AttnRes initial direction** — concrete code-level recipe for Patterns A/B, validation experiments, risk register | Implementing Time-AttnRes (not surveying alternatives) |
| [`references.md`](references.md) | Consolidated bibliography | When you need a paper link |

## Topic relationship graph

```
                          ┌────────────────┐
              ┌───────────┤  03 VLA        │───────────┐
              │           │  (action head) │           │
              │           └────────┬───────┘           │
              │                    │                   │
       perception inputs       planning           future state pred
              │                    │                   │
   ┌──────────┴────┐               │           ┌───────┴────────┐
   │ 01 video VLM  │◄──────────────┼──────────►│ 04 world model │
   │ (multi-cam,   │               │           │ (next-frame/   │
   │  temporal)    │               │           │  latent pred)  │
   └───────┬───────┘               │           └───────┬────────┘
           │                       │                   │
           │                       ▼                   │
   ┌───────┴───────┐    ┌─────────────────┐   ┌───────┴────────┐
   │ 02 BEV percp. │◄──►│ 05 end-to-end   │◄──┤  GAIA-2,       │
   │ (BEVFormer →  │    │ couplings       │   │  DriveDreamer-2│
   │  Q-Former)    │    │ (DriveVLM, etc) │   │  AutoVLA       │
   └───────────────┘    └─────────────────┘   └────────────────┘
```

- **01 ↔ 02** are sibling perception modalities (video vs BEV), can be fused.
- **03** is the action-output head that sits on top of either.
- **04** can be (a) a standalone aux head, (b) the entire objective (LeWorldModel style), or (c) the rollout engine for closed-loop VLA training.
- **05** is where the system becomes end-to-end and the modules stop being independently swappable.

## TL;DR positions

- **Lowest-effort, most-on-topic for a perception role**: Tier A in [`01_video_vlm.md`](01_video_vlm.md) — 4-frame temporal SFT on top of an existing in-domain VLM. ~1 week.
- **Highest novelty without scaling out**: Tier A in [`04_world_models.md`](04_world_models.md) — add a latent-dynamics head ("next vision-embeds") to the current SFT ckpt. Reuses AttnRes idea as Spatio-Temporal AttnRes.
- **Highest-ceiling architectural bet**: Spatio-Temporal AttnRes as the core mechanism in a long-context video VLM (Tier C of 01) — Kimi-Linear's KDA + AttnRes is genuinely well-matched here.
- **Out of scope for a single-node 5090 setup**: SANA-WM / GAIA-2 / LeWorldModel-scale video pretraining. Document the limitation, don't try.

Detailed base-choice, asset inventory, and work-plan are in [`06_strategy_workplan.md`](06_strategy_workplan.md). The code-level recipe for the Time-AttnRes architecture extension is in [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md) — that's an *initial-direction* doc (not a survey).

## Doc-type convention

| Type | Purpose | Tone | Examples here |
|---|---|---|---|
| **Survey / research** | Landscape, citations, comparison tables | descriptive | `01`–`05` topic files |
| **Strategy** | Choices, tradeoffs, recommendation | argumentative | `06_strategy_workplan.md` |
| **Initial direction** | Concrete recipe, code stubs, experiment plan | actionable | `07_time_attnres_recipe.md` |

Initial-direction docs are subject to revision as experiments run; they are *not* the same as research surveys.
