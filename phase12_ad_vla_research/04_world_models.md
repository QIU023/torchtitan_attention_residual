# 04 — World models (WM)

> **Scope**: predicting future state — next frame, next latent, or next BEV — conditioned on past observations and optionally an action. Pure VLM (text-out) is in [`01_video_vlm.md`](01_video_vlm.md) / [`02_bev_perception.md`](02_bev_perception.md); pure action-out VLA is in [`03_vla_planning.md`](03_vla_planning.md); end-to-end couplings that fuse WM with VLA are in [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md).

Last revised: 2026-05-17 (new file — added in response to "SANA-WM and LeWorldModel research direction" question).

## What "world model" actually means

The term is over-loaded. Three substantively different architectures all get called "world model":

| Tier | What's predicted | Output head | Pretrain scale |
|---|---|---|---|
| **A. Latent dynamics** (DreamerV3 / TD-MPC2 style) | next latent z(t+1) given (z(t), a(t)) | small MLP / shallow transformer on top of an existing encoder | SFT-scale (small) |
| **B. AR video / token WM** (VideoPoet / LeWorldModel / Cosmos-AR style) | next video token given prior tokens (+ action) | LM head over a video-token vocabulary; needs a separate video tokenizer (MAGVIT-v2 etc.) | 10k+ GPU-hours; ~1B+ video tokens |
| **C. Latent video diffusion WM** (SANA-WM / GAIA-2 / DriveDreamer-2 style) | next latent frames via diffusion conditioned on past + action + (optional) text | DiT / U-Net diffusion sampler with linear or full attention | 50k+ hours driving video, multi-week 100+ GPU-h |

These are **different architectures**, not different scales of the same thing. Tier A reuses a perception encoder + adds a small head. Tier B needs a video tokenizer + an AR LM trained on video tokens. Tier C is its own family entirely (diffusion sampler over latent video).

## Tier A — latent dynamics on top of the current SFT model

**Feasibility on our stack: high. ~1 week.**

The idea: take the current Kimi-Linear + SigLIP + projector + LM (post-stage-2 SFT). Add a small head that consumes `(vision_embeds(t), action_embed(t))` and predicts `vision_embeds(t+1)` (or a small VQ-quantized projection of it).

Loss: MSE on the embeds, optionally + InfoNCE for representation quality. Train on short sequential clips from DriveLM / nuScenes / Ego4D where consecutive frames + ego state are available.

**Why this fits our project specifically:**
- AttnRes already does "softmax-weighted aggregation across an axis" — depth in the original paper. Apply along the **time axis** and you get the same Spatio-Temporal AttnRes idea from [`01_video_vlm.md`](01_video_vlm.md) Tier C, except here it's predicting future, not aggregating past for current.
- The current SFT ckpt is the encoder; the head is small (~1–5M params).
- No new tokenizer; no video corpus pretrain; no diffusion sampler.

**What this is NOT**: it's not a generative world model — you can't sample novel futures from it, only do one-step prediction in latent space. That's still useful for:
- Auxiliary loss during VLA training (model learns dynamics-aware features).
- Counterfactual ranking ("which planned action leads to a future closest to a safe target?") for closed-loop planning.
- Distillation target for a larger downstream WM.

**Interview pitch**: "We added a latent next-state head to our SFT VLM as a dynamics-aware auxiliary task. AttnRes generalizes naturally — depth-AttnRes for current-state aggregation, temporal-AttnRes for next-state prediction. Single ablation: dynamics-aware features lift downstream VLA planning by X."

## Tier B — autoregressive video token WM

**Feasibility on our stack: medium. ~4–6 weeks.**

Replace SigLIP with a video tokenizer (MAGVIT-v2, OmniTokenizer, or Cosmos-Tokenizer). Train Kimi-Linear LM (autoregressive) on `[text_tokens, video_tokens, action_tokens]` sequences to predict the next video token.

**What changes:**
- Vision encoder → video tokenizer (frozen pretrained).
- Vocabulary extended with video and action tokens.
- Pretrain on a video-token corpus (e.g. WebVid, HowTo100M, or driving-specific datasets).
- LM trained ~10× the compute of the current SFT (long sequences).

**Where our stack fits:**
- Kimi-Linear LM is autoregressive — directly reusable.
- KDA's O(N) attention is the right primitive for very long token sequences (1k+ video tokens per second of video).
- AttnRes carrier helps with the long-context aggregation.

**What's missing**: a serious video tokenizer + the pretrain compute. MAGVIT-v2 weights are open; the pretrain compute is the gate.

**Interview pitch**: "Our linear-attention backbone is the right primitive for AR video WM; we sketched the integration with MAGVIT-v2 tokens and demonstrated 1-step rollout, but full pretrain is out of single-node scope."

## Tier C — latent video diffusion WM (SANA-WM / GAIA-2)

**Feasibility on our stack: low. Structural mismatch.**

SANA-WM is structurally close to our stack in *one* sense: it uses linear attention + a deep-compression VAE. Both pieces are in our portfolio (KDA + a potential VAE adapter). But the rest of SANA-WM is a different beast:

- **Diffusion sampler** (DiT-style), not autoregressive — different training objective, different inference.
- **Video output**, not text — different output head, different eval metrics (FVD / FID, not perplexity).
- **Action conditioning via cross-attention**, not in-sequence — different conditioning architecture.
- **Pretrain corpus**: 1.4M video clips × 25 frames @ 768×448 for SANA-WM; GAIA-2 is on the same order. Multi-week training on 8–32 A100s.

What honestly transfers from our stack:
- The fact that linear attention works on long sequences — credibility-only, not code.
- AttnRes as a mechanism — interesting to mention but a non-trivial port.

**Our position**: this tier is out of single-node-5090 scope. The right write-up is *"we evaluated SANA-WM as a long-term direction; the architecture overlap with our linear-attention stack is partial, but the pretrain compute and data scale are out of scope for this project. We recommend Tier A as the production extension and Tier B as the architecture-research bet."*

## Decision matrix

| If you want to... | Pick |
|---|---|
| Demonstrate a research-quality WM extension within 1 week, with our compute | **Tier A** |
| Make a serious architectural bet that justifies Kimi-Linear + AttnRes on long video | **Tier B** (with a partial corpus or a pretrained tokenizer) |
| Build something competitive with SANA-WM / GAIA-2 | **Out of scope.** Document as a limitation. |
| Combine WM with action output for closed-loop AD-VLA | See [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md) — DriveDreamer-2 / GAIA-2 are the relevant references |

## How "WM as auxiliary head" interacts with our other tracks

- **+ [`01_video_vlm.md`](01_video_vlm.md)**: Tier A WM head shares the perception encoder; the head reads `vision_embeds` and predicts next-step. Cleanest pairing.
- **+ [`02_bev_perception.md`](02_bev_perception.md)**: predict next-step BEV cells; clean output space; useful for occupancy forecasting.
- **+ [`03_vla_planning.md`](03_vla_planning.md)**: WM head used as either (a) aux loss during VLA SFT, (b) reward signal for GRPO (low next-state surprise = good plan), or (c) imagination engine for model-based planning.
- **+ [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md)**: where this all converges into end-to-end systems.

## Limitations / open questions

- Tier A's MSE loss in latent space is known to collapse to averages without a contrastive or perceptual term — needs careful loss design.
- Tier B needs a video tokenizer choice and a non-trivial pretrain — even a "small" 100h video corpus is a serious data engineering task.
- WM as reward for GRPO is theoretically elegant but empirically fragile (reward hacking on the surprise term is easy).

See [references.md → World models](references.md#world-models) for citations.
