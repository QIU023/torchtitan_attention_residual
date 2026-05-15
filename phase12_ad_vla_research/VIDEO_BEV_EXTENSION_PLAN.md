# Phase 12 Extension — Video / BEV Perception Pretrain + SFT

> Companion to `AD_VLA_RESEARCH.md`. The base doc covered AD-**VLA** (action
> output) strategy. This doc covers the **CV / VLM / AD perception**
> framing — which is what's actually being asked in interviews and is
> more on-topic for perception-algorithm roles than a complete GRPO demo.

Status: scoping doc. Created 2026-05-15.
Inputs: this repo (Kimi-Linear + AttnRes stack) + the DriveLM-Qwen2.5-VL
LoRA project (`/workspace/DriveLM_VLM_Project`).

---

## 1. Why this framing fits the role better than "complete GRPO demo"

For a CV / VLM / AD-perception algorithm role:

| | Video/BEV perception pretrain + SFT | "Complete GRPO RLHF demo" |
|---|---|---|
| Aligns with role title | ✅ direct match | ⚠️ infra/RL flavor |
| Demonstrates VLM structural understanding | ✅ vision encoder × temporal × LLM | ⚠️ mostly orchestration |
| Demonstrates CV video understanding | ✅ direct | ❌ |
| Demonstrates BEV / 3D-aware perception | ✅ if BEV track | ❌ |
| Demonstrates efficient-VLM craft | ✅ token compression / distillation / quant | ⚠️ tangential |
| AttnRes / linear-attention is **substantive contribution** | ✅ (spatio-temporal long-context, see §4) | ⚠️ only as infra |
| Reproducible result without 7-day RL infra grind | ✅ | ❌ |
| Interview talking points (architecture, ablations, scaling) | ✅ rich | ⚠️ thin on perception |

**Verdict**: route the prototype as **video/BEV perception SFT**, not as
"finish GRPO end-to-end." The GRPO infra work stays in the resume as
"shipped RL post-training pipeline w/ NCCL traffic catalog" — a separate,
adjacent contribution — but it isn't the headline.

---

## 2. Two-repo asset inventory

### From `torchtitan_attention_residual` (this repo)

| Asset | Where | Value for video/BEV perception |
|---|---|---|
| **Block AttnRes** (paper-faithful impl, KDA + MLA + MoE backbone) | `torchtitan/torchtitan/experiments/{attn_res,kimi_linear}` | depth-wise softmax aggregation; trivially generalizes to **temporal-wise** aggregation (§4) |
| **KDA — linear attention O(N)** | same | the natural win for long video sequences (T×spatial → easily 10k+ tokens) |
| **AttnRes inference overlay** (SGLang) — full bf16/fp16/fp8 matrix verified | `/sgl-workspace/sglang` fork + our overlay | inference deployment story for video VLM |
| **NCCL / fabric traffic catalog** (phase7 + phase11 trace_grpo dirs) | `phase7_nccl_traffic_catalog/`, `phase11_rlhf_grpo_infra/rlhf/trace_*` | infra story (separate from perception) |
| **3D / 4D parallelism** | torchtitan submodule | useful for scaling video VLM pretrain (long-context T×S→PP+TP+EP+DP) |
| **Stage-0 LM base @ val 2.78 (target)** | `phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_C/` | a Kimi-Linear LM ready as VLM backbone (small but trained) |

### From `DriveLM_VLM_Project` (the other repo)

| Asset | Where | Value for video/BEV perception |
|---|---|---|
| **Qwen2.5-VL-3B LoRA r16/α32 on DriveLM v1.1** | `checkpoints/baseline/checkpoint-46000` | in-domain driving VLM baseline (single-image CAM_FRONT) |
| **4 visual-token compression methods** | `scripts/visual_compress.py` (224 lines): `_fastervlm`, `_prumerge`, `_pyramiddrop`, `_avg_pool` | apply to surround-cam × temporal video token explosion (precisely where it pays the most) |
| **SATS-CRP — user's own method** | `scripts/visual_compress.py:_crp`, `_crp_merge` + `SATS_VLM_impl_plan_v2.md` | region-aware self-attention distillation; **7B→3B** (not 32B→3B as doc said); regions ⇆ video patches or BEV grid cells |
| **GGUF Q4_K_M quant + llama.cpp deploy** | `models/qwen25vl-3b-drivelm-*-gguf/*.gguf` (1.8GB) | full deployment story for the video VLM artefact |
| **DriveLM v1.1 data pipeline** | `data_processed/{train,val}.json` (377k QA pairs, 696 scenes) | in-domain QA SFT data, image-grounded |
| **nuScenes raw image pipeline** | `data/nuscenes/samples/` (24k images, 6 cams) | **the multi-view raw material is already there** — just not wired yet |
| **YAML-driven training framework** | `scripts/train_lora.py` (724 lines) + `configs/*.yaml` (inheritance) | reusable for new ablations (video/BEV experiments) |
| **B200/GH200/4070Ti config trio** | `configs/gh200.yaml` / `4070ti.yaml` / `4070ti_bf16.yaml` | hardware-portable; the 3B LoRA fits on 4070Ti |
| **Continual-learning roadmap (PR'23 paper continuation)** | `docs/exploration_directions.md` | distillation/KD interview narrative depth |

### Where they complement (the actual integration story)

- **DriveLM has the domain data + the efficient-VLM craft + the
  quantization deployment path.**
- **This repo has the long-context architecture + the inference infra +
  the parallelism plumbing.**
- The **gap on both sides is video / BEV input**. Neither has it wired
  yet. Closing this gap is the phase-12 deliverable.

---

## 3. Proposed phase-12 deliverables (perception-role-targeted)

Three deliverable tiers, each independently shippable:

### Tier A (4–7 days) — Video-DriveLM SFT prototype on Qwen2.5-VL (Base A)

The fastest, most demonstrably-on-topic result.

- **Input**: temporal sliding window 4–8 frames from a single nuScenes
  camera (CAM_FRONT is fine for v1; matches current DriveLM v1.1 setup).
- **Recipe**: take `scripts/train_lora.py` + add a temporal-batch
  collator; vision encoder processes each frame, projector projects
  per-frame, optional **temporal pooling / cross-frame attention** before
  LLM input.
- **Token compression**: wire `_fastervlm` / `_prumerge` / `_pyramiddrop`
  through the temporal-stacked tokens (4–8× tokens before compression).
- **Eval**: held-out DriveLM-nuScenes QA accuracy vs single-frame baseline.
- **Headline story**: "DriveLM single-frame LoRA + temporal extension +
  4× token compression — recovers vs single-frame quality with N× more
  temporal context at the same effective token budget."

### Tier B (1–2 weeks) — BEV-DriveLM SFT prototype

Higher novelty; same time as Tier A but adds a BEV encoder.

- **Input**: nuScenes 6-cam → frozen pretrained BEV encoder (BEVFormer
  or LSS — public weights available) → 200×200 BEV feat → either
  **adaptive-pool structured tokens** (OpenDriveVLA style: scene/track/map
  ~64–256) or **32-query Q-Former** (BEVDriver/VLA-MP style).
- **Apply SATS-CRP** to the BEV-token attention inside the LLM: regions ⇆
  **BEV grid cells** is a natural mapping. This is where the user's
  SATS-CRP method gets a *novel* application (BEV instead of 2D image).
- **Eval**: DriveLM perception/planning QA + ADE/FDE on trajectory if
  added.
- **Headline story**: "BEV-tokenized DriveLM with grid-cell SATS-CRP
  distillation — first application of region-pooled attention KD to BEV
  inputs."

### Tier C (the architecture-research bet) — Spatio-Temporal AttnRes on Base B for long video

This is where **AttnRes "可能奇效"** materializes.

The Block AttnRes mechanism is a **depth-wise** soft attention over
prior block outputs (paper Figure 2). It's a *generic* aggregation —
the "depth" axis is just an axis. The same mechanism, applied along a
**temporal** axis, gives:

**"Temporal AttnRes"**: at each frame `t`, the carrier hidden state is
a softmax-weighted aggregation of frames `[t−K, …, t−1]` using a learned
pseudo-query per layer. This is **structurally identical to depth-wise
AttnRes**, just with the loop axis renamed.

Why this might genuinely matter for video:

1. **Long-context regime is where KDA + AttnRes were designed to win**
   (paper claims the carrier keeps magnitudes bounded across the
   aggregation axis — true for depth, should be true for time).
2. **Temporal pooling is the weak link** in most video VLMs (mean-pool /
   first-token / SlowFast-style hand-designed). Replacing it with a
   *learned, content-aware, depth-tested* aggregation primitive is a
   real architectural contribution.
3. **KDA's O(N) linear attention** is the right kernel for long video
   sequences; AttnRes-on-temporal-axis runs O(T) per layer, vs O(T²) for
   standard self-attention.
4. **Cross-block-AttnRes already handles "limited memory" via the carrier**
   — the same mechanism limits per-frame KV state growth across time.

**Concrete experiment**: take the Kimi-Linear backbone with AttnRes,
add a temporal axis at the projector or first decoder block, evaluate on
a video QA benchmark (Video-MME / EgoSchema / NeXT-QA — public, ungated,
small enough to fit in days). Ablate: (a) standard temporal-pool
baseline, (b) AttnRes along temporal axis, (c) AttnRes along
temporal × depth axes.

**Headline story for an interview**: "Generalized the Kimi Block-AttnRes
depth-aggregation primitive to the temporal axis for video; combined
with KDA linear attention this gives O(T) long-context video VLM with
learned, content-aware temporal aggregation — and we showed it
out-performs mean-pool / first-token baselines by X% on Video-MME at T=K
frames." That's a publishable-shape result, and it's exactly what an
"AD perception researcher with VLM background" pitch needs.

---

## 4. AttnRes generalization to video (technical detail)

### 4.1 What AttnRes actually computes

From `torchtitan/torchtitan/experiments/attn_res/attn_res.py` (paper-faithful):

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = stack(blocks + [partial_block])           # [N+1, B, T, D]
    K = norm(V)                                    # RMSNorm
    logits = einsum("d, n...d -> n...", proj.weight, K)   # per-block scalar
    weights = softmax(logits, dim=0)               # over the N+1 axis
    return softmax_weighted_sum(weights, V)        # convex combo
```

The pseudo-query `proj.weight` is a single learned vector per layer; the
axis being aggregated over is the **block** axis (commits across depth).

### 4.2 Reinterpretation: the aggregation axis is generic

Nothing about the math requires the axis to be depth. Replace `blocks` =
"prior depth commits" with `blocks` = "prior temporal frames" and the
formula is identical:

```python
def temporal_attn_res(frames, current_frame, proj, norm):
    V = stack(frames + [current_frame])           # [T_window, B, S, D]
    K = norm(V)
    logits = einsum("d, t...d -> t...", proj.weight, K)
    weights = softmax(logits, dim=0)              # over the T_window axis
    return softmax_weighted_sum(weights, V)       # convex combo over time
```

Where `S` is the spatial token count per frame. The aggregation
collapses time → a single set of (B, S, D) tokens passed to the LLM.

### 4.3 Why this is more than re-labelling

- **Replaces hand-designed temporal pool** with a learned, per-layer,
  content-aware mechanism. Tests at depth showed it produces bounded
  carrier magnitudes (paper §5) — same property should hold across time
  for a sufficiently trained model.
- **Composes with depth-wise AttnRes** — at each layer you get
  (depth-aggregate, then temporal-aggregate), or vice versa, or a
  tensor-product 2D aggregation. Two orthogonal axes of soft aggregation
  is novel.
- **Composes with KDA O(N) linear attention** — KDA handles the spatial
  within-frame attention cheaply (S can be 196 → 1024); temporal AttnRes
  handles the across-frame aggregation cheaply (T window can be 16+).
  Total compute scales as O(T·S) not O((T·S)²).
- **Bounded memory across time** — the carrier stream replaces unbounded
  KV cache growth with a softmax-weighted convex combination; per-frame
  state is O(D) not O(t·D).

### 4.4 What you'd ablate (interview-ready)

1. Baseline: mean-pool over T frames.
2. First-frame baseline: ignore time (only use frame 0).
3. Cross-frame softmax-attention baseline (T² cost).
4. **Temporal AttnRes** (proposed; T cost per layer).
5. Temporal AttnRes + depth AttnRes (full 2D aggregation).
6. KDA spatial + temporal AttnRes (the full Base-B story).

Eval on Video-MME (short/medium/long splits) + EgoSchema. Report
accuracy × throughput.

---

## 5. Interview narrative (how to talk about the project)

For a "CV VLM AD-perception algorithm" interview:

### The 90-second pitch

"I built two complementary stacks. On the Qwen2.5-VL side I have a
DriveLM-LoRA fine-tune with a four-method visual-token compression
toolkit and a GGUF deployment path — the efficient-VLM-on-driving story.
On the Kimi-Linear side I have a from-scratch Block AttnRes
linear-attention LM-then-VLM stack with multimodal SFT and a full
inference infra. The next step is to merge them in a video/BEV
perception extension: take the DriveLM data + compression toolkit, add a
temporal/BEV scene encoder, and demonstrate that the AttnRes
depth-aggregation primitive *generalizes* to the temporal axis for video
VLM — replacing hand-designed temporal pooling with a learned, bounded
aggregation that runs O(T) per layer and outperforms mean-pool
baselines on Video-MME."

### The structural understanding talking points

- Where vision tokens flow in a VLM: encoder → projector → LLM (and
  where the four compression methods sit between projector and LLM).
- Why pre-norm residual streams grow unbounded with depth, and how
  Block AttnRes' learned convex aggregation bounds them — the same
  argument applies time.
- BEV-token vs multi-cam-video tradeoff (token budget, spatial
  consistency, fine appearance, encoder cost) — §4 of the base doc.
- Q-Former vs adaptive-pool-structured-tokens for BEV → LLM (two main
  patterns in the BEV-VLA literature, both converge on ~32–256 tokens).
- Why fp16/fp8 inference depends on training-time activation magnitudes
  (we ran the matrix; we have the data).

### The CV video classics to namedrop (and why)

If asked about "video CV classics":

| Classic | Why it matters here |
|---|---|
| **Two-Stream** (RGB + optical flow) | precursor to temporal modeling; informs why "first-frame baseline" loses |
| **C3D / I3D** | 3D conv as joint spatio-temporal — but quadratic in T |
| **SlowFast** | dual-path fast/slow framerate decomposition; informs why temporal aggregation should be *learned* per layer (different layers want different temporal scales) |
| **TSM (Temporal Shift Module)** | cheap temporal mixing without extra FLOPs — competitor to temporal AttnRes |
| **Video Swin / TimeSformer / ViViT** | factorized space-time attention — direct ancestor of "temporal-axis self-attention" baselines |
| **Video-MAE / V-JEPA** | self-supervised video pretrain — informs the "video pretrain" half of "video pretrain + SFT" |
| **MViT** | multi-scale pooling — adjacent to AttnRes's per-layer aggregation idea |
| **Q-Former (BLIP-2)** | the standard "small fixed query set" pattern adopted by BEV-VLA works |

### The honest weakness

The Kimi-Linear backbone (Base B) is still in pretrain (~50% of stage
0). The completed result is on Qwen2.5-VL (Base A). For an interview,
this is honest framing: "completed pipeline on Base A (the efficient-VLM
side), architecture research on Base B (the linear-attention side)."

---

## 6. What changes vs `AD_VLA_RESEARCH.md` recommendation

The base doc recommended **BEV-token + Base A** for the v1 prototype, on
infra/economics grounds. This extension doc reaches the same conclusion
for the perception-role pitch — Tier A (Video-DriveLM on Qwen2.5-VL) is
the most demo-able fastest result.

But for the **architecture contribution** angle that matters for an
algorithm interview, **Tier C (Temporal AttnRes on Base B)** is the
substantive bet. It is what makes the answer to "and why should we hire
you for this specifically" go from "I integrated tools" to "I extended
an architecture primitive to a new axis."

Run **Tier A** for the demo, write **Tier C** as the next research
direction with a concrete first experiment. Tier B (BEV) is optional
ambition.

---

## 7. Concrete first-week work plan (if greenlit)

Day 1–2: temporal-DriveLM data loader + 4-frame collator on Qwen2.5-VL +
re-train baseline (no compression) at T=4. Eval on DriveLM val.

Day 3: wire `_fastervlm` 4× compression through the temporal stack.
Compare to single-frame baseline.

Day 4: copy `block_attn_res` from this repo into a small standalone
module; replace temporal pooling with Temporal AttnRes (just one layer
to start); A/B vs mean-pool. (This is the "AttnRes generalization"
ablation.)

Day 5: write up: data flow diagram, ablation table, interview-ready
narrative. Push.

Day 6–7 (optional): redo Day 4 on a public video benchmark
(Video-MME-Short) to get a cleaner public-benchmark number for the
deck.

---

## References (additions to AD_VLA_RESEARCH.md)

### Video CV classics
- I3D — https://arxiv.org/abs/1705.07750
- SlowFast — https://arxiv.org/abs/1812.03982
- TSM — https://arxiv.org/abs/1811.08383
- TimeSformer — https://arxiv.org/abs/2102.05095
- Video Swin — https://arxiv.org/abs/2106.13230
- ViViT — https://arxiv.org/abs/2103.15691
- Video-MAE — https://arxiv.org/abs/2203.12602
- V-JEPA — https://arxiv.org/abs/2404.08471

### Video VLM
- Video-LLaVA — https://arxiv.org/abs/2311.10122
- VideoChat — https://arxiv.org/abs/2305.06355
- LLaVA-NeXT-Video — https://github.com/LLaVA-VL/LLaVA-NeXT
- Video-MME — https://arxiv.org/abs/2405.21075
- EgoSchema — https://egoschema.github.io/

### Video benchmarks (small enough to run)
- Video-MME (Short/Medium/Long, 900 videos) — https://video-mme.github.io
- EgoSchema (5031 clips, 3-min ego video) — https://egoschema.github.io
- NeXT-QA (~5k videos, 47k QA) — https://github.com/doc-doc/NExT-QA
