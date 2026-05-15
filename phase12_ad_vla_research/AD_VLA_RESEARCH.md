# Phase 12 — Autonomous-Driving VLA: Research & Strategy

Status: scoping doc. Created 2026-05-14. Revised 2026-05-15
(added BEV-VLA track + DriveLM-Qwen experience integration).

Question: can the current multimodal Kimi-Linear + SigLIP stack (SFT→GRPO pipeline)
extend into a Vision-Language-Action (VLA) model for autonomous driving?
Two sub-questions, now treated explicitly:
1. **Spatial modality** — go with **multi-cam video** (per-view encode + temporal
   frames), or with **BEV (bird's-eye-view) raster/features** as the scene token?
2. **Base** — extend from the existing **DriveLM Qwen2.5-VL LoRA** project,
   or from **this** (Kimi-Linear) project?

---

## 1. Bottom line

- **Feasible and well-trodden.** VLA-for-AD (VLA4AD) is an active field with a
  dedicated survey + "awesome" repos. The VLM→VLA extension is standard.
- **The SFT→GRPO recipe maps almost 1:1.** AutoVLA (NeurIPS 2025) is an end-to-end
  driving VLA trained with exactly SFT + GRPO reinforcement fine-tuning. RFT gave
  it +10.6% PDMS (NAVSIM) and −66.8% runtime. The GRPO half of our pipeline is
  not a "nice to have" — it is a published, meaningful AD-VLA capability.
- **Two viable spatial modalities, not one.** The original draft only considered
  multi-cam video. The literature is in fact split: a large cluster of strong
  AD-VLA/LLM4AD works (OpenDriveVLA, VLA-MP, BEVDriver, Talk2BEV, BEV-InMLLM)
  feed a **BEV representation** into the LLM instead of raw multi-view frames.
  BEV is **spatially consistent, ego-centred, and token-cheap** (~32–256 tokens
  vs ~5k–12k for surround video). See §4.
- **Recommendation (revised — see §3).** Two facts now dominate:
  (a) BEV gives a small, fixed token budget, which **shrinks** Kimi-Linear's
  linear-attention advantage in the near term; and (b) the user already has
  **deep compression / distillation / quantization assets on the Qwen2.5-VL
  side** (the DriveLM-Qwen project — SATS-CRP distillation, 4 token-compression
  methods, GGUF quant). Both push the **near-term prototype toward Base A
  (DriveLM Qwen2.5-VL)**, especially in a **BEV-token configuration** where the
  quadratic-attention cost is no longer the deciding factor. Kimi-Linear remains
  the long-term architecture bet — but its win condition is specifically the
  **multi-cam-video** regime, not BEV.

---

## 2. VLA4AD landscape (2023–2026)

The architecture pattern is consistent: **VLM backbone + action/trajectory head**,
optionally + RL fine-tuning in (open- or closed-loop) sim. The works split by
**what spatial modality enters the LLM**.

### 2a. Video / multi-view-token track

| Work | Backbone / vision | Action output | RL? | Notes |
|------|-------------------|---------------|-----|-------|
| **AutoVLA** (NeurIPS 2025, UCLA) | VLM, multi-view frames | trajectory tokens; dual fast/slow "thinking" | **GRPO RFT** | nuPlan+nuScenes (10k→185k samples), open- & closed-loop (CARLA). Directly validates SFT→GRPO. |
| **RT-2 / OpenVLA** (robotics, not AD) | ViT/DINOv2/SigLIP + LLM | discretized action tokens (reuse LM head) | — | the canonical "VLM→VLA via action tokenization" pattern |
| **DriveLM-Qwen** (the user's project) | Qwen2.5-VL 3B, multi-image, LoRA r16 | driving QA (perception/planning) | — | in-domain VLM fine-tune; carries the compression+distillation assets in §6 |

### 2b. BEV-token track (new section — detail in §4)

| Work | BEV encoder → LLM | BEV tokens | Action output | RL? | Notes |
|------|-------------------|-----------:|---------------|-----|-------|
| **OpenDriveVLA** | per-view BEV feat → 2D adaptive-pool → `<SCENE>/<TRACK>/<MAP>` tokens via per-type 2-layer MLP; Qwen2.5-Instruct, **full-param** | scene+agent+map, O(10²) | waypoints tokenized to text, autoregressive | — | hierarchical 2D+3D tokens; agent-env-ego interaction; nuScenes open-loop SoTA-class |
| **VLA-MP** | RGB+LiDAR → BEV fusion encoder → **BLIP-2 Q-Former** projector → LLM | Q-Former queries (~32) | GRU-bicycle dynamics adapter → physics-consistent trajectory | — | 3-stage training (perception pretrain → BEV-language align → joint); CARLA LangAuto DS 44.3/63.5/78.4 |
| **BEVDriver** | InterFuser BEV enc (ResNet-50 img + ResNet-18 LiDAR, 5.37M params, 256-d) → **modified Q-Former, 32 query tokens @768-d** → LLM (**LoRA r16**) → GRU waypoint head | **32** | waypoints | — | closed-loop CARLA LangAuto: long-route DS 48.9 / RC 59.7, beats AD-H by +11.1 DS |
| **BEV-InMLLM** | frozen LSS / BEVFormer (200×200) → **instruction-aware BEV Q-Former** (BEV queries ⊕ instruction tokens, cross-attn) → residual-fused into a multi-view MLLM | **32** queries | QA (perception/prediction/planning) | — | plug-and-play "BEV injection"; +4.1 pt on spatial tasks, +2.1 pt holistic |
| **Talk2BEV** | BEV map from cameras+LiDAR, per-object VL features | object-level | language QA over BEV | — | language-enhanced BEV map interface; ICRA'24 |

Data scale takeaway: AutoVLA used **10k–185k** driving samples; OpenDriveVLA/
VLA-MP/BEVDriver are nuScenes- / LMDrive-scale. Our current SFT uses **558k**
image-caption pairs. **Same order of magnitude** — driving-data scale is *not*
the bottleneck for a research prototype.

---

## 3. Two candidate bases — strategic comparison (revised)

### Base A — DriveLM Qwen2.5-VL LoRA (the other project)

| | |
|---|---|
| **Domain** | ✅ already a *driving* VL fine-tune (DriveLM-nuScenes QA/perception) |
| **Vision** | ✅ Qwen2.5-VL: native dynamic resolution + **native video** + multi-image — close to multi-cam/temporal driving input out of the box; also a clean home for a **BEV-raster-as-image** input |
| **Serving** | ✅ sglang has **first-class Qwen2.5-VL support** — "no sglang pipeline" is cheaply fixable |
| **Post-training** | ❌ no GRPO pipeline today — but GRPO infra is **portable** (the actor/rollout/reward loop in `phase11_rlhf_grpo_infra/rlhf` is largely model-agnostic; the hard part — the driving env wrapper — is needed either way) |
| **Compression / distillation assets** | ✅ **NEW & decisive.** The DriveLM-Qwen project already carries: SATS-CRP region-aware self-attention distillation (Qwen2.5-VL 32B→3B), 4 integrated token-compression methods (FasterVLM, PruMerge, PyramidDrop, SATS-CRP) giving 4×–16× visual-token reduction with ≤2.4% accuracy loss, and a BF16→GGUF Q4_K_M quantization path. See §6. None of this exists on Base B. |
| **Adaptation depth** | ⚠️ LoRA only — limits how much the backbone can reshape for an action head; may need full-param or larger LoRA rank for the action modality (note: OpenDriveVLA used **full-param** Qwen2.5; BEVDriver kept **LoRA r16** + GRU head and still got closed-loop SoTA-class — LoRA is *not* disqualifying for BEV) |
| **Attention** | ⚠️ standard quadratic attention — costly for long multi-cam-video token sequences; **but largely a non-issue in the BEV-token regime** (~32–256 tokens) |

### Base B — current Kimi-Linear + SigLIP (this repo)

| | |
|---|---|
| **Domain** | ❌ never seen driving data; SigLIP is single-image 224² — multi-cam/temporal or BEV encoder must be built from scratch |
| **Post-training** | ✅ SFT→GRPO pipeline + sglang overlay already built and (as of phase 11) debugged: 4D-mesh TP fix, val-loss wiring, checkpoint-save fix all landed |
| **Architecture fit** | ✅ KDA **linear attention is O(N)** — wins specifically in the **long multi-cam-video** regime (5k–12k vision tokens); AttnRes carrier also aids long-context aggregation. ⚠️ **In the BEV-token regime this advantage is small** — at ~32–256 tokens the attention term is negligible either way. |
| **Compression assets** | ⚠️ none ported; and see §6 — token compression has **lower marginal value** on a linear-attention backbone, so Base B benefits less from the user's compression toolkit anyway |
| **Training** | ✅ full-param; ⚠️ 447M active is small for competitive AD (fine for a prototype) |
| **Maturity** | ⚠️ the model itself is still in stage-0 pretrain (~step 3.4k/12.75k) |

### Recommendation (revised)

**Near-term AD-VLA prototype → start from Base A (DriveLM Qwen2.5-VL), in a
BEV-token configuration.** Three reinforcing reasons:
1. **In-domain + video/BEV-capable + sglang-served.** Fastest path to a working
   open-loop driving VLA; porting GRPO is tractable engineering, not research risk.
2. **The user's compression/distillation/quantization stack lives here.** SATS-CRP
   distillation, the 4-method token-compression integration, and the GGUF quant
   path are all Qwen2.5-VL-side assets. Reusing them on Base A is near-zero-cost;
   re-deriving them on Base B is a research project (§6).
3. **BEV neutralises Base A's one real weakness.** The quadratic-attention cost
   was the headline argument for Base B — but it only bites in the long-video
   regime. With BEV tokens (~32–256), the attention term is small, so Base A's
   quadratic attention is no longer a meaningful disadvantage.

**Keep Base B (Kimi-Linear) as the architecture research track — with a sharper
win condition.** Linear attention is the correct long-term bet **specifically for
the multi-cam-video modality** (surround-cam × temporal → 5k–12k tokens). It is
*not* a differentiator for BEV. Revisit Base B as the primary base when (a) the
project commits to dense multi-cam-video input rather than BEV, (b) stage-0 +
follow-on pretrain produce a strong enough backbone, and (c) the Base-A prototype
has proven the data/reward/eval loop.

Key reframe (updated): **the GRPO+sglang pipeline is portable; the
linear-attention architecture is not — but neither is the user's
compression/distillation stack.** Two non-portable asset piles now sit on
*opposite* bases. For a near-term BEV prototype the compression stack matters
more than linear attention, because BEV already solves the token-budget problem
that linear attention was meant to solve. Hence the tilt toward Base A.

What we'd still want to confirm: DriveLM-Qwen eval-harness state, whether the
3B is the intended size, and whether closed-loop sim (CARLA/nuPlan) is in v1 scope.

---

## 4. BEV-VLA track — how BEV enters the LLM (new section)

### 4.1 Why BEV at all

Separately encoding 6–8 surround cameras and concatenating tokens **does not
model spatial consistency** — the LLM has to re-derive "where things are"
from view-tagged 2D tokens. A BEV representation is **ego-centred, metric, and
spatially consistent by construction**: object location/scale/distance are
explicit, which is exactly what distance- and planning-sensitive driving tasks
need (BEV-InMLLM reports +4.1 pt on spatial tasks from BEV injection alone).
BEV is also the native interface to HD-maps and to most existing AD perception
stacks.

### 4.2 The three integration patterns

**(A) BEV-feature → projector → tokens (the dominant pattern).**
A BEV encoder (BEVFormer / LSS / InterFuser-style) produces a dense BEV feature
map — BEVFormer's default is a **200×200** grid over a 102.4 m × 102.4 m
ego-centred area (0.512 m/cell), 6 deformable-attention encoder layers. That
dense map is **far too large to feed raw** (40k cells), so every BEV-VLA work
compresses it before the LLM:
- **Adaptive pooling + per-type MLP** (OpenDriveVLA): 2D-adaptive-pool each view's
  BEV feature, concat into a global `<SCENE>` token; separate `<TRACK>` (agents)
  and `<MAP>` tokens; each token type gets its own 2-layer GeLU MLP into language
  space. Result: O(10²) structured tokens.
- **Q-Former** (VLA-MP, BEVDriver, BEV-InMLLM): a BLIP-2-style Q-Former with a
  small fixed set of **learnable BEV queries** cross-attends the BEV feature map.
  All three converge on **32 query tokens** (BEV-InMLLM ablates 64 → diminishing
  returns). Optionally instruction-conditioned (BEV-InMLLM concatenates
  instruction tokens with BEV queries so extraction is query-aware).

**(B) BEV-raster-as-image.** Render the BEV (semantic/occupancy/HD-map raster)
as an image and feed it through the *existing* vision encoder — zero new encoder,
but throws away metric precision and needs the encoder to relearn top-down
semantics. Useful as a cheap baseline; not what the strong works do.

**(C) BEV injection / residual fusion** (BEV-InMLLM). Keep the multi-view token
stream as primary; use the BEV Q-Former output to **residually fuse** spatial
awareness back into the multi-view tokens. Plug-and-play; gives the spatial
benefit without replacing the vision encoder. This is the natural "video + BEV
hybrid" — relevant if we don't want to choose.

### 4.3 Token budget — BEV vs multi-cam video

| Modality | Tokens into the LLM | vs current (~196) |
|----------|--------------------:|------------------:|
| Current single-image VLM | ~196 | 1× |
| BEV via Q-Former (VLA-MP/BEVDriver/BEV-InMLLM) | **~32** | ~0.2× |
| BEV via pooled structured tokens (OpenDriveVLA: scene+track+map) | **~64–256** | ~0.3–1.3× |
| Multi-cam (6–8) × temporal (4–8 frm) × ~196, no compression | **~5k–12k** | ~25–60× |
| Multi-cam × temporal **with 4×–16× compression** (user's toolkit) | **~0.3k–3k** | ~1.5–15× |

**Implication:** BEV is **1–2 orders of magnitude cheaper** in tokens than raw
surround-video. It also makes the token count **fixed and scene-size-independent**
— it does not grow with the number of cameras or frames (temporal context is
folded into the BEV encoder, e.g. BEVFormer's recurrent BEV). The price: a
**heavier, separately-trained BEV encoder**, and BEV quality becomes a hard
dependency (frozen BEVFormer/LSS is the common choice — VLA-MP pretrains the
perception module in a dedicated Stage 1).

### 4.4 BEV vs video — when to pick which

| | BEV-token | Multi-cam video |
|---|---|---|
| Token budget into LLM | ✅ tiny, fixed (~32–256) | ❌ large, grows with cams×frames |
| Spatial consistency / metric grounding | ✅ explicit, ego-centred | ⚠️ implicit, LLM must infer |
| HD-map / occupancy fusion | ✅ native | ⚠️ awkward |
| Fine appearance detail (signs, lights, text) | ❌ lost in BEV projection | ✅ preserved |
| New components to build | ❌ a full BEV encoder + perception pretrain | ✅ reuse VLM vision encoder; mainly a temporal/multi-image adapter |
| Linear-attention (Kimi-Linear) payoff | ⚠️ small — token budget already tiny | ✅ large — this is the regime KDA wins |
| Fit to user's compression toolkit | ⚠️ less needed (already few tokens) | ✅ directly applicable (4×–16× on the big stream) |

**Pragmatic read:** BEV-token + Base A is the **fastest, cheapest, most
spatially-grounded** prototype, and it sidesteps both the quadratic-attention
problem and the need for the compression toolkit. Multi-cam video + Base B is
the **higher-ceiling, appearance-rich, longer-horizon** bet where *both*
linear attention *and* the compression toolkit pay off. A **BEV-injection hybrid
(pattern C)** is the honest "don't choose yet" option.

---

## 5. Architecture mapping: VLM → AD-VLA (applies to either base)

| Component | Current (image-caption VLM) | AD-VLA target — video track | AD-VLA target — BEV track |
|-----------|------------------------------|------------------------------|----------------------------|
| Vision | single image → encoder → projector → tokens | **multi-view (6–8 cams) + temporal (4–8 frm)**; per-view encode + concat, with token compression on the big stream | **BEV encoder** (BEVFormer/LSS/InterFuser) → pooled structured tokens or Q-Former queries → projector |
| Extra inputs | — | ego state (speed/accel/yaw), nav command | same + native HD-map / occupancy fusion |
| Backbone | LM | unchanged (VLM backbone reused) | unchanged |
| Output head | LM head → text | trajectory: (a) discretized action tokens reusing LM head (RT-2/AutoVLA), (b) waypoint regression, (c) physics-constrained head | same; VLA-MP/BEVDriver both pair BEV with a **GRU / bicycle-dynamics** waypoint head |
| Post-train | SFT on captions → GRPO on text reward | SFT on (obs→trajectory) → GRPO with driving reward | same; VLA-MP shows a clean **3-stage** schedule (perception pretrain → BEV-language align → joint) that maps onto our SFT stages |

---

## 6. Reusing the DriveLM-Qwen project's compression + distillation assets (new section)

The DriveLM-Qwen project (Qwen2.5-VL 3B, LoRA r16, DriveLM-nuScenes) produced
four reusable asset classes. How each transfers to an AD-VLA effort:

### 6.1 The assets

1. **SATS-CRP — region-aware self-attention distillation.** Distills the
   *class-region-pooled self-attention scores* over visual tokens inside the LLM
   decoder layers (Qwen2.5-VL 32B teacher → 3B student). Gives a denoised
   inter-region visual-relationship signal. SATS + LLaVA-KD baseline beat the
   baseline KD by ~1% on DriveLM LoRA accuracy.
2. **4-method visual-token compression** (FasterVLM, PruMerge, PyramidDrop,
   SATS-CRP) integrated between the vision encoder and the LLM — **4× reduction
   (480→120 tokens)** with maintained or +1% accuracy.
3. **Profiled accuracy–compression tradeoff** across 4×–16×: at the **16×
   extreme (480→30 tokens)** only **−2.4%** accuracy.
4. **BF16→GGUF Q4_K_M quantization** — 3.9× size reduction, deployment path.

### 6.2 Transfer to Base A (DriveLM Qwen2.5-VL) — near-zero cost

All four assets are Qwen2.5-VL-native and transfer **directly**:
- In the **video track**, the 4×–16× compression toolkit is *exactly* the lever
  that makes Base A's quadratic attention affordable on 5k–12k-token surround
  video — it converts the ~600–3600× attention blow-up (see §7) down toward the
  ~40–230× range. This is the single biggest reason Base A's quadratic-attention
  weakness is *mitigable*, and the user already owns the mitigation.
- In the **BEV track**, the BEV encoder already emits only ~32–256 tokens, so the
  *compression* toolkit is largely redundant — but **SATS-CRP-style attention
  distillation still transfers**: distill the BEV-token inter-region attention
  from a larger teacher (e.g. Qwen2.5-VL 32B or a strong BEV-VLA) into the 3B
  student. SATS-CRP's "region-pooled attention" is conceptually a perfect match
  for BEV, where regions = BEV grid cells / map elements.
- The GGUF Q4_K_M path gives a deployable artefact for free.

### 6.3 Transfer to Base B (Kimi-Linear + AttnRes) — partial, and lower-value

This is the key nuance. **Token compression has a different marginal value on
linear vs quadratic attention:**
- **On Qwen2.5-VL (quadratic).** Cutting tokens N→N/4 cuts the attention term
  ~16× (∝N²). Compression is *super-linearly* rewarding — which is why the user's
  4×–16× results are so strong on Qwen2.5-VL.
- **On Kimi-Linear KDA (linear, O(N)).** Cutting tokens N→N/4 cuts the
  attention/state term only **~4× (∝N)**. The MLP/projection FLOPs (also ∝N)
  scale the same on both — so compression still helps Base B *linearly*, but it
  loses the super-linear bonus. The headline "16× compression, −2.4%" tradeoff
  was *measured on quadratic attention*; on linear attention the compute saved
  per accuracy point is smaller. **Token compression and linear attention are
  partial substitutes** — they both attack the same long-sequence cost, so
  stacking them yields less than the sum.
- **Practical consequence.** Base B benefits less from the compression toolkit
  precisely because KDA already gives it O(N). And porting the toolkit to Base B
  is not free: FasterVLM keys off the vision encoder's `[CLS]` attention (SigLIP
  has no `[CLS]` in the same form as Qwen2.5-VL's ViT — needs re-derivation);
  PruMerge/PyramidDrop are method-portable but need re-tuning; SATS-CRP's
  *intra-decoder* attention distillation **does not directly apply** to KDA
  layers, which have no softmax attention map to pool — it would have to be
  re-formulated against KDA's state/decay structure or restricted to the
  AttnRes/full-attention sublayers. That re-formulation is a genuine research
  task, not a port.
- **What *does* transfer to Base B:** the *idea* of region-pooled relational
  distillation, the accuracy–compression *evaluation methodology*, and the GGUF
  quant discipline. The *implementations* mostly do not.

### 6.4 Net effect on strategy

The compression/distillation stack is a **second non-portable asset pile, sitting
on Base A** — and it is most valuable exactly where Base A is weakest (long-video
quadratic attention) and least needed where Base B is strong (BEV / linear
attention). This is a substantive new weight toward **Base A for the near-term
prototype**, and it specifically argues that *if* the project goes the
multi-cam-video route, Base A + the compression toolkit may be competitive with
Base B + linear attention — the very comparison the original draft assumed Base B
would win.

---

## 7. Compute estimates

- **Data scale**: not the bottleneck — AD-VLA papers use 10k–185k driving
  samples vs our 558k captions (same OOM).
- **Dominant multiplier = vision tokens per sample**, and this now **depends on
  the chosen modality**:

  | Configuration | Tokens/sample | Per-step compute vs current SFT |
  |---|---:|---|
  | **BEV track** (~32–256 tokens) | ~0.2–1.3× | **~0.2–1.5×** — essentially free; quadratic vs linear attention *irrelevant* at this scale |
  | **Video, linear attn (Base B), no compression** | ~25–60× | ~25–60× (FLOPs ∝ N) |
  | **Video, quadratic attn (Base A), no compression** | ~25–60× | **~600–3600×** in attention blocks (∝ N²) |
  | **Video, quadratic attn (Base A), + user's 4× compression** | ~6–15× | **~40–230×** in attention blocks — the compression toolkit is what makes this tractable |
  | **Video, quadratic attn (Base A), + user's 16× compression** | ~1.5–4× | **~2–14×** — at the −2.4% accuracy cost the user profiled |

- **Backbone size**: AutoVLA-class systems are 3–7B; OpenDriveVLA/VLA-MP/BEVDriver
  are in that range. The DriveLM-Qwen 3B sits right in the sweet spot.
  447M–1B (Base B) is fine for a research prototype, competitive systems want
  3–7B (+~10–15×).
- **Verdict**:
  - **BEV-track research prototype** (3B backbone, nuScenes-scale, open-loop
    trajectory): **comfortably feasible on the current 8× RTX 5090**, days of
    training — the cheapest of all options because the token budget is tiny.
  - **Video-track research prototype** (≤1–3B, open-loop): feasible, but only
    with frame subsampling and/or the user's compression toolkit.
  - **Closed-loop GRPO** (CARLA/nuPlan in the rollout loop): adds significant
    orchestration + compute; AutoVLA shows it is doable.
  - **Competitive system** (3–7B, large-scale logs, closed-loop): needs a
    cluster, ~100–1000× current.

---

## 8. Work breakdown (phased)

1. **Decide modality + base.** Recommended default: **BEV-token + Base A** for
   the v1 prototype (cheapest, in-domain, sidesteps quadratic-attention cost and
   the need for the compression toolkit). Keep **video + Base B** as the research
   track, and **BEV-injection hybrid** as the fallback if the choice can't be made.
2. **Driving data pipeline** — nuScenes / nuPlan / Waymo / LMDrive are open;
   build a driving-data variant of the multimodal loader (obs + ego + cmd →
   trajectory label). For BEV: decide on a BEV encoder (BEVFormer/LSS/InterFuser)
   and whether to use a frozen pretrained one.
3. **Scene encoder** — the biggest new component:
   - *BEV track*: integrate + (pre)train a BEV encoder; choose the projector
     (pooled structured tokens à la OpenDriveVLA, or a 32-query Q-Former à la
     VLA-MP/BEVDriver/BEV-InMLLM).
   - *Video track*: multi-view temporal adapter on the existing vision encoder;
     **wire in the user's 4×–16× token-compression toolkit from day one** — on
     Base A it is what makes the token budget affordable.
4. **Action head + tokenization** — start with discretized action tokens (reuse
   LM head); BEV works pair well with a **GRU / bicycle-dynamics** waypoint head.
5. **SFT** on (observation → trajectory). For BEV, follow VLA-MP's **3-stage**
   schedule (perception pretrain → BEV-language alignment → joint) — it maps
   cleanly onto our SFT staging.
6. **(Optional) distillation** — apply **SATS-CRP-style region-pooled attention
   distillation** from a larger teacher into the prototype; on BEV the "regions"
   are grid cells / map elements, a natural fit.
7. **GRPO** — port the `phase11_rlhf_grpo_infra/rlhf` actor/rollout/reward loop; design the
   driving reward (safety / comfort / progress / rule-compliance); open-loop
   reward first, then closed-loop sim wrapper.
8. **Eval harness** — open-loop (ADE/FDE, PDMS-style) then closed-loop
   (CARLA/nuPlan LangAuto-style: Driving Score / Route Completion).
9. **(Optional) deployment** — reuse the BF16→GGUF Q4_K_M quant path for a
   deployable artefact.

---

## 9. Open questions

- **Modality**: BEV-token, multi-cam-video, or BEV-injection hybrid for v1?
  (This doc recommends BEV-token for the prototype, but it is a real choice.)
- DriveLM-Qwen project: eval-harness state? Is 3B the intended size? Is the
  compression toolkit currently wired for video/multi-image or single-image only?
- Closed-loop sim in scope, or open-loop trajectory prediction only for v1?
- BEV encoder: train our own, or adopt a frozen pretrained BEVFormer/LSS? What
  sensor suite — camera-only, or camera+LiDAR (VLA-MP/BEVDriver use both)?
- Target: research demo, or a path toward a deployable system?
- If Base B (Kimi-Linear): wait for a stronger pretrained backbone, or prototype
  on the current stage-0 checkpoint? And — only worth it if committing to the
  **video** modality, since BEV erases Base B's linear-attention edge.

---

## References

### VLA4AD — general / video track
- A Survey on Vision-Language-Action Models for Autonomous Driving (ICCVW 2025) — https://arxiv.org/pdf/2506.24044
- AutoVLA (NeurIPS 2025) — https://arxiv.org/abs/2506.13757 ; https://autovla.github.io
- Awesome-VLA4AD — https://github.com/JohnsonJiang1996/Awesome-VLA4AD
- Awesome-LLM4AD (LLM/VLM/VLA/World-Model for AD) — https://github.com/Thinklab-SJTU/Awesome-LLM4AD
- VLA Models for Autonomous Driving: Past, Present, and Future — https://arxiv.org/html/2512.16760v2

### BEV-VLA track
- OpenDriveVLA — https://arxiv.org/abs/2503.23463 ; https://arxiv.org/html/2503.23463v2 ; https://drivevla.github.io/
- VLA-MP — https://www.mdpi.com/1424-8220/25/19/6163 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12526522/
- BEVDriver: Leveraging BEV Maps in LLMs for Robust Closed-Loop Driving — https://arxiv.org/html/2503.03074v1
- Holistic AD Understanding by BEV-Injected Multi-Modal Large Models (BEV-InMLLM / NuInstruct) — https://arxiv.org/html/2401.00988v1
- Talk2BEV: Language-Enhanced Bird's-Eye-View Maps (ICRA'24) — https://llmbev.github.io/talk2bev/ ; https://github.com/llmbev/talk2bev
- DiffVLA: Vision-Language Guided Diffusion Planning — https://arxiv.org/html/2505.19381
- BEVFormer (ECCV 2022) — https://github.com/fundamentalvision/BEVFormer ; https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690001.pdf

### Visual-token compression (DriveLM-Qwen toolkit context)
- FasterVLM ([CLS] Attention is All You Need for Training-Free Visual Token Pruning) — https://arxiv.org/html/2412.01818v1
- LLaVA-PruMerge — https://github.com/42Shawn/LLaVA-PruMerge
- Awesome-Token-Compress — https://github.com/daixiangzi/Awesome-Token-Compress
- SparseVLM (ICML 2025) — https://icml.cc/virtual/2025/poster/46297
</content>
</invoke>
