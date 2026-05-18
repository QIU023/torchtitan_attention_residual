# 08 — Downstream experiment proposal: what to do next after Stage-2 SFT

> **Doc type**: initial-direction / decision doc (per [`README.md`](README.md#doc-type-convention)).
> Written 2026-05-18. Companion to [`06_strategy_workplan.md`](06_strategy_workplan.md) and [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md).
> **Scope**: pick the next concrete experiment after the stage-2 step-3800 ckpt landed; produce a runnable plan, not another survey.

---

## 1. Status recap (as of 2026-05-18)

### What we actually have

| Asset | Location | State |
|---|---|---|
| Stage-2 SFT ckpt (DCP, 17 GB, 8 shards) | `/workspace/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-3800` | trained; not yet benchmark-scored |
| Stage-2 SFT ckpt prior | `…/checkpoint/step-3600` | kept as fallback (KEEP_K=2) |
| Stage-1 alignment ckpt (projector warmed) | `…/runs/stage1_alignment_447m/checkpoint/step-2000` | served as init for stage 2 |
| Backbone | `kimi_linear_447m_aligned_block_attn_res_n4_fp8` config | 447M params, KDA + MLA + MoE + Block AttnRes carrier (depth-axis, K=4) |
| Vision tower | `google/siglip-base-patch16-224` (frozen) | 196 vision tokens / image, 224² only |
| Projector | 2-layer MLP, vision_dim→lm_dim | trained jointly stage 1+2 (~14M params) |
| Tokenizer | `NousResearch/Meta-Llama-3.1-8B` | `IMAGE_TOKEN_ID=32000` reserved |
| Stage-2 data | LLaVA-1.5 `mix665k` minus the ~80K OCR-VQA whose images were never on disk | image–instruction multi-turn, gpt-only loss masking |
| Hyperparams used | gbs=128, lbs=8, seq=1024 (paper 2048 → OOM at lbs=8+AC), 5200 steps, LR=2e-5, cosine 20%, warmup=156 | paper-aligned except seq_len (truncation tail only; p99 ≈ 1500) |

Vision tokens are produced in [`multimodal_dataset.py`](../phase5_vlm_multimodal_sft/multimodal_dataset.py) (`IMAGE_TOKEN_ID`, `N_VISION_TOKENS=196`) and scattered inside `KimiLinearAttnResModel.forward` via `masked_scatter` (see [`torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py`](../torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py) lines ~270–320). That scatter path is what any temporal/multi-image extension must respect.

### What's missing from the SFT ckpt (by design)

- **OCR/TextVQA weak** — SigLIP-Base-224 caps resolution at 224² and we dropped the 80K OCR-VQA slice. Anything that needs reading text in images will under-perform.
- **No multi-image / no video** — dataset emits exactly one image per sample; the `vision_embeds` tensor is shape `[B, 196, D]`. Multi-image/video is a dataloader + scatter change.
- **No driving data ever seen** — Base B has zero in-domain exposure (see [`06_strategy_workplan.md` § Two candidate bases](06_strategy_workplan.md#two-candidate-bases)).
- **Benchmark scores: not run yet.** The eval scripts exist in [`phase5_vlm_multimodal_sft/eval_benchmarks/`](../phase5_vlm_multimodal_sft/eval_benchmarks/) (Priority A: TextVQA, POPE, MM-Vet, LLaVA-Bench-Wild, ScienceQA-IMG, MMBench; Priority B: VQAv2, GQA; MMMU separate) and the runner stub is `eval_stage2_ckpt.sh`, but the *downstream* pipeline (generation → answer extraction → metric scoring) is still pending. This is the task #76-class item that gates any "did stage-2 actually work" claim.

### Time-AttnRes recipe — 30-second summary

Block AttnRes computes `carrier = softmax(Q · norm(V)) @ V` where V is the stack of prior block outputs along **depth**. The mechanism is axis-agnostic; substituting **time** (past K frames) gives Spatio-Temporal AttnRes. [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md) lays out four insertion patterns:

- **Pattern A** — wrap projector output, ~10K new params, no LM changes. **1 week.** *Prototype.*
- **Pattern B** — parallel time-carrier inside each LM block. ~per-layer-D new params, ~400 lines of `attn_res_model.py` rewrite. **4–6 weeks.** *Paper bet.*
- **Pattern C** — frame-cascade (full LM per frame then aggregate). Cheap baseline only.
- **Pattern D** — concat all frame tokens, let KDA handle the long sequence. Ignores AttnRes mechanism (would belong in a "does KDA alone suffice" ablation, not a Time-AttnRes paper).

---

## 2. Tier ranking of downstream experiments

### Inclusion criterion

Every row below is **startable on our actual setup** (8×5090 32 GB, single node, no sim, no new pretrain corpus, base must be a ckpt we currently own). DriveLM-Qwen-track experiments are included because that ckpt is on disk under `/workspace/DriveLM_VLM_Project/`.

### Ranking (sorted by adjusted ROI = research-novelty × feasibility ÷ wall-time)

| Rank | Experiment | Base | Source doc | Novelty | Feasibility | Wall (est) | ROI rationale |
|---:|---|---|---|:--:|:--:|---:|---|
| 1 | **Time-AttnRes Pattern A, single-cam 4-frame, on Kimi-Linear stage-2 ckpt** | B (Kimi) | [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md) §2 | high (axis swap of AttnRes is genuinely new) | high (no LM rewrite; ~10K new params) | ~1 wk eng + ~12 h compute | uniquely exercises **both** our differentiators (KDA + AttnRes) in the regime where they matter; produces ablations directly publishable |
| 2 | **Stage-2 eval pipeline → Priority A benchmark scores** | B (Kimi) | [`phase5_vlm_multimodal_sft/eval_benchmarks/`](../phase5_vlm_multimodal_sft/eval_benchmarks/) + task #76 | low (no new architecture) | high (download scripts done; just need the runner) | ~3 days eng + ~6 h inference | **prerequisite** for any "X improves stage-2 by Y" claim; without this we are flying blind on Pattern A's eval signal too |
| 3 | **DriveLM-Qwen Tier A single-cam temporal SFT (4 frames + DriveLM-QA)** | A (Qwen-3B LoRA) | [`01_video_vlm.md`](01_video_vlm.md) Tier A | low (well-trodden) | high (Qwen has native video + multi-image, compression toolkit wired) | ~1 wk eng + ~6 h compute | the production-pragmatic deliverable; in-domain on day 1; pairs cleanly with the compression-toolkit story |
| 4 | **Tier A WM head — latent next-state predictor on Kimi stage-2** | B (Kimi) | [`04_world_models.md`](04_world_models.md) Tier A | medium (auxiliary head, modest novelty) | high (head is ~1–5M params, MSE+InfoNCE loss) | ~1 wk eng + ~10 h compute | natural follow-on to Pattern A — same encoder, swap "aggregate past for present" for "predict next from past"; mostly shared code |
| 5 | **VLA Tier A — open-loop trajectory SFT on DriveLM-Qwen** | A | [`03_vla_planning.md`](03_vla_planning.md) Tier A | low | high (label swap on existing SFT loop) | ~1 wk | best path to a runnable VLA; needs in-domain Qwen base, not Kimi |
| 6 | **DriveLM-Qwen Tier B — 6-cam × 4-frame + compression sweep (1×/4×/8×/16×)** | A | [`01_video_vlm.md`](01_video_vlm.md) Tier B | medium (compression × surround is the user's own toolkit; first surround eval of it) | medium (needs multi-image extension of compression methods) | ~2 wk eng + ~12 h compute | the right "compression toolkit headline" experiment, but blocked on compression-method multi-image port |
| 7 | **Time-AttnRes Pattern B — parallel time-carrier per LM block** | B (Kimi) | [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md) §3 | very high (real architecture paper) | medium (~400 LOC `attn_res_model.py` rewrite; FSDP shard semantics are touchy) | ~3 wk eng + ~24 h compute | the paper bet, but only worth running **after Pattern A proves the axis-swap principle** (per §7 of the recipe doc) |
| 8 | **VLA Tier B — open-loop GRPO RFT on DriveLM-Qwen** | A + phase-11 GRPO port | [`03_vla_planning.md`](03_vla_planning.md) Tier B | medium (replicates AutoVLA on smaller base) | medium (requires phase-11 GRPO port to Qwen-VL; SGLang adapter for Qwen-VL needed) | ~2–3 wk | best "uses our GRPO infra on driving" story, but blocked on SGLang adapter for Qwen2.5-VL |
| 9 | **SATS-CRP distillation on Kimi stage-2 → smaller student** | B | [`06_strategy_workplan.md` phase 6](06_strategy_workplan.md#phased-work-plan) | medium | low (SATS-CRP is wired for Qwen attention layout; porting to KDA's chunked state is non-trivial) | ~3 wk eng | high potential but architecture port is a real cost |
| 10 | **WM Tier B — AR video-token WM with MAGVIT-v2 + Kimi-Linear LM** | B | [`04_world_models.md`](04_world_models.md) Tier B | very high | low (needs ~10× SFT compute + video tokenizer integration) | months | published direction, but documented out-of-scope for single-node |

### Excluded (out-of-scope for this proposal)

- WM Tier C (SANA-WM / GAIA-2 / DriveDreamer-2 scale). Documented limit in [`04_world_models.md`](04_world_models.md#tier-c--latent-video-diffusion-wm-sana-wm--gaia-2).
- VLA Tier C closed-loop sim. Multi-week sim integration; [`03_vla_planning.md`](03_vla_planning.md#tier-c--closed-loop-grpo-rft-in-sim).
- Anything requiring a >3B base. Compute budget table in [`06_strategy_workplan.md` § Compute estimates](06_strategy_workplan.md#compute-estimates-8-rtx-5090).

### Headline recommendation

> **Do rank-2 (eval pipeline) and rank-1 (Time-AttnRes Pattern A) in parallel — eval first because Pattern A's signal is meaningless without it.** Eval starts immediately (download already done; the gap is the runner). Pattern A engineering starts in parallel; Pattern A's Exp 1 sanity (50 steps) gates on eval-pipeline producing a stage-2 baseline number. Rank-3 (DriveLM-Qwen temporal SFT) is the **second deliverable** in a parallel-track week-4+ slot — it does not compete with rank 1 for the Kimi backbone.

The reason rank-1 beats rank-3 despite rank-3 being more "in-domain": rank-3 is **portable** (any team with a Qwen-VL ckpt could do it), whereas rank-1 is **only doable on a Kimi-Linear + AttnRes stack**. The interview/paper value of "we showed AttnRes generalizes from depth to time" is uniquely ours. Rank-3 is a deliverable, not a research bet.

---

## 3. Time-AttnRes Pattern A — concrete engineering plan

This section translates [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md) §2 + §4 + §6 into a sequenced plan. Code is **not** in this doc — see §3.3 for the file-by-file diff outline.

### 3.1 Files to create

| File | Purpose | LOC est | Notes |
|---|---|---:|---|
| `phase5_vlm_multimodal_sft/time_attn_res.py` | `TimeAttnRes` module from [`07_time_attnres_recipe.md` §2.1](07_time_attnres_recipe.md#21-new-module) | ~80 | exact sketch already in recipe |
| `phase5_vlm_multimodal_sft/multimodal_video_dataset.py` | T-frame-stacked dataset, mirrors `multimodal_dataset.py` `LlavaInstructSFTDataset` but each `<image>` sentinel position expands to `T × N_VISION_TOKENS` positions | ~150 | DriveLM v1.1 has scene/frame indexing; nuScenes CAM_FRONT JPEGs |
| `phase5_vlm_multimodal_sft/launch_stage3_temporal.sh` | Launch script subclass of `launch_stage2.sh` with `--mm.video-mode --mm.num-frames=4 --checkpoint.initial-load-path=<stage2/step-3800>` | ~120 | copy `launch_stage2.sh`; flip the new flags |
| `phase5_vlm_multimodal_sft/tests/test_time_attn_res.py` | Unit tests: alpha non-uniform on toy input; gradient flows; causal mask honored | ~60 | gate Exp-1 sanity |

### 3.2 Files to edit

| File | Change | LOC est |
|---|---|---:|
| `phase5_vlm_multimodal_sft/train_mm.py` | (1) Add `--mm.video-mode`, `--mm.num-frames` CLI flags. (2) In `__init__`: if video mode, construct `TimeAttnRes`, wrap under FSDP2 over batch mesh (mirror projector wrap at lines ~313–325), append to `proj_optim` param groups. (3) In `post_dataloading_process`: detect `pixel_values` rank 5 `[B, T, 3, H, W]`, flatten to `[B*T, 3, H, W]` for vision tower, reshape back to `[B, T, N, D]`, run through `TimeAttnRes`, flatten `[B, T*N, D]` before scatter. | ~80 |
| `phase5_vlm_multimodal_sft/multimodal_dataset.py` | Add video-mode branch on `__init__` returning the video dataset class; keep existing single-image path unchanged | ~20 |
| `torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py` | **NONE** — the scatter logic already handles arbitrary `n_vis = vision_embeds.size(1)` via `masked_scatter` (see lines 270–320: `n_vis_max = vision_embeds.size(1)`). Pattern A respects this contract by flattening `T × N` before passing in. | 0 |

### 3.3 Datasets

For the initial Pattern A run we want **two complementary streams**:

| Dataset | Where | Volume | Per-sample shape | Role |
|---|---|---|---|---|
| nuScenes CAM_FRONT 4-frame clips with DriveLM-QA labels | `/workspace/DriveLM_VLM_Project/` already has DriveLM v1.1 indexing; need to extract `[t−3, t−2, t−1, t]` JPEGs per QA sample | ~30K clips (estimate — DriveLM v1.1 has 377K QA over 696 scenes) | `[T=4, 3, 224, 224]` after SigLIP processor | the **driving signal** |
| Ego4D or Something-Something 4-frame clips with generic captions | optional — only if domain-transfer ablation is wanted | small subset only (Exp 5+) | same | the **non-driving generalization signal** |

**Caveat**: nuScenes raw images are heavy on disk. The `/workspace` mount currently has 84G free of 309G. Pre-extracting only DriveLM-QA-relevant frames (rather than the full 24K-image nuScenes dump) keeps us under budget. Suggest a `precompute_drivelm_4frame.py` one-shot script that emits a JSONL manifest `(frames: [paths], question, answer)` plus an SSD-local JPEG cache.

### 3.4 Memory math — single 5090 (32 GB)

Per-GPU breakdown for a 4-frame video config, sharded with the same parallelism as stage-2 (dp=8, no PP, fp8 LM + bf16 projector/TimeAttnRes):

| Term | Stage 2 (single-img, lbs=8, seq=1024) | Pattern A (4-frame, lbs=4, T=4, seq=1024) | Notes |
|---|---:|---:|---|
| LM params (fp8 → bf16 grads/optim shard) | ~12 GB | ~12 GB | unchanged; FSDP shard |
| Projector | ~0.05 GB | ~0.05 GB | unchanged |
| TimeAttnRes | — | ~0.04 MB | 10K params; negligible |
| Vision tower (frozen, no grad) | ~0.7 GB | ~0.7 GB | unchanged |
| Vision-tower activations (no_grad) | ~0.3 GB | ~1.2 GB | 4× — runs T frames through SigLIP |
| Projector activations | ~0.2 GB | ~0.8 GB | 4× |
| TimeAttnRes activations | — | ~0.1 GB | softmax over K=4 past frames per current frame |
| LM activations (AC on) | ~12 GB | ~14 GB | seq grows from 1024 to ~1024 (text) + 4×196 = 1808; ≈1.5× attn cost; +20–30% activations after AC | (estimate)
| Misc / KDA chunk state / NCCL | ~3 GB | ~3 GB | unchanged |
| **Headroom** | ~4 GB | ~1 GB | tight; gates lbs reduction |

**Decision**: start at **lbs=4, gbs=64** (grad_accum=2). If OOM, drop to lbs=2 + grad_accum=4 (keeps gbs=64; matches stage-2 paper-equivalent token throughput per step but doubles step time). Do **not** drop seq_len below 1024 — DriveLM-QA answers regularly exceed 256 text tokens.

Token cost per step at lbs=4, T=4: 4 × (4×196 + 828) ≈ 6.4K vision-augmented tokens/sample × 8 GPUs × grad_accum=2 = **~102K tokens/step**. Versus stage-2: 8 × 1024 × 8 × 2 = 131K. So Pattern A is **cheaper** per step in raw tokens but ~1.4× more in attention cost due to the longer effective sequence.

### 3.5 Smoke → MVP → full progression

Matches recipe doc §4 Exp 1 → Exp 2 → Exp 3+4:

| Stage | Config | Steps | Wall (est) | Gate (pass/kill criterion) |
|---|---|---:|---:|---|
| **Smoke (Exp 1)** | freeze LM+projector, train TimeAttnRes only; 4 clips × 4 frames | 50 | ~5 min | **Pass**: loss strictly decreases over 50 steps; alpha-weights non-uniform (std > 0.1 across the K=4 past frames, averaged over batch). **Kill**: alpha collapses to one-hot on `t-1` or to uniform — both signal the mechanism isn't learning. |
| **MVP (Exp 2)** | unfreeze projector + LM; full DriveLM 4-frame clip subset; vs mean-pool baseline (same data, no TimeAttnRes) | 1000 × 2 configs | ~3 h | **Pass**: TimeAttnRes ≥ mean-pool by ≥1 pp on held-out DriveLM-QA accuracy. **Kill**: TimeAttnRes underperforms mean-pool by >0.5 pp — the mechanism is not contributing. |
| **K sweep (Exp 3)** | K ∈ {1, 2, 4, 8}; otherwise Exp 2 config | 1000 × 4 | ~6 h | **Pass**: monotone or near-monotone improvement from K=1 → K=4; identify diminishing-returns K. **Kill**: K=1 (= no temporal) wins — temporal context isn't being used. |
| **Freeze ablation (Exp 4)** | freeze-LM vs unfreeze-LM under Exp-2 config | 1000 × 2 | ~3 h | reports how much gain is from TimeAttnRes vs more training |
| **Full (Exp 5 — only if Pattern A passes)** | best K, unfrozen, full DriveLM-QA training subset | 5000 | ~8 h | reports final accuracy; basis for the headline number |

Total compute to a publishable Pattern A result: **~20 h on 8×5090**. Engineering to first smoke: **~5 days** (dataloader is the long pole — vision tower / scatter / LM unchanged).

### 3.6 Validation criteria — quantitative

| Stage | Metric | Threshold | Tooling |
|---|---|---|---|
| Smoke | training loss monotone (Spearman ρ with step > 0.3) | n/a | parse `tb/` events |
| Smoke | alpha entropy in [0.5, 1.3] over log-K | not 0 (one-hot) or log K (uniform) | hook in `TimeAttnRes.forward` |
| MVP | DriveLM-QA accuracy delta vs mean-pool | ≥ +1 pp | new `eval_drivelm_qa.py` (3-day item from §2 rank-2; can share with Priority-A pipeline) |
| K-sweep | best-K accuracy vs K=1 | ≥ +2 pp | same |
| Full | final accuracy vs DriveLM-Qwen-3B LoRA single-frame baseline (the other repo) | informational only (different base; we're 447M vs 3B) | cross-repo comparison table |

---

## 4. Compute budget

Per [`06_strategy_workplan.md` § Compute estimates](06_strategy_workplan.md#compute-estimates-8-rtx-5090) baseline, expanded with Pattern A specifics. All wall-times are 8×5090 single-node.

| Experiment | Config | Per-step | Steps | Wall (est) | Source |
|---|---|---:|---:|---:|---|
| Stage-2 eval (Priority A) | inference only, lbs=8, seq=1024, ~5 benchmarks × 1–5K samples each | ~0.3 s/sample × 8-way | ~15K samples total | ~6 h compute + 1–2 days runner eng | this doc |
| Pattern A Exp 1 (smoke) | lbs=4, T=4, frozen-LM | ~5 s | 50 | 5 min | [`07_time_attnres_recipe.md` §6](07_time_attnres_recipe.md#6-resource-estimate-50908) |
| Pattern A Exp 2 (MVP, 2 configs) | lbs=4, T=4, unfrozen | ~5 s | 2 × 1000 | 3 h | same |
| Pattern A Exp 3 (K sweep) | 4 configs | ~5 s | 4 × 1000 | 6 h | same |
| Pattern A Exp 4 (freeze ablation) | 2 configs | ~5 s | 2 × 1000 | 3 h | same |
| Pattern A Exp 5 (full) | best K, lbs=4, T=4 | ~5 s | 5000 | 8 h | same |
| **Pattern A total** | — | — | — | **~20 h compute + ~1 wk engineering** | — |
| DriveLM-Qwen Tier A (rank 3) | Qwen2.5-VL-3B LoRA r16, single-cam 4-frame | (estimate) ~7 s/step | ~2000 | ~6 h compute + ~1 wk engineering | [`01_video_vlm.md`](01_video_vlm.md) Tier A |
| WM Tier A head (rank 4) | latent next-state, MSE+InfoNCE, lbs=4 | (estimate) ~5 s/step | ~2000 | ~10 h compute + ~1 wk engineering | [`04_world_models.md`](04_world_models.md) Tier A |
| Pattern B (rank 7) | per-layer parallel time-carrier, full 4-frame | (estimate) ~7 s/step (30% slower per §3.3 recipe) | 5000 × 2 (vs A) | ~20 h compute + ~3 wk engineering | [`07_time_attnres_recipe.md` §3 + §6](07_time_attnres_recipe.md#3-pattern-b--code-sketch-deeper) |
| VLA Tier A on Kimi (deprioritized) | trajectory token SFT, single-cam | ~3 s/step | ~2000 | ~2 h compute + ~1 wk engineering | [`03_vla_planning.md`](03_vla_planning.md) Tier A |

Per-step numbers are **estimates** drawn from the recipe doc; we have no measurement for these configs yet. The stage-2 measured step time at lbs=8 seq=1024 was ~4 s (from `tb/` and recent log inspection — to be confirmed when a fresh smoke runs).

---

## 5. Risks and gates

Per the **[[feedback-no-lazy-shortcuts]]** doctrine — risks are listed concretely with a gating mechanism, not as a hand-wave.

### Existing known risks (carried over from phase 3/11)

| Risk | Source | Status | Gate |
|---|---|---|---|
| KDA `chunk_gated_delta_rule_fwd_h` device-side assert on sm_120 | [`SGLANG_FORK_STATE_AND_GAPS.md` P0.2](../phase11_rlhf_grpo_infra/sglang_validation/SGLANG_FORK_STATE_AND_GAPS.md) | training-time workaround: dataset shuffle rotates bad samples | run smoke (Exp 1) on the **exact shuffle seed** that stage-2 used; if assert reappears, fall back to non-chunked path |
| flashinfer_mla NaN on Blackwell fp16 (inference) | same, P0.1 | fp32 MLA fallback active | unchanged — affects sglang-served inference only, not training |
| Pattern B per-frame state breaks FSDP shard semantics | [`07_time_attnres_recipe.md` §5](07_time_attnres_recipe.md#5-risk-register) | known | use Path 1 (batch-axis frames) per recipe §3.2 — NOT Path 2 (streaming state) |
| Pattern A alpha collapse (argmax to `t-1`, or uniform = mean-pool) | recipe §5 | mitigated by Exp-1 gate (alpha entropy check) | kill criterion in §3.5 above |

### New risks specific to this proposal

| # | Risk | Likelihood | Impact | Gate / Mitigation |
|---|---|:--:|---|---|
| N1 | Stage-2 ckpt **never benchmarked** → Pattern A's "+1 pp vs mean-pool" claim has no error bar | medium | high — invalidates the headline | rank-2 eval pipeline is a **hard prerequisite** before Exp 2 starts |
| N2 | DriveLM-QA on **a 447M base never trained on driving** may be near-random; +1 pp signal could be noise | high | high | report **both** absolute accuracy and Δ vs mean-pool; if absolute < 15%, fall back to a generic-caption Ego4D 4-frame ablation where the base has some chance |
| N3 | Disk: nuScenes per-frame pre-extraction risks pushing `/workspace` below the **15 GB panic floor** ([[feedback-disk-panic-protocol]]) | medium | catastrophic (locked-out box) | extract only DriveLM-QA-referenced frame indices (not the full 24K nuScenes images); pre-job disk check; abort if `/workspace` free < 30 GB |
| N4 | Single-image Pattern A integration breaks the existing single-image SFT scatter path (regression on stage-2 dataloader) | medium | medium | (a) keep single-image path as the default branch; video-mode is opt-in via `--mm.video-mode`; (b) the unit test in `tests/test_time_attn_res.py` includes a single-image regression check |
| N5 | TimeAttnRes module gets sharded incorrectly under FSDP2 if wrapped on the wrong mesh | medium | high (silent divergence — same failure mode as the original projector FSDP issue, see `train_mm.py:294-325` comments) | mirror the projector's `batch_mesh` wrapping exactly; assert post-wrap that `time_attn_res.q` is on `batch_mesh` and unsharded across the dp axis as intended |
| N6 | DriveLM v1.1 frame indexing is per-scene, not per-frame timestamp — naive `[t−3:t]` may cross scene boundaries → garbage temporal context | low | medium | dataset emitter checks scene-id of each frame in the clip; reject samples that cross scenes |
| N7 | `masked_scatter` shape contract — current scatter assumes every sample has the same `n_vis = T * N`; mixing video and single-image samples in one batch breaks this | medium | high (silent wrong loss) | enforce video-mode purity per batch; do not interleave with single-image samples |
| N8 | Mean-pool baseline (Exp 2's control) is too weak — beating it doesn't prove temporal attention is the right inductive bias vs e.g. a small Q-Former | medium | medium (weak interview claim) | add a **third arm** to Exp 2: small temporal Q-Former (8 learned queries cross-attending past frames). Only ~30% extra engineering, much stronger ablation. *Recommended addition over the recipe doc's two-arm Exp 2.* |

### Hard kill criteria (stop Pattern A and pivot)

1. **Exp 1 smoke fails** (loss flat or alpha collapsed) AND a 2-day debug doesn't recover → pivot to rank-3 (DriveLM-Qwen Tier A); Pattern B becomes moot.
2. **Exp 2 MVP shows TimeAttnRes < mean-pool by >0.5 pp** → re-run with re-initialized query (orthogonal init per recipe §5); if still fails, Pattern A is dead, but the negative result is itself paper-worthy ("AttnRes axis-swap fails for time"). Document and pivot.
3. **Disk free drops below 20 GB at any point** → halt all training (per [[feedback-disk-panic-protocol]]), clear pre-extracted frames first (recoverable), keep ckpts.

---

## 6. What to NOT do

The following are tempting but explicitly **out of scope** for this proposal window.

1. **Closed-loop sim (CARLA / nuPlan / NAVSIM) for VLA Tier C** — per [`03_vla_planning.md`](03_vla_planning.md#tier-c--closed-loop-grpo-rft-in-sim), sim integration is the real engineering lift; the model side is incremental. We have no sim wrapper today and the multi-week port competes directly with everything in §2. Document as future work; build VLA Tier A (open-loop) only if §2 ranks 1–3 finish ahead of schedule.

2. **Scaling the Kimi-Linear base to 3B / 7B** — per [`06_strategy_workplan.md`](06_strategy_workplan.md#compute-estimates-8-rtx-5090), a competitive 3–7B system needs ~100–1000× current compute. 447M is fine for the research prototype that the Pattern A bet is built around. Picking now to do a new pretrain on a larger backbone would push every other deliverable 2+ months out.

3. **WM Tier B (AR video-token WM) or Tier C (SANA-WM / GAIA-2)** — per [`04_world_models.md`](04_world_models.md#tier-b--autoregressive-video-token-wm), Tier B needs a video tokenizer + non-trivial pretrain; Tier C is in a different architecture family entirely. Tier A WM head (rank 4 in §2) is the only WM direction that fits this window.

4. **Switching the headline base to DriveLM-Qwen** (giving up the Kimi research bet) — the temptation here is "we get in-domain on day 1" but as argued in §2, that loses the unique-to-us angle. The right framing is two parallel deliverables (the proposal does both — rank 1 Kimi-Pattern-A, rank 3 Qwen-Tier-A), not a swap.

---

## 7. Sequence — 1-week and 4-week timelines

### 7.1 One-week plan (Pattern A smoke + eval pipeline only)

| Day | Track A: Pattern A engineering | Track B: Eval pipeline | End-of-day deliverable |
|---|---|---|---|
| 1 | Read recipe §2 + audit `train_mm.py` lines 270–340 (projector wrap, optimizer split); scaffold `time_attn_res.py` from recipe sketch | Audit `eval_benchmarks/` download state; pick 2 fastest benchmarks (POPE + MMBench-en) for first runner pass | `time_attn_res.py` compiles; download check passes |
| 2 | Write `multimodal_video_dataset.py` mirroring `multimodal_dataset.py` `LlavaInstructSFTDataset`; add scene-boundary check (N6) | Implement `eval_pope.py` and `eval_mmbench.py` (generation loop reusing `train_mm.py` model load + `Projector` + scatter) | dataset emits one valid 4-frame clip; eval runner does 1 sample end-to-end |
| 3 | `precompute_drivelm_4frame.py` — extract DriveLM-QA-referenced frames from nuScenes; emit JSONL manifest. Disk check (N3) | Run POPE on stage-2 step-3800; record baseline number | manifest exists; POPE score logged |
| 4 | Patch `train_mm.py` — add `--mm.video-mode`, FSDP2 wrap of `TimeAttnRes` on batch mesh (mirror projector path); `tests/test_time_attn_res.py` (alpha non-uniform; gradient; single-image regression — N4) | Run MMBench-en on step-3800 | unit tests pass; second baseline number |
| 5 | Run Exp 1 (smoke, 50 steps, frozen LM). Check alpha entropy. | Glue script: one-shot `eval_all_priority_a.sh` writes a JSON of all scores | **Pass/fail on smoke; baseline JSON committed** |
| 6 | If smoke passed: launch Exp 2 (1000 steps, MVP, vs mean-pool + Q-Former — N8 third arm) overnight | If time: TextVQA + ScienceQA-IMG runners | Exp 2 launched |
| 7 | Read Exp 2 results; if pass, plan Exp 3 (K sweep) | Aggregate baseline numbers into a stage-2 scorecard | **Decision: Pattern A continues (→ 4-week plan) or pivots** |

### 7.2 Four-week plan (Pattern A full + eval + parallel Qwen-Tier-A)

| Week | Track A: Kimi-Pattern-A | Track B: Stage-2 eval + Qwen-Tier-A | Track C: WM-Tier-A head |
|---|---|---|---|
| **1** | Pattern A engineering + Exp 1 smoke (per 1-week plan) | Eval pipeline → all Priority A benchmarks scored on step-3800 | — |
| **2** | Exp 2 MVP (TimeAttnRes vs mean-pool vs temporal Q-Former); Exp 3 K-sweep | Start Qwen-Tier-A on `DriveLM_VLM_Project`: 4-frame collator, FasterVLM @ 4× compression | — |
| **3** | Exp 4 freeze ablation; if all green, launch Exp 5 (full 5K steps); start drafting Pattern A writeup with ablation table | Qwen-Tier-A first results on DriveLM-QA; baseline vs single-frame | Scaffold WM head: `latent_next_state.py` on top of Pattern A's projector outputs (shares the per-frame stack) |
| **4** | Pattern A Exp 5 results; cross-base comparison table (Kimi-447M Pattern A vs Qwen-3B Tier A); decide whether to commit to Pattern B (3-week follow-on, separate proposal) | Qwen-Tier-A wraps; eval cleaned up into a reusable harness | WM head smoke + MSE+InfoNCE training; gates on Pattern A's frame pipeline being stable |

Key dependency: Track C is **blocked by Track A weeks 1–2** (shared dataloader + frame pipeline). Track B is independent of A.

Daily milestones (if needed) live in the launch scripts' headers; this doc tracks the week-grain plan only.

---

## 8. Open items punted to a follow-on doc

These are real questions but answering them is the next decision cycle, not this one:

- **Pattern B detailed plan** — only worth writing once Pattern A passes Exp 2. The recipe doc §3 covers the architecture sketch; a Pattern B proposal would add the 400-LOC `attn_res_model.py` rewrite plan + FSDP shard semantics audit.
- **VLA Tier B GRPO port to Qwen-VL** — depends on rank-3 + a SGLang-Qwen-VL adapter that we don't have today.
- **SATS-CRP port from Qwen to KDA** — requires deeper read of the SATS-CRP code in `DriveLM_VLM_Project/scripts/region_relation_loss.py`; separate scoping doc.

---

## 9. Cross-references

- Strategy / base choice: [`06_strategy_workplan.md`](06_strategy_workplan.md)
- Time-AttnRes mechanism + code sketches: [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md)
- Video VLM tier A/B/C: [`01_video_vlm.md`](01_video_vlm.md)
- VLA tier A/B/C: [`03_vla_planning.md`](03_vla_planning.md)
- World-model tier A/B/C: [`04_world_models.md`](04_world_models.md)
- End-to-end couplings: [`05_couplings_end_to_end.md`](05_couplings_end_to_end.md)
- SGLang fork state (inference risks): [`../phase11_rlhf_grpo_infra/sglang_validation/SGLANG_FORK_STATE_AND_GAPS.md`](../phase11_rlhf_grpo_infra/sglang_validation/SGLANG_FORK_STATE_AND_GAPS.md)
- Bibliography: [`references.md`](references.md)
