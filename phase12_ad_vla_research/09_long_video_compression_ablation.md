# 09 — Long-video cross-frame compression ablation (overnight)

> **Doc type**: execution plan / decision doc. Written 2026-05-19.
> Companion to [`08_downstream_experiment_proposal.md`](08_downstream_experiment_proposal.md) (which proposed Time-AttnRes Pattern A as the top experiment). This doc supersedes the rank-1 choice for the **immediate next overnight**: we are doing the JD-relevant `cross-frame token compression` study first, because it pays JD skills off the headline ("multimodal architecture + long video frame handling") that Time-AttnRes does not directly target.

## 1. Why this run, why now

After the current nuScenes-planning 4-frame SFT finishes (~09:36Z), we have a paper-comparable baseline on the standard short-horizon driving VLA protocol. The natural next direction is **long video**: 16–32 frames of past context. Three reasons:

1. **JD coverage**: the autonomous-driving JDs we audited (XPeng) emphasise multimodal-architecture craft + long-video handling alongside the standard VLM/VLA skills; the 4-frame baseline alone doesn't exercise the long-video axis.
2. **Public benchmark fit**: nuScenes scenes are ~20 s long. 16 frames at 2 Hz = 8 s past; 32 frames at 2 Hz = ~16 s past, nearly a whole scene. Both are within the LongVideoBench / Video-MME / MLVU long-video standards.
3. **Architecture craft**: the existing DriveLM-Qwen compressors (FasterVLM / PruMerge / PyramidDrop / SATS-CRP) are **single-frame** — they don't model cross-frame redundancy. Long video makes cross-frame compression the load-bearing question.

We are **not** doing Time-AttnRes in this overnight (per user direction). Its module interface is left compatible with the compressor signature so it can drop in later as a fourth method.

## 2. Core question

> At 16 / 32 frames of CAM_FRONT past context, can a **cross-frame token compressor** match the L2 / collision performance of a **no-compression baseline** at a fraction of the LLM-side seq_len?

This is the canonical compression-vs-quality tradeoff, but the axis being compressed is **temporal redundancy** (not spatial like our prior DriveLM-2D work).

## 3. Ablation matrix

| Row | Config | Compressor | LLM seq | Step time | 3-ep wall (8 × 5090) |
|---|---|---|---|---|---|
| **R1** | 4-frame, no compression | — | ~620 | 1.5 s | 2 h |
| **R2** | 16-frame, no compression | — | ~2300 | 11 s | ~7 h |
| **R3** | 16-frame + temporal mean-pool→N | mean-pool every T frames | ~620 | 1.7 s | 2.5 h |
| **R4** | 16-frame + VTM cross-frame merge→N | ToMe-style bipartite merge along (frame × spatial) | ~620 | 2 s | 3 h |
| **R5** | 16-frame + LongVU adaptive prune→N | DINOv2-similarity score, prune high-redundancy frames harder | ~620 | 2 s | 3 h |
| **R6** | 32-frame + temporal mean-pool→N | as R3 | ~620 | 2 s | 3 h |
| **R7** | 32-frame + VTM→N | as R4 | ~620 | 2.5 s | 3.5 h |
| **R8** | 32-frame + LongVU→N | as R5 | ~620 | 2.5 s | 3.5 h |
| R9 (optional) | 32-frame, no compression | — | ~4540 | ~28 s | ~17 h (single-run, user-attended) |

**Compression target**: keep LLM seq_len ≈ 4-frame baseline (~620 tokens). This makes step time roughly constant across compressed configs; the comparison is **quality-given-equal-compute**.

**Compressor interface** (uniform across methods, so adding Time-AttnRes later is zero-cost):

```python
class CrossFrameCompressor(nn.Module):
    """Maps [B, T, N, D] frame token tensor to [B, N', D] compressed tokens."""
    def forward(self, frames: Tensor) -> Tensor: ...
```

Insert in `train_lora.py` between vision-tower-merger output and LM input (mirrors the existing `vision_embeds` scatter path).

## 4. Why 16-frame winner does NOT transfer to 32-frame

User's challenge (correct): a method's compression behaviour is **frame-count dependent**:
- **mean-pool** averages T past frames into 1 representative; at T=4 (8 s) it preserves enough; at T=8 (16 s) it may erase motion onset details — over-smoothing risk.
- **VTM bipartite merge** has a fixed merge ratio; with more frames the merge graph has more candidates and similarity threshold semantics shift.
- **LongVU similarity prune** uses DINOv2 cosine — with more frames there's more "very similar" pairs, so the prune mask gets sparser → may keep more tokens than intended at fixed compression budget.

→ **Each method must be re-run at 32-frame**, not just the 16-frame winner. R6–R8 are all 3 methods at 32-frame.

## 5. Data: no new download

nuScenes scenes are 1000 × ~40 keyframes @ 2 Hz. The existing pipeline (`scripts/planning_dataset.py` in DriveLM_VLM_Project) loads the prev_sample_token chain back N steps. Going from 4 → 16/32 is a config change (`planning_num_past_frames: 16 / 32`); pipeline filters out samples whose history is shorter than N frames (scene-start samples).

Sample-count loss vs current 23,930 train:

| Past frames | Time window | Usable train samples | Loss vs 4-frame |
|---|---|---|---|
| 4 | 2 s | 23,930 | baseline |
| 16 | 8 s | ~16–17 K | −30 % |
| 32 | 16 s | ~8–10 K | −65 % |

Acceptable for 3-epoch SFT. No need to download `sweeps/` (12 Hz intermediate frames) — keyframes at 2 Hz are the standard for paper planning evaluation.

## 6. Disk discipline (HARD)

`keep_latest_k = 2` (user setting). Stage 0/1/2 ckpts (56 GB) untouchable.

| State | Disk used |
|---|---|
| Baseline (stage ckpts only) | 56 GB |
| After each Rn finishes (only final model.safetensors retained ~6 GB; optimizer/scheduler state deleted) | +6 GB per Rn |
| Total after all 8 Rn complete | 56 + 8 × 6 = ~104 GB |
| **During training of Rn** | +36 GB (k=2 × 18 GB transient) |
| **Peak during transitions** | ~150 GB |

Hard rule: **after eval of Rn passes**, delete everything except `model.safetensors` + `eval_results.json` for that run. The optimizer/scheduler state is recoverable from re-training but the model and eval result are the deliverables.

**Disk watchdog** runs in background through the whole overnight: if `/workspace` free drops below 10 GB, **SIGTERM all training** and emit a report listing cleanup candidates ranked by size.

## 7. Sequence (no rush, may exceed 18 h check-in window)

The user is checking back at the 18-h mark, not deadlining. Schedule is end-to-end ~30 h excluding R9:

| Phase | Hours from now | Task |
|---|---|---|
| A | 0 – 1.5 | Wait for current 4-frame SFT (R1) to finish |
| B | 1.5 – 2.0 | `planning_eval.py` on R1 final ckpt → 4-frame baseline L2 / collision |
| C | 2.0 – 5.0 | Parallel subagents: (1) mean-pool compressor (2) VTM compressor (3) LongVU compressor + main: dataset config knobs + train_lora.py compressor mount point |
| D | 5.0 – 6.0 | Smoke 3 compressors @ 16-frame on nuScenes-mini, 5 step each, gate pass/fail |
| E | 6.0 – 14.0 | Sequential 3-ep training: R3 (mean-pool) → R4 (VTM) → R5 (LongVU); inline eval each |
| F | 14.0 – 21.0 | R2 (16-frame no compression baseline, 7 h) — unattended run during user-offline window |
| G | 21.0 – 22.0 | Eval R2 |
| H | 22.0 – 32.0 | R6 → R7 → R8 (32-frame × 3 methods); inline eval each |
| I | 32.0 – 33.0 | `REPORT.md`: 8 (or 9) ablation rows + paper baselines table + training-loss curves + alpha entropy (where applicable) |
| J (optional) | 33.0 – 50.0 | R9 17 h reference run, only if user authorises after seeing R1–R8 |

## 8. Watchdogs

| Watchdog | Trigger | Action |
|---|---|---|
| disk-panic | `/workspace` free < 10 GB | SIGTERM all training; emit cleanup ranking |
| SFT-finish | parent process of current run exits | trigger B (eval) + C (spawn compressor agents) |
| OOM | `[OOM]` line in any training log | tag config as failed; skip to next |
| NaN | `[NaN]` lines > 5 in 100 steps | kill training; tag config as failed |

## 9. Risks specific to this experiment

| # | Risk | Mitigation |
|---|---|---|
| L1 | 16-frame no-compression baseline (R2) OOMs at bs=1 grad_accum=4 (4× longer LLM seq) | Pre-flight VRAM check on 10 batches; if OOM, halve grad_accum to 2 (global bs 16, run 6 epochs to keep tokens-seen constant) |
| L2 | Compressor convergence dynamics differ from no-compression (sliding loss curve shape) | Sliding window already in place; if any compressor's loss plateaus > 1× baseline of R1 by epoch 1, kill and re-init |
| L3 | LongVU needs DINOv2 features — extra forward pass per frame | Use the existing SigLIP features as the similarity signal (avoids extra model); document this deviation |
| L4 | Sample-count loss at 32-frame (−65 %) hurts convergence | Run R6–R8 with grad_accum=4 (same recipe as R1) regardless; if loss is noisier, document and re-run with grad_accum=8 |
| L5 | Disk peak during R8 transitions with all prior finals retained | Aggressive intermediate-ckpt deletion (only final model.safetensors stays) per §6 |

## 10. Cross-references

- Strategy + base choice: [`06_strategy_workplan.md`](06_strategy_workplan.md)
- Time-AttnRes recipe (not used here but interface-compatible): [`07_time_attnres_recipe.md`](07_time_attnres_recipe.md)
- Prior experiment-ranking doc (Time-AttnRes was rank-1; this overnight is the JD-aligned alternative): [`08_downstream_experiment_proposal.md`](08_downstream_experiment_proposal.md)
- Video VLM tier A/B/C surveys: [`01_video_vlm.md`](01_video_vlm.md)
- Cross-repo: training code + planning dataset live in `/workspace/DriveLM_VLM_Project/` on branch `video_vla`; compressors will land in `scripts/compressors/`.
