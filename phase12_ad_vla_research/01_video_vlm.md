# 01 — Video VLM

> **Scope**: multi-camera surround input + temporal sliding-window frames as the perception modality entering the LLM. Output is text/QA only (action is in [`03_vla_planning.md`](03_vla_planning.md); future-state prediction is in [`04_world_models.md`](04_world_models.md)).

## Definition

A Video VLM extends a single-image VLM by:
1. Encoding **multiple cameras** (typically 6–8 surround) per timestep.
2. Stacking **temporal frames** (typically 4–8) as additional tokens or via a temporal pooler/cross-attention.
3. (Optional) **Compressing** the resulting big token stream before it enters the LLM.

The output head is still text — captions, QA, scene description. No trajectory.

## Landscape (representative)

| Work | Vision input | Temporal handling | Compression | Notes |
|---|---|---|---|---|
| **AutoVLA** (NeurIPS 2025) | multi-view frames | implicit in token order | none → BIG token stream | trajectory output → see [`03_vla_planning.md`](03_vla_planning.md) |
| **Qwen2.5-VL** | dynamic-res + native video | per-frame ViT + naïve concat | none | first-class sglang support; the natural Base A for video |
| **DriveLM-Qwen2.5-VL LoRA** (DriveLM_VLM_Project) | single CAM_FRONT today; multi-cam raw images present | none yet (TODO) | 4 methods integrated (FasterVLM/PruMerge/PyramidDrop/SATS-CRP) | data + compression toolkit already on Qwen side |
| **Kimi-Linear + SigLIP** (this repo) | single image 224² | none — backbone is text-LM | none | KDA O(N) attention is the natural fit for long video |

## Token-budget arithmetic

For a single timestep, single 224² image: **~196 tokens** post-encoder.
Multi-cam × temporal scales this multiplicatively:

| Modality | Tokens/sample | vs single-image |
|---|---:|---:|
| 1 cam, 1 frame (today's SFT) | ~196 | 1× |
| 1 cam, 8 frames | ~1.5k | 8× |
| 6 cams, 1 frame | ~1.2k | 6× |
| 6 cams, 8 frames, no compression | **~9k** | **~48×** |
| 6 cams × 8 frames + 4× compression | ~2.3k | ~12× |
| 6 cams × 8 frames + 16× compression (−2.4% acc) | ~580 | ~3× |

**Compression is mandatory at full multi-cam × temporal scale**, period. The DriveLM-Qwen project's 4×–16× toolkit is the most direct lever (`scripts/visual_compress.py` in that repo).

## Where Kimi-Linear's KDA matters

KDA is O(N) in sequence length. At ~9k tokens, quadratic attention is ~50× more expensive than linear in the attention term. Critically:
- On quadratic attention (Qwen2.5-VL): 4×–16× compression cuts attention cost ~16×–256× (super-linear in N).
- On linear attention (Kimi KDA): same compression cuts only ~4×–16× (linear).

So **token compression and linear attention are partial substitutes** — they attack the same problem (long visual sequence). Stacking them yields less than the sum. This is why the choice between Base A (Qwen + compression) and Base B (Kimi + linear) is non-trivial; see [`06_strategy_workplan.md`](06_strategy_workplan.md).

## AttnRes generalization: spatio-temporal aggregation

Block AttnRes computes a softmax-weighted aggregation of prior block outputs along the **depth** axis (`K = norm(V); h = softmax(query·K) @ V`). The depth axis is generic — the same mechanism along a **temporal** axis gives:

> **Spatio-Temporal AttnRes**: at frame `t`, the carrier hidden state is a softmax-weighted aggregation of frames `[t−K, …, t−1]` using a learned pseudo-query per layer. Structurally identical to depth-wise AttnRes, just with the loop axis renamed.

This is genuinely novel — it would be a new variant of the AttnRes mechanism, applied where the long-context regime (KDA + AttnRes were designed for it) actually pays off. The ablation surface (depth-only vs temporal-only vs both, K sweep) is interview-rich.

## Three tiered deliverables

### Tier A — single-cam temporal SFT prototype (1 week)
- Take DriveLM-Qwen baseline (CAM_FRONT, single-frame, LoRA r16 on Qwen2.5-VL-3B).
- Add a temporal-batch collator producing 4-frame stacks from nuScenes.
- Optional: wire FasterVLM/PruMerge at 4× compression.
- Eval: DriveLM-QA accuracy vs single-frame baseline.
- **Headline**: temporal extension recovers single-frame quality with 4× more temporal context at the same effective token budget.

### Tier B — full surround-cam × temporal + compression (2 weeks)
- 6 cams × 4 frames input on Qwen2.5-VL-3B base.
- All four compression methods evaluated; ablate 1×/4×/8×/16×.
- Eval: nuScenes-DriveLM open-loop QA.
- **Headline**: 16× compression preserves QA accuracy on full surround-temporal input.

### Tier C — Spatio-Temporal AttnRes on Kimi-Linear (research bet, 4–6 weeks)
- Apply AttnRes along the temporal axis in `KimiLinearAttnResModel`.
- Pretrain on a long-video corpus (e.g. Ego4D, Something-Something) before driving SFT.
- Eval: long-context video QA + driving QA; ablate depth-AttnRes / temporal-AttnRes / both.
- **Headline**: AttnRes generalizes from "across-depth aggregation" to "across-time aggregation" with the same mechanism; KDA gives O(N) headroom for very long sequences.
- **Risk**: requires a stronger Kimi backbone than current 447M and is gated on stage-0 pretrain completion + non-trivial implementation work.

## When to pick this track over BEV ([`02_bev_perception.md`](02_bev_perception.md))

- ✅ Pick video when **appearance detail** matters (signs, lights, text, weather).
- ✅ Pick video when you want to exercise **linear attention** (the Kimi-Linear / KDA story).
- ❌ Skip video when you can get away with metric/spatial-only reasoning (BEV is much cheaper).

## Limitations / open questions

- Multi-cam raw nuScenes is heavy on disk; need a streaming dataloader or pre-extracted token cache.
- The DriveLM-Qwen toolkit's compression methods are currently wired for single-image; multi-image/temporal extension is ~1 week of work.
- Temporal AttnRes has not been empirically validated — the depth-AttnRes paper's claims don't transfer for free.

See [references.md → Video VLM](references.md#video-vlm) for citations.
