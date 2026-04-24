# Kimi Linear 436M as a multimodal backbone — where we are and what's realistic

## Short answer

The current 12,500-step Kimi Linear 436M FSDP run produces ~307M
training tokens — only **~0.35% of the paper's 119B token budget**
for the 436M scaling-law size. That ckpt is **too weak to serve as a
real LM backbone** for multimodal pretraining that actually lifts
downstream benchmarks. But it is **completely sufficient as an
architecture-validation bench** for a multimodal scaffolding — end-
to-end forward/backward, projector training dynamics, checkpoint
compatibility, cross-stage adapter behaviour under vision+text token
mixing. The fork's demonstrable contribution isn't "we trained the
best Kimi-Linear-VL", it's "torchtitan finally has a Kimi Linear +
AttnRes + multimodal full-stack scaffolding that actually runs".

Frame the goal accordingly: **architecture demo, not benchmark chase.**

---

## Two different bars for "good enough backbone"

### (a) Real multimodal pretraining (LLaVA-style projector + LLM end-to-end)

Needs a capable LLM — strong instruction-following, real world
knowledge, solid compositional reasoning. Typical bar: validation loss
< 2.0 (perplexity ~7), usually after 100B+ training tokens at the
436M scale.

Our current run: final loss 3.83 ≈ perplexity 46. That's basically a
"unigram model plus a bit of syntax" regime. **Not usable** as a
strong multimodal backbone — the LLM will dominate multimodal failures
rather than amplifying the vision signal.

### (b) Architecture-validation backbone (the realistic target)

The point isn't "does the model answer VQA correctly", it's "does my
multimodal scaffolding train cleanly":
- vision tokens actually flow through the LLM forward
- projector loss decreases during PT-stage training
- ckpt format survives the LLM → multimodal transfer
- PP + AttnRes adapter still behaves on mixed vision-text sequences
- no dtype / device / FSDP / compile surprises

For this bar, **our 12,500-step ckpt is fine.** Loss magnitude doesn't
matter; convergence behaviour and end-to-end plumbing do.

---

## Why from-scratch multimodal pretraining isn't realistic on 4× RTX 5090

- AttnRes paper's 48B-A3B main run: **1.4T tokens**, N=9 AttnRes
  blocks, estimated thousands of H100-days.
- Chinchilla-optimal for 432M params: 432M × 20 = **8.6B tokens** (the
  lower bar for "well-trained-at-compute-cap").
- AttnRes paper's 436M scaling-law row (Table 2): **119B tokens** —
  about 14× Chinchilla-optimal, essentially "compute-over-trained"
  regime.
- 4× RTX 5090 sustained throughput: ~280M tokens per 12h overnight run.
- Extrapolation: reaching even 8.6B tokens (Chinchilla) ≈ **15-20 days
  continuous run**; reaching the paper's 119B ≈ **~150 days**. Neither
  is feasible on this hardware.

So the realistic question isn't "how many more steps to paper quality",
it's "how do we wire up multimodal with the imperfect-but-working LLM
we have".

---

## Realistic path — three phases using the existing ckpt

The scaffolding already exists:
`torchtitan/experiments/kimi_linear/multimodal_model.py` ships
`KimiLinearMultimodalModel` + `KimiVisionProjector`, using
`vision_token_id=-200` as the sentinel for image-token insertion into
the text stream (LLaVA convention).

### Phase 5a — projector-only caption pretraining

- Load step-12500 ckpt into `KimiLinearMultimodalModel`
- Freeze LLM backbone weights
- Freeze vision encoder (SigLIP-SO400M or CLIP-L/14)
- Train only the 2-layer MLP projector
- Dataset: COCO Captions / CC3M subset (~600K image-caption pairs),
  a few epochs fits in 2-4h on 4× RTX 5090
- Expected loss trajectory: caption loss starts ~10 (random projector,
  random vision-to-text mapping), settles around 3-5 once projector
  learns to roughly align vision features with text embeddings

**What this validates:**
- Vision features survive projector + LLM forward without NaNs
- Gradient flows from LM loss → projector → vision encoder (if
  unfrozen) cleanly
- FSDP + frozen backbone + trainable projector combo works in
  torchtitan
- Multimodal ckpt save/load round-trips correctly

### Phase 5b — LLaVA-style instruction SFT (optional, demonstrates full stack)

- Unfreeze LLM (keep vision encoder frozen)
- LLaVA Visual Instruct / ALLaVA subset
- 4× RTX 5090 can push 1-2 epochs overnight
- Result: LLM starts producing vision-conditioned answers, but quality
  will be noticeably weaker than LLaVA-1.5 / Qwen-VL because our
  backbone LLM is undertrained

**What this validates:**
- End-to-end gradient flow through the full stack
- Mixed vision-text sequences don't break anything in KDA / MLA / MoE /
  AttnRes layers
- Effective batch / LR tuning transfers from LM-only to multimodal

### Phase 5c — stress test with PP + adapter (if time permits)

- Same Phase 5b config + `launch_pp4_kimi.sh`-style PP=4 wiring
- Exercises the cross-stage cache adapter with multimodal data
- Most rigorous demonstration the fork handles production-realistic
  deployment

---

## What this path can demo (and what it can't)

**Can demo:**
- ✅ Kimi Linear backbone forwards mixed vision-text tokens cleanly
  (KDA + MLA + MoE all handle the modality mix)
- ✅ AttnRes layers don't blow up on multimodal sequences
- ✅ The cross-stage cache adapter still functions when vision tokens
  are in the mix (Phase 5c)
- ✅ The full torchtitan integration is backbone-agnostic — swap in a
  strong backbone later and Phase 5b becomes publishable

**Can NOT demo:**
- ❌ SOTA multimodal benchmarks — LLM capability ceiling is way too
  low at 307M tokens trained
- ❌ Anything requiring real multi-step reasoning over images — that
  needs the full 100B+ pretraining + proper instruction data

---

## Recommended execution plan (minimum cost)

1. **Lowest-cost validation first:** Phase 5a with the step-12500 ckpt,
   2-4h on 4× RTX 5090 — projector-only caption training. Watching
   caption loss drop from ~10 to ~4-5 proves the architecture is
   alive; beyond that, gains require a stronger backbone.

2. **If Phase 5a passes cleanly:** Phase 5b (LLaVA-SFT) overnight —
   unlocks "end-to-end stack works" claim for the PR/RFC, even though
   vision-QA quality will be weak.

3. **If you want a strong multimodal story:** the LLM backbone has to
   be continued-pretrained to 10B+ tokens on H100/H200 compute,
   reaching loss ~2.5, before any multimodal pretraining makes sense.
   That is a separate project's resource envelope.

---

## TL;DR

The fork's value is architecture validation, not benchmark chase.
Our step-12500 ckpt IS usable for Phase 5a/5b multimodal scaffolding
work on 4× RTX 5090 — just don't confuse "architecture walks
end-to-end" with "beats LLaVA-1.5". Reaching the latter requires
H-class compute for continued LM pretraining first.
