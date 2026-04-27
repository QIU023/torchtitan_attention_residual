# Phase 5 — Multimodal full-parameter fine-tune of AttnRes-Kimi-436M

The current Phase 5 (the deprecated KD/MiniPLM phase is at
`phase5_distillation_deprecated/` and is preserved as a negative
result writeup).

This phase takes the **AttnRes-Kimi-436M ckpt from Phase 4** and uses
it as the LM backbone of a LLaVA-style multimodal model, training
end-to-end (full-parameter fine-tune) on standard caption data.

## Goals (in priority order)

1. Demonstrate the AttnRes-co-pretrained 436M can serve as a
   functioning multimodal LM backbone — caption loss converges,
   simple VQA scoring is meaningful.
2. Exercise the full Phase 4 → multimodal integration end-to-end
   on the 4× RTX 5090 box: same submodule, same trainer, same
   FSDP + compile path.
3. Provide a follow-on plumbing target for Phase 3 PP+adapter
   on multimodal sequences (vision tokens + text tokens).

**Non-goals:**
* Beating LLaVA-1.5 / 1.6 benchmarks. The 436M LM is ~10-20× smaller
  than the standard LLaVA backbone (Vicuna-7B); we are NOT
  competitive on VQA. The point is integration, not score.
* A/B vs vanilla-Kimi-436M backbone *in this phase*. The Phase 4
  baseline ckpt was deleted in disk cleanup. The relevant A/B
  (AttnRes vs vanilla 436M) was already done at the LM-only stage
  in Phase 4 (val_loss 3.73 vs 3.74). Multimodal is the
  forward-extension, not a re-A/B.

## Architecture

```
SigLIP-Base-Patch16-224 (frozen, ~92M params)
   image (224×224)
   └─ 196 vision tokens × 768 dim
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ Projector: 2-layer MLP          │   trained from scratch
   │   768 → 1168 → 1168 (Kimi dim)  │   ~3M params
   └─────────────────────────────────┘
                   │
                   ▼
   [196 image-aligned tokens in LM space]
                   │
                   │ inserted at <image> sentinel positions
                   ▼
   AttnRes-Kimi-436M (Phase 4 step-12500 ckpt)
                                          ALL PARAMETERS TRAINABLE
                                          (full-parameter fine-tune)
                   │
                   ▼
   caption tokens autoregressive (CE loss on text tokens only,
   image-token positions masked via labels = -100)
```

## Single-stage end-to-end training (no separate projector pretrain)

LLaVA-1.5's two-stage recipe (frozen-LM projector pretrain →
unfreeze-LM SFT) was designed for the case where projector starts
fully random and LM is huge + already-strong (Vicuna-7B). For
our small backbone (436M, partially-pretrained), we collapse to
**one stage**: projector + LM both trainable from step 1.

* **Frozen**: SigLIP vision tower
* **Trainable**: projector (random init) + LM (Phase 4 step-12500
  init) all params
* **Loss**: CE on caption text tokens; image positions get
  `labels = -100` so they don't contribute to loss

## Data

* **LLaVA-Pretrain-558K** (`liuhaotian/LLaVA-Pretrain` HF dataset):
  558K image-caption pairs filtered from CC3M / LAION. Standard
  small-scale multimodal pretrain corpus. ~10 GB of images +
  metadata.

If we run out of signal at 558K, can scale to LLaVA-Instruct-665K
(VQA + multi-turn) in a follow-on. v1 just does captions.

## Tokenizer

Llama-3.1 BPE (vocab 128,256). Same as the AttnRes-Kimi-436M was
trained with (Phase 4 used `./assets/hf/Llama-3.1-8B` tokenizer).
Image-token sentinel: 32,000 (one of Llama's reserved special
tokens) — replaced at the embedding layer with the projector's
output for that position.

## Hardware budget

* 4× RTX 5090, FSDP2 across 4 ranks for the LM
* SigLIP frozen (~0.3 GB / rank, replicated)
* Projector (~0.01 GB / rank, replicated trainable)
* LM (Kimi 436M FSDP-shard, ~0.5 GB weights / rank + activations
  + AdamW state)
* Per-image: 196 vision tokens + ~30 caption tokens ≈ 226 token seq
* `local_batch_size=8`, `seq_len=256` (short caption seqs)
* Throughput target: ~5-10K image-text pairs/min on 4× 5090
* 558K pairs / 6K per min = ~93 min per epoch
* 3 epochs ≈ 5 hours overnight
* Total: **1 overnight**

## Files (planned)

* `data_prep.py` — download LLaVA-Pretrain via HF datasets, pre-tokenize
  captions, save image paths + token ids as a parquet/jsonl shard.
* `multimodal_dataset.py` — `IterableDataset` that yields
  (pixel_values, input_ids, labels) batches, handles image loading
  + padding + label masking.
* `model.py` — wrapper `MultimodalLM` that holds frozen SigLIP +
  trainable projector + AttnRes-Kimi-436M; forward inserts
  projected vision tokens into LM input embeddings at sentinel
  positions; loss = CE on text tokens.
* `train_mm.py` — `MultimodalTrainer` subclass of torchtitan
  Trainer; replaces dataloader and forward path; LM ckpt loaded
  via `--checkpoint.initial_load_path`.
* `launch_train.sh` — torchrun launcher with all defaults wired.
* `eval_caption.sh` — caption loss + simple VQA accuracy on a small
  held-out set.

## How to reproduce

```bash
# Step 1: download data + vision tower (~12 GB total, ~30 min on
# typical bandwidth)
python phase5/data_prep.py

# Step 2: smoke run (5 steps, single GPU smoke)
STEPS=5 LOCAL_BS=2 bash phase5/launch_train.sh

# Step 3: full overnight (~5h)
bash phase5/launch_train.sh

# Step 4: eval the resulting ckpt
bash phase5/eval_caption.sh
```

## Notes

* Code lives entirely in `phase5/` workspace dir. Not intended for
  PR merge into torchtitan upstream — this is a project-specific
  multimodal extension that composes torchtitan's pieces but
  doesn't extend any torchtitan-core protocol.
* The existing `torchtitan/experiments/kimi_linear/multimodal_model.py`
  scaffolding (vision_token_id=-200, KimiVisionProjector) is a
  reference but we re-implement freshly here to keep the workspace
  self-contained.
