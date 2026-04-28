# Phase 5 — Multimodal AttnRes-Kimi-VL: dual-arm validation

The current Phase 5 (the deprecated KD/MiniPLM phase is at
`phase5_distillation_deprecated/` and is preserved as a negative
result writeup).

This phase takes the **AttnRes-Kimi-436M ckpt from Phase 4** and uses
it as the LM backbone of a LLaVA-style multimodal model. **Two arms,
same data, same model, only parallelism strategy differs** -- the
same dual-arm pattern as Phase 3 (naive vs adapter on 175M Llama3)
and Phase 4 (Problem A FSDP vs Problem B PP-adapter on Kimi 436M LM):

```
                same data: LLaVA-Pretrain-558K
                same model: SigLIP frozen + MLP projector + AttnRes-Kimi-436M
                same ckpt init at step 0
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
       Arm 1 (primary, quality)        Arm 2 (systems, novel)
       FSDP2=4, PP=1                   FSDP2=1, PP=4 V=2 + cache adapter
       3 epochs, ~5h overnight         5k-10k steps smoke
       converged multimodal model      cross-modality cache invariant
                                       validation
```

**Arm 2 is the headline new result** -- the first open-source
verification that AttnRes' cross-stage caching adapter preserves
loss invariants under **mixed vision-text sequences with variable-
length padding**. Kimi's next-generation models will almost certainly
be multimodal + AttnRes; their internal team has surely solved this
already, but there is no public paper or infra writeup. This arm
fills that gap in the open-source ecosystem.

## Goals (in priority order)

1. **(Arm 1)** Demonstrate the AttnRes-co-pretrained 436M can serve
   as a functioning multimodal LM backbone -- caption loss converges,
   simple VQA scoring is meaningful.
2. **(Arm 2)** Validate that the Phase-3 cross-stage cache adapter
   preserves loss equivalence with the FSDP reference under
   **mixed vision+text sequences** -- a regime never previously
   tested. PP=4 V=2 with `TORCHTITAN_ATTNRES_CACHE=1` vs FSDP=4
   baseline at matched steps.
3. Exercise the full Phase 4 → multimodal integration end-to-end on
   the 4× RTX 5090 box: same submodule, same trainer, same FSDP +
   compile path.

**Non-goals:**
* Beating LLaVA-1.5 / 1.6 benchmarks. The 436M LM is ~10-20× smaller
  than the standard LLaVA backbone (Vicuna-7B); we are NOT competitive
  on VQA. The point is integration, not score.
* A/B vs vanilla-Kimi-436M backbone *in this phase*. The Phase 4
  baseline ckpt was deleted in disk cleanup. The relevant A/B
  (AttnRes vs vanilla 436M) was already done at the LM-only stage in
  Phase 4 (val_loss 3.73 vs 3.74). Multimodal is the forward-extension,
  not a re-A/B.

## Architecture (shared by both arms)

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
   AttnRes-Kimi-436M (Phase 4 ckpt)
                                          ALL PARAMETERS TRAINABLE
                                          (full-parameter fine-tune)
                   │
                   ▼
   caption tokens autoregressive (CE loss on text tokens only,
   image-token positions masked via labels = -100)
```

* **Frozen**: SigLIP vision tower
* **Trainable**: projector (random init) + LM (Phase 4 ckpt init)
  all params
* **Loss**: CE on caption text tokens; image positions get
  `labels = -100`

## Arm 1 — FSDP2 long-run (primary deliverable)

Single-stage end-to-end training (no separate projector pretrain).
LLaVA-1.5's two-stage recipe (frozen-LM projector pretrain →
unfreeze-LM SFT) was designed for the case where projector starts
fully random and LM is huge + already-strong (Vicuna-7B). For our
small backbone (436M, partially-pretrained), we collapse to **one
stage**: projector + LM both trainable from step 1.

**Setup:**
* 4× RTX 5090, FSDP2 across 4 ranks for the LM, PP=1 (single-root
  forward)
* SigLIP frozen (~0.3 GB / rank, replicated)
* Projector (~0.01 GB / rank, replicated trainable)
* LM (Kimi 436M FSDP-shard, ~0.5 GB weights / rank + activations
  + AdamW state)
* Per-image: 196 vision tokens + ~30 caption tokens ≈ 226 token seq
* `local_batch_size=8`, `seq_len=256` (short caption seqs)
* Throughput target: ~5-10K image-text pairs/min on 4× 5090
* 558K pairs / 6K per min = ~93 min per epoch
* 3 epochs ≈ 5 hours overnight
* `TORCHTITAN_ATTNRES_CACHE` is **unset** (cache adapter off)

**Gating:** Arm 1 uses the Phase 4 retrain ckpt (in progress as of
2026-04-28). When that ckpt completes, Arm 1 launches.

## Arm 2 — PP=4 V=2 cache-adapter cross-modality smoke (systems deliverable)

**Question:** does the cross-stage cache adapter that Phase 3 / 4
validated on text-only sequences still preserve loss invariants when
the input sequence contains 196 vision-token-aligned positions
(scattered at sentinel positions before the LM forward) followed by
~30 caption tokens?

**Setup:**
* 4× RTX 5090 (rented separately so it runs in parallel with the
  Phase 4 retrain on the original box)
* FSDP2=1, `pipeline_parallel_degree=4`, `pipeline_parallel_schedule=Interleaved1F1B`,
  `pipeline_parallel_layers_per_stage=2` (8 virtual stages, every
  block boundary aligns with a stage boundary)
* `TORCHTITAN_ATTNRES_CACHE=1` (cache adapter on)
* `local_batch_size=1`, `seq_len=256` (PP requires smaller per-rank
  batch; matches Phase 4 Problem B convention)
* `global_batch_size=12` so num_microbatches = 12 ≥ 8 virtual stages

**Independence from Arm 1:** Arm 2 measures the PP-vs-FSDP loss
**delta** at matched steps -- the absolute loss value is irrelevant.
Arm 2 can run on **any well-defined ckpt init**, including the
weak Phase 4 ckpt or even fresh random init. **Arm 2 does not block
on the Phase 4 retrain.**

**Two init points** (run both for confidence):

1. **Fresh random init** (strict alignment test): loss high, gradient
   dynamics large, any numerical divergence between PP and FSDP
   shows up immediately. Run 1-2k steps.
2. **Weak Phase 4 ckpt init** (production-realistic): loss already
   near floor; tests that the adapter behaves cleanly when gradients
   are small. Run 3-5k steps.

Both setups must show PP-adapter loss inside the FSDP seed-vs-seed
noise band over the same horizon (analog to Phase 3's
|Δ_naive→adapter| ≤ |Δ_naive→naive| pattern).

### Concrete gaps to close (vs the current `phase5/` code)

| # | Item | Difficulty | Estimated effort |
|---|---|---|---|
| 1 | Remove `train_mm.py:176-179` `NotImplementedError` PP-disable guard | trivial | hours |
| 2 | Vision scatter must complete on stage 0 BEFORE the stage 0→1 P2P send (vision_embeds + image_mask must be consumed inside stage 0's forward) | medium | 2-3 days |
| 3 | Variable-length sequence handling under `PipelineSchedule` (current setup uses fixed N_vision=196 + caption pad to max_len=60, so per-microbatch seq is uniform; verify no `dynamic shape` assumption in P2P sends) | medium | 1-3 days |
| 4 | Cache adapter smoke: PP=4 V=2 + `TORCHTITAN_ATTNRES_CACHE=1` vs FSDP=4 baseline at matched steps; loss-curve comparison + per-rank cache size accounting | key milestone | 3-5 days + debug |

**Total Arm 2 effort: 2-3 weeks** authentic engineering on rented
4-GPU box.

### Validation expected

* Loss curves: Arm 2 PP+adapter vs Arm 1 FSDP at matched steps
  (Arm 1 may not be done yet -- compare against an FSDP baseline run
  on the rented box at matched ckpt and step count if needed).
* Per-rank cache size at steady state.
* Throughput: tps for PP+adapter vs FSDP single-root forward.
* If alignment **fails**: that itself is publishable -- root-cause
  the divergence (most likely candidates: image_mask slicing under
  microbatch split, vision scatter timing relative to P2P send).

## Two parallel boxes — concrete schedule

| Week | Original box (Phase 4 retrain) | Rented box (Phase 5 Arm 2) |
|---|---|---|
| Now (W0) | Phase 4 retrain in progress | Rent box, env setup, gap 1+2 + 1-microbatch PP=2 smoke |
| W1 | Phase 4 retrain | Gap 3+4 + 2k-step fresh-init alignment test |
| W2 | Phase 4 retrain finishes | 3-5k-step weak-ckpt-init alignment test + plot |
| W3 | Arm 1 FSDP overnight (Phase 4 new ckpt loaded) | Arm 2 writeup + commit |
| W4 | Arm 1 finishes, write phase 5 final report | — |

**Total end-to-end: ~4 weeks**, vs ~6-8 weeks if serialized.

## Data

* **LLaVA-Pretrain-558K** (`liuhaotian/LLaVA-Pretrain` HF dataset):
  558K image-caption pairs filtered from CC3M / LAION. Standard
  small-scale multimodal pretrain corpus. ~10 GB of images +
  metadata.
* If we run out of signal at 558K, can scale to LLaVA-Instruct-665K
  (VQA + multi-turn) in a follow-on. v1 just does captions.

## Tokenizer

Llama-3.1 BPE (vocab 128,256). Same as the AttnRes-Kimi-436M was
trained with (Phase 4 used `./assets/hf/Llama-3.1-8B` tokenizer).
Image-token sentinel: 32,000 (one of Llama's reserved special
tokens) -- replaced at the embedding layer with the projector's
output for that position.

## Files

* `data_prep.py` -- download LLaVA-Pretrain via HF datasets,
  pre-tokenize captions, save image paths + token ids as a
  parquet/jsonl shard.
* `multimodal_dataset.py` -- `IterableDataset` that yields
  (pixel_values, input_ids, labels) batches, handles image loading +
  padding + label masking.
* `multimodal_model.py` -- `MultimodalLM` = SigLIP frozen + projector
  + AttnRes-Kimi LM, with vision-embed scatter inside `lm.forward`
  (single FSDP-root requirement for Arm 1; Arm 2 will adjust the
  scatter timing for PP).
* `train_mm.py` -- `MultimodalTrainer` subclass of torchtitan
  `Trainer`; replaces dataloader and forward path; LM ckpt loaded
  via `--checkpoint.initial_load_path`. **Arm 2 work removes the
  PP-disable `NotImplementedError` here.**
* `launch_train.sh` -- torchrun launcher for Arm 1 (FSDP defaults).
* `launch_pp_adapter.sh` *(Arm 2, to add)* -- PP=4 V=2 +
  `TORCHTITAN_ATTNRES_CACHE=1` launcher.
* `eval_caption.sh` -- caption loss + simple VQA accuracy on a small
  held-out set.

## How to reproduce

### Arm 1 (FSDP, primary)

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

### Arm 2 (PP cache-adapter, systems)

```bash
# Step 1: same data prep
python phase5/data_prep.py

# Step 2: PP=2 1-microbatch smoke (verify vision scatter survives PP)
NGPU=4 PP=2 LOCAL_BS=1 GLOBAL_BS=2 STEPS=5 \
    bash phase5/launch_pp_adapter.sh

# Step 3: PP=4 V=2 fresh-init alignment (1-2k steps)
NGPU=4 PP=4 V=2 LOCAL_BS=1 GLOBAL_BS=12 STEPS=2000 INIT=fresh \
    bash phase5/launch_pp_adapter.sh

# Step 4: PP=4 V=2 weak-ckpt-init alignment (3-5k steps)
NGPU=4 PP=4 V=2 LOCAL_BS=1 GLOBAL_BS=12 STEPS=5000 \
    INIT_CKPT=path/to/phase4/weak_ckpt \
    bash phase5/launch_pp_adapter.sh

# Step 5: compare PP vs FSDP loss curves at matched steps
python phase5/compare_pp_vs_fsdp.py \
    --pp phase5/runs/arm2_pp4_adapter/tb \
    --fsdp phase5/runs/arm2_fsdp_baseline/tb
```

## Notes

* Code lives entirely in `phase5/` workspace dir. Not intended for
  PR merge into torchtitan upstream -- this is a project-specific
  multimodal extension that composes torchtitan's pieces but doesn't
  extend any torchtitan-core protocol.
* The existing `torchtitan/experiments/kimi_linear/multimodal_model.py`
  scaffolding (vision_token_id=-200, KimiVisionProjector) is a
  reference but we re-implement freshly here to keep the workspace
  self-contained.
* Arm 2 may surface real bugs in the PP scheduler / vision scatter
  timing; **failing to align is itself a publishable result** if
  root-caused. Do not optimize for "alignment passes" -- optimize
  for "we know exactly what does or doesn't align and why".
