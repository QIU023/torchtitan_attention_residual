# Phase 5 Report — Multimodal AttnRes-Kimi-VL (dual-arm validation)

**Date**: 2026-04-27 (scaffolding) → 2026-04-28 (handoff doc + smoke results from initial Arm-1 attempt)
**Status**: **Arm 1 scaffolding done, smoke ran 2K steps and stalled (LM-bottleneck) → triggered the Phase-4 100K retrain. Arm 2 is fully spec'd in a self-contained handoff doc and gated on a separate rented box.**
**Hardware**: original 4× RTX 5090 PCIe (Phase-4 retrain runs here) + a separate rented 4-GPU box (≥16 GB / GPU sufficient) for Arm 2.

The deprecated KD/MiniPLM phase moved to `phase5_distillation_deprecated/` (preserved as a negative result). The phase described here is the **dual-arm multimodal** phase.

---

## 1. Goal

Take the **Phase-4 AttnRes-Kimi-436M ckpt** as the LM backbone of a LLaVA-style multimodal model (SigLIP frozen + 2-layer MLP projector + AttnRes-Kimi-436M LM, single-stage full-param fine-tune on LLaVA-Pretrain-558K) and run two arms with the **dual-arm pattern** that Phases 3/4 used at the unimodal stage:

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
       converged multimodal model      cross-modality cache invariance
                                       validation
```

**Arm 2 is the headline new result** — to the project author's literature search, the first **open-source verification that AttnRes' cross-stage caching adapter preserves loss invariants under mixed vision+text sequences with variable-length padding**. Kimi's internal team has presumably solved this; no public writeup exists.

Non-goals: beating LLaVA-1.5 / 1.6 on benchmarks (the 436M LM is 10–20× smaller than Vicuna-7B; not competitive on VQA — point is integration, not score). And a fresh AttnRes-vs-vanilla A/B at the multimodal stage (already done at the LM-only stage in Phase 4).

---

## 2. What shipped

### 2.1 Workspace (`phase5/`, **not** in the torchtitan PR)

| File | Role |
|---|---|
| `README.md` | dual-arm spec, why this isn't a re-A/B, architecture diagram, Arm 1 + Arm 2 setups, two-box parallel schedule, data + tokenizer details, file-by-file role |
| `HANDOFF_arm2_pp_adapter.md` | **38 KB self-contained handoff doc** for a fresh Claude session on a rented box. Covers project context, hardware budget, env setup, the four engineering gaps, an init-strategy menu, a code map, and a **likely-bugs checklist (10.1 → 10.12)** the prior agent's analysis surfaced |
| `data_prep.py` | LLaVA-Pretrain-558K download via `huggingface_hub.snapshot_download` (with `HF_HUB_DISABLE_XET=1` to dodge a known xet-client thread-deadlock); unzips images (~28 GB); sanity-checks bucket extraction |
| `multimodal_dataset.py` | `LlavaPretrainDataset` `IterableDataset`: per-sample sequence layout `[<img> × N_vision] [BOS] [caption tokens] [EOS]` with `IMAGE_TOKEN_ID=32000` (Llama-3.1 reserved special token), `N_VISION_TOKENS=196` (SigLIP-Base @ 224×224 patch16 → 14×14), `IGNORE_INDEX=-100` at image + BOS positions; sharded across (dp_rank, world_size); loops infinitely; `collate_with_pad` for batching |
| `multimodal_model.py` | `Projector` (2-layer MLP `vision_dim → lm_dim → lm_dim` with GELU and trunc-normal init); `multimodal_loss` (vision_tower forward under `no_grad`, projector trainable, single FSDP-root LM call with vision-token scatter happening **inside** `lm.forward`); enforces "exactly N_vision image tokens per row" invariant |
| `train_mm.py` | `MultimodalTrainer` subclass of torchtitan `Trainer`. Overrides `__init__` to load vision_tower frozen + tokenizer + image_processor + projector AdamW; replaces dataloader; overrides `forward_backward_step`. **Currently raises `NotImplementedError` on PP** (line 176-179) — this is the first thing Arm 2 must remove |
| `launch_train.sh` | Arm 1 launcher (FSDP=4, PP=1) |
| `launch_pp_adapter.sh` | Arm 2 launcher — **does NOT exist yet**; HANDOFF doc contains the suggested template (§ 11) |
| `eval_caption.sh` | caption loss + simple VQA accuracy on a small held-out set |
| `tests/__init__.py` | placeholder; no tests yet |

### 2.2 LM backbone reuse

No new production code in `torchtitan/`. Arm 1 reuses:
- `torchtitan/experiments/kimi_linear/attn_res_model.py:KimiLinearAttnResModel` — already accepts `vision_embeds + image_mask` kwargs; line 263-267 does the scatter inside `forward` after `embed_tokens`
- `torchtitan/experiments/kimi_linear/parallelize.py` for FSDP wrapping
- `torchtitan/experiments/attn_res/pipeline_adapter.py:pipeline_llm_with_cache_adapter` — Arm 2 will need to extend its kwargs dispatch to forward `vision_embeds + image_mask` to stage 0 only

### 2.3 The deprecated KD phase

`phase5_distillation_deprecated/` (sibling of `phase5/` at workspace root): MiniPLM-style knowledge distillation experiment, abandoned. Preserved as a negative-result writeup; not part of the current dual-arm phase.

---

## 3. Architecture (shared by both arms)

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

- **Frozen**: SigLIP vision tower (`google/siglip-base-patch16-224`)
- **Trainable**: projector (random init) + LM (Phase 4 ckpt init), all params
- **Loss**: CE on caption text tokens; image positions get `labels = IGNORE_INDEX = -100`

---

## 4. Arm 1 — FSDP2=4, PP=1 (primary, quality)

### 4.1 Setup

Single-stage end-to-end (no separate LLaVA stage-1 projector pretrain). LLaVA-1.5's two-stage recipe assumes a huge already-strong LM (Vicuna-7B) + random-init projector; for our 436M partially-pretrained backbone, single-stage is the right call.

| Knob | Value |
|---|---|
| Hardware | 4× RTX 5090, FSDP2 across 4 ranks for the LM, PP=1 |
| SigLIP | frozen, ~0.3 GB / rank, replicated |
| Projector | ~0.01 GB / rank, replicated, trainable |
| LM | Kimi 436M FSDP-shard, ~0.5 GB weights / rank + activations + AdamW state |
| Per-image | 196 vision tokens + ~30 caption tokens ≈ 226-token seq |
| `local_batch_size` | 8 |
| `seq_len` | 256 (short caption seqs) |
| `TORCHTITAN_ATTNRES_CACHE` | unset (cache adapter off) |
| Throughput target | ~5–10 K image-text pairs/min on 4× 5090 |
| Per-epoch wallclock | ~93 min (558 K / 6 K per min) |
| Total | 3 epochs ≈ 5 h overnight |

### 4.2 Status

- Initial Arm-1 smoke ran 2K steps and **stalled near caption loss 3.8** (caller of the diagnostic).
- Diagnosis: the LM was the bottleneck. After Phase 4 the LM had only seen ~320 M tokens (12 500 × 12 × 2048), far short of chinchilla-optimal ~9 B for a 436 M model. Captions inherit the LM's linguistic ceiling; the multimodal experiment can't validate AttnRes on a robust LM until the LM is robust.
- **This diagnostic is what triggered Phase 4's 100K continuation** (`launch_continuation_100k.sh`) targeting val_loss ≤ 3.0 on C4. Arm 1 is gated on that.
- Phase 4's continuation is currently in progress on the original box; Phase-4's parallel "from-scratch + grad_accum=8 + paper LR" alternative also runs.

### 4.3 Reproduction (once Phase-4 retrain ckpt is ready)

```bash
# Step 1: download data + vision tower (~12 GB total, ~30 min on typical bandwidth)
python phase5/data_prep.py

# Step 2: smoke run (5 steps, single GPU)
STEPS=5 LOCAL_BS=2 bash phase5/launch_train.sh

# Step 3: full overnight (~5 h)
bash phase5/launch_train.sh

# Step 4: eval the resulting ckpt
bash phase5/eval_caption.sh
```

---

## 5. Arm 2 — FSDP2=1, PP=4 V=2 + cache adapter (systems, novel)

### 5.1 Setup

| Knob | Value |
|---|---|
| Hardware | 4× GPUs (≥16 GB / card sufficient for 436M cache-adapter run; 32 GB recommended for SEQ=2048 fallback) |
| `parallelism.pipeline_parallel_degree` | 4 |
| `parallelism.pipeline_parallel_schedule` | Interleaved1F1B (cache-adapter prerequisite) |
| `parallelism.pipeline_parallel_layers_per_stage` | 2 (V=2 virtual stages × lps=2 = 8 virtual stages, every block boundary aligns with stage boundary) |
| `TORCHTITAN_ATTNRES_CACHE` | 1 (adapter ON) |
| `local_batch_size` | 1 (PP fits) |
| `global_batch_size` | 12 (= num_microbatches; ≥ V·PP = 8 satisfies Interleaved1F1B's lookahead requirement) |
| `seq_len` | 258 (196 vision + 60 caption + bos + eos) |
| LR | 1e-5 (full-param fine-tune from ckpt; small) |
| `data_parallel_shard_degree` | 1 (no FSDP sharding, replicated) |

### 5.2 Memory budget per rank (PP=4 V=2, LBS=1, SEQ=258, FSDP=1 replicate)

| component | rank 0 | rank 1–2 | rank 3 |
|---|---|---|---|
| LM 4 layers (bf16) | ~216 MB | ~216 MB | ~216 MB |
| LM AdamW state | ~1.5 GB | ~1.5 GB | ~1.5 GB |
| `embed_tokens` (vocab × hidden) | ~300 MB | — | — |
| `lm_head` + AdamW | — | — | ~2.1 GB |
| `final_attn_res_*` + AdamW | — | — | ~30 MB |
| `vision_tower` (frozen) | ~184 MB | — | — |
| `projector` + AdamW | ~50 MB | — | — |
| PP cache (worst-case 8 blocks × 12 mb) | ~70 MB | ~250–400 MB | ~700 MB |
| Activations (SEQ=258) | ~300 MB | ~300 MB | ~500 MB |
| PyTorch CUDA reserved | ~1–2 GB | ~1–2 GB | ~1–2 GB |
| **rank total** | **~3.7 GB** | **~3.5 GB** | **~6.5 GB** |

Per-block size: `B × T × D × 2 (bf16) = 1 × 258 × 1168 × 2 = 0.6 MB`. Total rank-3 cache: 96 × 0.6 MB = **58 MB** — fits trivially.

### 5.3 Independence from Phase-4 retrain — three init strategies

Arm 2 measures PP-vs-FSDP loss **delta** at matched steps; absolute loss value is irrelevant. Hence three valid init strategies (recommended order **A → B → C**):

- **Strategy A (fresh random init, recommended for first alignment test)**: no checkpoint load. Loss starts ≈ log(vocab) ≈ 11.7, large gradient dynamics → any numerical divergence between PP-adapter and FSDP shows up immediately. Run 1–2 K steps; pass criterion `|Δ| ≤ ` Phase-3 measured FSDP seed-vs-seed band (~0.13 nats).
- **Strategy B (weak Phase-4 ckpt)**: copy `phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500` (~15 GB) over. Loss starts already-near-floor (~3.7 train), gradients small. Tests the adapter under tiny-grad regime. 3–5 K steps.
- **Strategy C (post-retrain ckpt)**: wait for Phase-4 retrain to finish (val ~3.0 target). Most realistic for Arm 1 hand-off but blocks.

### 5.4 The four engineering gaps (concrete Arm-2 work)

Documented in `HANDOFF_arm2_pp_adapter.md` §§ 6–9.

1. **Gap 1 — Remove PP-disable guard** (5 min): delete `train_mm.py:176-179` `NotImplementedError`. Necessary, not sufficient — gaps 2/3/4 surface once removed.
2. **Gap 2 — Vision scatter timing under PP** (medium, 2–3 days): under FSDP=1 PP=1, scatter happens **inside** `lm.forward` so the FSDP root sees one forward call. Under PP=4, `lm.forward` is split into 4 stages; the multimodal scatter MUST happen on stage 0 BEFORE stage 0 sends its hidden state to stage 1. Two fixes: **A (recommended)** make sure `vision_embeds + image_mask` kwargs make it through `pipeline_llm`'s call dispatch to stage 0's submodule (HANDOFF doc has the exact dispatch points to extend); **B** pre-scatter in trainer and pass `inputs_embeds=h` (introduces FSDP/PP plumbing complications).
3. **Gap 3 — Variable-length sequences under PP** (medium, 1–3 days): `PipelineSchedule` sends fixed-shape tensors over P2P; recv buffers are pre-allocated from the FIRST microbatch's shape. Caption tokens vary 5–60 chars → shape mismatch on later mb crash (typical NCCL message: "Tensors must have the same shape"). **Fix**: change `collate_with_pad` to pad to a `GLOBAL_MAX_LEN = 196 + 60 + 2 = 258` deterministic across all microbatches. Drop captions longer than 60 (`max_caption_tokens` arg already does this). Wastes some compute on padding; the alternative (dynamic-shape PP) is a much bigger refactor.
4. **Gap 4 — Cache adapter cross-modality smoke** (key milestone, 3–5 days + debug): set `TORCHTITAN_ATTNRES_CACHE=1`; run PP=4 V=2 + FSDP=4 baseline at matched seed. Validation: `|loss_pp_adapter[step] − loss_fsdp_baseline[step]| ≤ noise_band`. If alignment passes — ship. If alignment fails — root-cause + ship the failure mode as the result.

Total estimated effort: **2–3 weeks** authentic engineering on the rented 4-GPU box.

### 5.5 Likely bugs (debug checklist 10.1–10.12)

The prior agent's analysis identified 12 candidate failure modes. Each is documented with **symptom → detection → fix**. Highlights:

- **10.1 `pipeline_llm` doesn't route multimodal kwargs to stage 0** — vision_embeds=None at stage 0; loss computed on `embed_tokens(IMAGE_TOKEN_ID)` (random-init bucket).
- **10.3 vision_embeds doesn't survive microbatch split** — `(B_global, 196, 1168)` reaches stage 0 with full B_global while input_ids has been sliced to B_micro → shape-mismatch crash. **Most likely failure mode.** Fix: pass `pixel_values` through the input dict PP knows how to split; compute vision_tower + projector INSIDE stage 0's forward.
- **10.4 projector not a stage 0 module → its grads don't accumulate** — projector is "external" to any stage's module tree → forward graph detached from stage 0's backward → grads stay zero. Fix: wrap projector into stage 0's submodule.
- **10.6 attn_res_proj zero-init under multimodal at step 0** — projector starts random → vision-position hidden norms wildly different from text → after 1 grad step attn_res_proj shifts off zero, weights de-uniform, grad spikes. **Probably none-needed** (zero-init guarantees uniform softmax at step 0 regardless of magnitude), but watch for grad_norm spikes in early training.
- **10.7 cache leaks across batches** — `_install_step_drop_patch` from Phase 3 should still trigger; verify under multimodal.
- **10.8 recv buffer shape mismatch on first variable-length mb** — see Gap 3.
- **10.10 gradient flow mismatch at vision-position residuals** — the cache adapter's `_LocalCacheCapture` was designed for text-only; verify it correctly accumulates grads through vision-position values. Detection: compare `projector.fc1.weight.grad.norm()` between FSDP and PP-adapter at matched step.

Treat the checklist as: each item ends up either **fixed** or **confirmed not present and why**. A smoke that runs but can't account for what each bug did or didn't do is **not** a result.

---

## 6. Two-box parallel schedule

| Week | Original box (Phase 4 retrain) | Rented box (Phase 5 Arm 2) |
|---|---|---|
| W0 (now) | Phase 4 retrain in progress | Rent box, env setup, gap 1+2 + 1-microbatch PP=2 smoke |
| W1 | Phase 4 retrain | Gap 3+4 + 2 K-step fresh-init alignment test |
| W2 | Phase 4 retrain finishes | 3-5 K-step weak-ckpt-init alignment test + plot |
| W3 | Arm 1 FSDP overnight (Phase 4 new ckpt loaded) | Arm 2 writeup + commit |
| W4 | Arm 1 finishes, write phase 5 final report | — |

**Total end-to-end ~4 weeks**, vs ~6–8 weeks if serialized.

---

## 7. Findings (interim, as of 2026-04-28)

1. **Multimodal scaffold compiles and runs end-to-end on FSDP=4** at single-stage 558K-image LLaVA setup. CE loss ignoring image + BOS positions is correctly wired.
2. **Initial Arm-1 smoke stalled at caption loss 3.8** because the LM was under-trained. Diagnosis cleanly traced to the LM's val_loss 3.73 → motivated the Phase-4 100K retrain. **Without the Phase-4 retrain Arm 1 cannot be useful.**
3. **Arm 2 is fully spec'd but un-executed.** Total work: 4 engineering gaps + 12-bug debug checklist; 2–3 weeks on a rented box.
4. **The PP-adapter cross-modality alignment is potentially a publishable open-source first** per the project author's literature search — Megatron's open-source multimodal recipes do NOT solve this (they replicate vision tower and run PP only on the LM with full-shape send/recv pad-to-global-max). The cache-adapter twist on top of multimodal is the genuinely new content.

---

## 8. Pointers

- Workspace: [phase5/](../../phase5/) (data_prep, multimodal_dataset, multimodal_model, train_mm, launch_train, eval_caption, README)
- Self-contained Arm-2 handoff: [phase5/HANDOFF_arm2_pp_adapter.md](../../phase5/HANDOFF_arm2_pp_adapter.md)
- LM backbone reused: [torchtitan/experiments/kimi_linear/attn_res_model.py](../../torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py), [torchtitan/experiments/attn_res/pipeline_adapter.py](../../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py)
- Deprecated KD: [phase5_distillation_deprecated/](../../phase5_distillation_deprecated/) — preserved as negative result, not part of the current dual-arm phase
