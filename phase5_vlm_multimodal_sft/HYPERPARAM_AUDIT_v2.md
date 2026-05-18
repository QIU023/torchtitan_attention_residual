# Stage 2 SFT v2 — hyperparameter audit before launch

**Status**: audit doc, written **before** launching the v2 run (per the `feedback-no-lazy-shortcuts` memory rule).

**Date**: 2026-05-17 23:55

**Trigger**: v1 stage 2 run (launch_stage2.sh as of `c6a3c83`) reached step ~5500/10400 with val ~3.5-3.97 (noisy), train ~0.65-1.4 (spiking). Multiple deviations from LLaVA-1.5 paper hyperparameters caused suboptimal training. v2 is the corrected restart from stage 1 final ckpt (step-8720).

## v1 → v2 deviations (what we are changing)

| Param | v1 value | v2 value | LLaVA-1.5 paper | Reason for v1 deviation | Why v2 matches paper |
|---|---|---|---|---|---|
| `--training.global_batch_size` | 64 | **128** | 128 | "OOM at lbs=16" — never tried grad_accum | grad_accum=2 (`gbs/(lbs*ngpu) = 128/(8*8)`) keeps lbs=8 fitting 5090, matches paper gbs |
| `--training.local_batch_size` | 8 | 8 | 16 | torchtitan-side default | combined with grad_accum=2 → effective gbs=128 |
| `--training.seq_len` | 580 (= 196 + 384) | **1024** (= 196 + 828) | 2048 | dataset default + no AC | AC + 5090 32GB max-out; 1024 covers >95% of mix665k records (p99 ~1500) |
| `text_len` dataset kwarg | 384 (default) | **828** | n/a (paper packs) | dataset default not overridden | match seq_len math: 1024 - 196 vision = 828 text |
| `--training.steps` | 10400 (1 epoch @ gbs=64) | **5200** (1 epoch @ gbs=128) | 5200 | proportional to v1 gbs | paper-aligned epoch budget |
| `--lr_scheduler.warmup_steps` | 312 (0.03 × 10400) | **156** (0.03 × 5200) | 156 | proportional | paper-aligned warmup ratio |
| `--optimizer.lr` | 2e-5 | 2e-5 | 2e-5 | paper-match | unchanged |
| `--optimizer.weight_decay` | (torchtitan default 0.1) | **0.0** | 0.0 | never overridden — pure oversight | paper-match |
| `--lr_scheduler.min_lr_factor` | 0.1 (= 2e-6) | **0.0** | 0.0 | paper schedule decays to 0 | paper-match cosine-to-zero |
| `--lr_scheduler.decay_ratio` | 0.2 (decay last 20%) | 0.2 | (unspecified, but cosine over post-warmup) | torchtitan style | unchanged — paper doesn't specify split, this is reasonable |
| `--activation_checkpoint.mode` | (ignored — Phase 4c skipped AC) | **full** | n/a | AC not wired | required to fit seq=1024 lbs=8 in memory |
| `--mm.val-samples` | 0 → 512 (mid-run added) | 512 | n/a | initially disabled | keep |
| `--mm.val-freq` | 0 → 200 (mid-run added) | 200 | n/a | initially disabled | keep |
| `--mm.val-batches` | 0 → 16 | 16 | n/a | initially disabled | keep |
| `KEEP_K` | 2 | 2 | n/a | torchtitan requires ≥2 | unchanged |
| `SAVE_FREQ` | 500 | 500 | n/a | reasonable | unchanged |

## What we are NOT changing vs paper (or "accept the gap")

| Param | Our value | Paper | Gap | Why accept |
|---|---|---|---|---|
| Model size | 447M Kimi-Linear | 7B Vicuna | -15× | this IS our research project (small model + linear attention) |
| Vocab | 163840 (Llama-3.1) | 32000 (Vicuna) | +5×; +1.6 entropy floor | model choice, fixed |
| Vision encoder | SigLIP-Base 224² | CLIP-L 336² | weaker | model choice, fixed |
| `seq_len` | 1024 | 2048 | -50% | 2048 + lbs=8 + AC marginal on 32GB; can revisit if AC headroom allows |
| Data | mix665K filtered to image-only ~624K | mix665K full 665K | -6% (no OCR-VQA images) | OCR-VQA HF mirror empty; flagged in PR13 |

## Memory math (to validate seq=1024 lbs=8 grad_accum=2 + AC fits)

```
Per-step memory budget (per GPU, 32GB 5090):
  Model (FSDP shard, bf16):         1.4B/8 × 2 = 0.35 GB
  Gradient (sharded, bf16):          0.35 GB
  AdamW state (m + v, fp32):         2 × 0.35 × 2 = 1.4 GB
  Vision tower (frozen, replicated): 100M × 2 = 0.2 GB
  FP8 quant scale buffers:           ~10 GB (Float8Linear bookkeeping)
  Activations with AC=full:
    lbs=8 × seq=1024 × hidden=1024 × 16 layers × 1 saves/layer (AC) × 2 bytes
    = 8 × 1024 × 1024 × 16 × 2 = 256 MB  (! AC only saves layer input, recompute rest)
  FSDP all-gather buffers:            ~3 GB
  KV cache (during forward):          8 × 1024 × 64 × 16 × 2 × 2 = 32 MB
  ─────────────────────────────────────────────
  Estimated TOTAL:                    ~16 GB
  Margin to 32GB:                     ~16 GB ✓ comfortable
```

vs v1 (no AC, seq=580): used 25.4 GB / 81%. With AC + 2× seq → estimated ~20 GB / 63% → margin grows.

If smoke test (50 steps) shows >28 GB used, fall back to seq=580 (still better than v1 due to other fixes).

## v2 launcher changes (summary, to be applied)

```diff
- STEPS="${STEPS:-10400}"
+ STEPS="${STEPS:-5200}"          # 1 epoch @ gbs=128 (paper)

- LOCAL_BS="${LOCAL_BS:-8}"
- GLOBAL_BS="${GLOBAL_BS:-64}"
+ LOCAL_BS="${LOCAL_BS:-8}"        # unchanged; grad_accum=2 yields gbs=128
+ GLOBAL_BS="${GLOBAL_BS:-128}"    # paper

- SEQ_LEN="${SEQ_LEN:-580}"
+ SEQ_LEN="${SEQ_LEN:-1024}"      # 196 vision + 828 text; covers p95 of mix665k

+ TEXT_LEN="${TEXT_LEN:-828}"     # NEW; needs --mm.text-len plumbed through dataset

- WARMUP_STEPS="${WARMUP_STEPS:-312}"
+ WARMUP_STEPS="${WARMUP_STEPS:-156}"   # 0.03 × 5200, paper

# CLI args:
+ --optimizer.weight_decay 0.0        # NEW (paper)
+ --lr_scheduler.min_lr_factor 0.0    # WAS 0.1
+ --activation_checkpoint.mode full   # AC enabled (commit ba498b1)
```

Plus dataset code change: `LlavaInstructSFTDataset.__init__` already accepts `text_len`; needs `--mm.text-len` CLI arg added to `train_mm._parse_mm_args` and passed through `_mm_ds_kwargs`.

## v2 launch plan

1. Apply launcher changes.
2. Smoke test: `--training.steps 50` to verify (a) memory fits, (b) no boot errors, (c) loss decreases.
3. If smoke OK: full v2 run from stage 1 step-8720 ckpt.
4. ETA: 5200 steps × ~3s/step (AC adds ~25% overhead) = ~4.3h + ~2 KDA retries × 5min = **~4.5h to v2 stage 2 done**.
5. v2 val expected: 2.8-3.2 (paper-equivalent for 447M + vocab gap) — significantly lower than v1's 3.5-3.7.

## Risks

1. **seq=1024 still OOMs even with AC**: fall back to seq=896 (text_len=700) or seq=580 (v1 fallback). Keeps gbs=128 and wd=0 fixes.
2. **`--mm.text-len` plumbing breaks something**: revert to dataset default 384, set seq=580. Lose seq fix but keep gbs/wd/min_lr fixes.
3. **Stage 1 step-8720 ckpt path no longer correct after stage 2 retry**: should still be there (we explicitly did not auto-trim).

## Sign-off
Before launching v2: write this audit (✓), apply changes, smoke test, then full launch.
