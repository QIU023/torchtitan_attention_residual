# Phase 5 — Arm 2 (PP=4 V=2 + cache adapter cross-modality) handoff

This document is **self-contained**: another Claude session on a fresh
rented box should be able to pick this up cold and execute. It folds in
the project context, environment setup, the four engineering gaps, and
a detailed enumeration of likely bugs the prior agent expects.

---

## 0. Mission in one paragraph

The original AttnRes paper validated that the cross-stage cache adapter
ships only the **delta blocks** (P-1 blocks per hop steady-state) instead
of the full block stack. This was verified on text-only sequences in
Phase 3 (175M Llama3) and Phase 4 (436M Kimi LM). **Arm 2's job is to
verify the same loss invariance under mixed vision+text sequences** —
the LLaVA-style multimodal regime where 196 image-derived embeddings
are scattered into a sentinel-token-aligned prefix of each row, and the
LM forward then runs through PP=4 V=2 with `TORCHTITAN_ATTNRES_CACHE=1`.

If the adapter aligns inside the FSDP seed-vs-seed noise band, that's a
publishable open-source first. If it fails to align, **the failure mode
+ root cause** is the publishable result. The prior agent's analysis
strongly suggests at least 2-3 non-trivial bugs will show up; this
document enumerates them so you don't waste cycles re-discovering.

---

## 1. Project context (compressed)

**Phase 1-2** (early): AttnRes paper port — the "attention residual"
mechanism = N+1 softmax-pooled block residuals (zero-init pseudo-query
projection, paper §5).

**Phase 3** — `Phase 3 PP cache adapter`: the headline systems result.
Verified on 4-GPU PP=4 V=2 with 175M Llama3 backbone that the adapter
ships P-1 delta blocks per hop (constant per-hop bytes regardless of
depth), with per-rank cache shared across virtual stages. 41/41 CPU
tests green. Loss alignment vs naive PP at 1000 steps was within
bf16/NCCL noise.

**Phase 4** — `Phase 4 Kimi backbone scale-up`: ported Kimi Linear
(KDA + MLA + MoE-style stack) into torchtitan; AttnRes layered on top
as `KimiLinearAttnResModel`. The 436M flavor (`kimi_linear_436m_block_attn_res_n4`)
is what we use as the multimodal LM backbone. The Phase 4 retrain
(from-scratch with paper hparams + grad_accum=8 effective bs=96) is
**still running** on the original 4×5090 box at the time of this
handoff; expected to finish in ~2 days. Arm 2 does NOT block on it
(see § 4 below).

**Phase 5 Arm 1** (other arm, runs on original box): FSDP=4 multimodal
quality run. Single-stage end-to-end fine-tune of AttnRes-Kimi-436M +
frozen SigLIP + trainable MLP projector on LLaVA-Pretrain-558K.

**Phase 5 Arm 2** (THIS doc): PP=4 V=2 + cache adapter on the same
multimodal stack. Runs on a SEPARATE rented box.

---

## 2. Hardware

**Minimum**: 4 GPUs of 16 GB each (RTX 5060 Ti / 4060 Ti class).
At 436M Kimi + PP=4 V=2 + LBS=1 SEQ=258, rank 3 (the heaviest)
sits around 6.5 GB static + ~500 MB activations + ~700 MB PP cache,
all under 8 GB used. **16 GB box is sufficient for Arm 2** (the
correctness/alignment work; absolute tps is irrelevant).

**Recommended**: 4× H100-80GB or 4× RTX 5090 32GB. Cleaner memory
budget, room for SEQ=2048 fallback if vision/text pad-to-max blows
memory at SEQ=256.

**Required**: NVLink between cards for PP P2P bandwidth, OR PCIe is
acceptable but throughput will be ~50-70% of NVLink. PCIe is fine for
correctness validation; the absolute tps doesn't matter for Arm 2's
loss-alignment test.

**NOT required**: multi-node. Single-node 4-GPU is sufficient.

**Memory budget per rank (PP=4 V=2, LBS=1, SEQ=258, FSDP=1 replicate)**:

| component | rank 0 | rank 1-2 | rank 3 |
|---|---|---|---|
| LM 4 layers (bf16) | ~216 MB | ~216 MB | ~216 MB |
| LM AdamW state | ~1.5 GB | ~1.5 GB | ~1.5 GB |
| `embed_tokens` (vocab × hidden) | ~300 MB | — | — |
| `lm_head` + AdamW | — | — | ~2.1 GB |
| `final_attn_res_*` + AdamW | — | — | ~30 MB |
| `vision_tower` (frozen, no opt state) | ~184 MB | — | — |
| `projector` + AdamW | ~50 MB | — | — |
| PP cache (worst-case 8 blocks × 12 mb) | ~70 MB | ~250-400 MB | ~700 MB |
| Activations (SEQ=258, autograd live) | ~300 MB | ~300 MB | ~500 MB |
| PyTorch CUDA reserved overhead | ~1-2 GB | ~1-2 GB | ~1-2 GB |
| **rank total** | **~3.7 GB** | **~3.5 GB** | **~6.5 GB** |

If rank 3 exceeds 12 GB, drop to:

* **Smaller flavor**: `kimi_linear_194m_block_attn_res_n4` (45% the
  param count, fits easily under 4 GB rank 3). Algorithm/cache-adapter
  math is identical; alignment test results carry over to 436M when
  switching to a 32 GB box.
* **Disable compile**: `COMPILE=0` env var saves 1-2 GB transient
  during graph capture.

Never reduce LBS below 1 or GLOBAL_BS below `V × PP = 8` — the latter
breaks Interleaved1F1B's lookahead.

---

## 3. Software environment setup (cold-start, fresh box)

### 3.1 OS + driver baseline

Tested on Linux 6.8 + Ubuntu 22.04 + NVIDIA driver 550+. Newer
should also work. Verify:

```bash
nvidia-smi                # confirm 4× GPUs visible
nvcc --version            # CUDA 12.1+ recommended
```

### 3.2 Python venv

The reference box uses Python 3.12 in a `/venv/main/` venv. On a fresh
box:

```bash
python3.12 -m venv /venv/main
source /venv/main/bin/activate
pip install --upgrade pip
```

### 3.3 Core deps

```bash
# PyTorch + distributed (must be 2.7+ for FSDP2 + improved PP API)
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Kimi Delta Attention (KDA) needs fla-core
pip install fla-core>=0.5.0

# Standard ML stack
pip install transformers accelerate datasets sentencepiece
pip install pillow numpy tensorboard

# torchtitan deps from the submodule
pip install tyro pyarrow
```

After install, sanity-check:

```python
import torch; print(torch.__version__, torch.cuda.is_available(),
                    torch.cuda.device_count())
# expect: 2.7.0 True 4

from fla.modules import FusedRMSNormGated
from fla.layers.utils import ShortConvolution
# both should import cleanly

from transformers import AutoModel, AutoTokenizer
# clean import
```

### 3.4 Repo + submodule

```bash
mkdir -p ~/work && cd ~/work
git clone git@github.com:QIU023/torchtitan_attention_residual.git
cd torchtitan_attention_residual
git submodule update --init --recursive

# the submodule's branch is attention_residual_dev — ensure the local
# checkout matches HEAD recorded in the outer commit
cd torchtitan
git checkout attention_residual_dev
git pull origin attention_residual_dev
cd ..
```

### 3.5 HF cache prep (data + vision tower)

LLaVA-Pretrain images are ~28 GB. SigLIP is small. C4 is needed only
for any LM sanity smokes (~1 GB shard).

```bash
export HF_HOME=$HOME/hf_cache
mkdir -p $HF_HOME

# vision tower
python -c "
from transformers import AutoModel, AutoProcessor
AutoModel.from_pretrained('google/siglip-base-patch16-224')
AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
"

# tokenizer
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('NousResearch/Meta-Llama-3.1-8B')
"

# LLaVA-Pretrain — use snapshot_download to avoid xet client deadlock
# (xet client has a known thread-deadlock bug post-download; use unzip
#  on the images.zip directly)
python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
snapshot_download('liuhaotian/LLaVA-Pretrain',
                  repo_type='dataset',
                  local_dir='$HF_HOME/LLaVA-Pretrain',
                  local_dir_use_symlinks=False)
"

# unzip the images
cd $HF_HOME/LLaVA-Pretrain
unzip images.zip
# bucket dirs (00000/, 00001/, ...) extract directly under
# LLaVA-Pretrain/, NOT into images/. Confirm:
ls 00000/ | head
```

### 3.6 Verify the existing 41 CPU tests are still green

```bash
cd ~/work/torchtitan_attention_residual/torchtitan
python -m pytest torchtitan/experiments/attn_res/tests/ -v
python -m pytest torchtitan/experiments/kimi_linear/tests/ -v
# expect 41 / 41 passing (Phase 3+4 invariant). If anything fails,
# something in the env is broken — do NOT proceed.
```

### 3.7 Verify Phase 3 PP adapter still works on text-only

This is the critical dependency for Arm 2. Before adding multimodal
complications, confirm the adapter alone still aligns in this env:

```bash
cd ~/work/torchtitan_attention_residual
TORCHTITAN_ATTNRES_CACHE=1 NGPU=4 STEPS=50 \
    bash phase3_attnres_pp_integration/launch_pp4_naive_4gpu.sh
# (or the closest existing launcher; check phase3_attnres_pp_integration/runs/ for last
#  successful run config)
# expect loss curve within 0.1 nat of the naive (TORCHTITAN_ATTNRES_CACHE
# unset) baseline at matched 50 steps.
```

If THIS fails, you're not in a valid base state to do Arm 2 — the
adapter itself is broken in this env. Do not proceed; root-cause the
text-only failure first.

---

## 4. Independence from Phase 4 retrain (init-strategy menu)

The Phase 4 retrain on the original box is in progress (~36h ETA at
handoff time). Arm 2 does NOT need to wait. Two valid init strategies:

### Strategy A: fresh random init (recommended for first alignment test)

* Use the 436M `KimiLinearAttnResModel` with no checkpoint load.
* Loss starts ~`log(vocab) ≈ 11.7`, large gradient dynamics in early
  steps. **This is good for alignment testing** — any numerical
  divergence between PP-adapter and FSDP shows up at the largest
  loss-curve scale.
* Run 1-2k steps. Compare PP-adapter loss vs FSDP loss at matched
  steps (matched seeds, matched data shuffle).
* Pass criterion: |Δ| ≤ FSDP seed-vs-seed noise (~0.13 nats per
  Phase 3's measured noise band).

### Strategy B: weak Phase 4 ckpt init (production-realistic)

* Pull the **old** Phase 4 step-12500 ckpt (val ~3.73, NOT the new
  retrain). Stored at `phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500/`
  on the original box. **Copy it over** to the rented box once
  available. (~15 GB.)
* Loss starts already-near-floor (~3.7 train), gradients small.
* Tests that the adapter behaves cleanly when grads are tiny — a
  different regime than Strategy A.
* Run 3-5k steps. Same alignment criterion.

### Strategy C: post-retrain ckpt init (best, but blocks)

* Wait for Phase 4 retrain to finish on the original box (val target
  ~3.0). Copy ckpt over.
* Most realistic for downstream Phase 5 Arm 1 hand-off.
* Same alignment criterion.

**Recommended order: A → B → C** as ckpts become available.

---

## 5. Code map

```
phase5_vlm_multimodal_sft/
├── multimodal_dataset.py      # IterableDataset; (pixel_values, input_ids, labels)
│                                IMAGE_TOKEN_ID=32000, N_VISION_TOKENS=196
│                                Labels: caption tokens only, image+BOS = -100
├── multimodal_model.py        # Projector (2-layer MLP 768→1168→1168)
│                                + multimodal_loss (vision_tower no_grad,
│                                lm called with vision_embeds + image_mask kwargs)
├── train_mm.py                # MultimodalTrainer subclass
│                                * loads vision_tower frozen
│                                * loads tokenizer+image_processor
│                                * appends projector AdamW to optimizer container
│                                * REPLACES dataloader
│                                * OVERRIDES forward_backward_step
│                                * (line 176-179) raises NotImplementedError on PP — REMOVE
├── launch_train.sh            # Arm 1 launcher (FSDP)
├── launch_pp_adapter.sh       # Arm 2 launcher — DOES NOT EXIST YET; create it
├── data_prep.py               # download + extract LLaVA-Pretrain
└── eval_caption.sh            # caption loss eval

torchtitan/torchtitan/experiments/attn_res/
├── pipeline_adapter.py        # CrossStageCacheAdapter, RankLocalCache,
│                                pipeline_llm_with_cache_adapter
├── layout.py                  # BlockLayoutTables — per-rank cache layout
└── attn_res.py                # block_attn_res, AttnResProjection (zero-init pseudo-query)

torchtitan/torchtitan/experiments/kimi_linear/
├── attn_res_model.py          # KimiLinearAttnResModel
│                                * forward(tokens, blocks=None, *, inputs_embeds=None,
│                                          vision_embeds=None, image_mask=None)
│                                * vision scatter happens INSIDE forward at line 263-267
│                                * AttnRes block path at line 282-292
└── parallelize_kimi_linear    # FSDP/compile wrapping for the LM
```

---

## 6. Gap 1 — Remove PP-disable guard (trivial, 5 min)

`phase5_vlm_multimodal_sft/train_mm.py:176-179`:

```python
if self.parallel_dims.pp_enabled:
    raise NotImplementedError(
        "Multimodal trainer does not support PP. Run on FSDP only."
    )
```

Just delete those lines. It's the only thing stopping a PP launch from
even reaching the forward path. **But** the next 3 gaps will surface
once removed; this is necessary, not sufficient.

---

## 7. Gap 2 — Vision scatter timing under PP (medium, 2-3 days)

### The problem

Under FSDP=1 PP=1 (Arm 1's setup), `multimodal_model.py:multimodal_loss`
flows like this:

```
vision_tower (no_grad) → vision_embeds  (B, 196, 768)
projector              → vision_embeds  (B, 196, 1168)
lm.forward(input_ids, vision_embeds=..., image_mask=...)
   └── inside lm.forward:
       h = embed_tokens(input_ids)
       h[image_mask] = vision_embeds.reshape(-1, 1168)  ← scatter
       for layer in layers: h = layer(h)
       h = norm(h)
       logits = lm_head(h)
```

The vision scatter happens **inside** `lm.forward` so the FSDP root sees
exactly one forward call. Good.

Under PP=4 (Arm 2's setup), `lm.forward` is **split into 4 stages** by
torchtitan's `pipeline_module_split`. Stage 0 has `embed_tokens` +
layers 0-3, stage 1 has layers 4-7 (no embed_tokens), etc. The
multimodal scatter MUST happen on stage 0 BEFORE stage 0 sends its
hidden state to stage 1 over P2P.

### Two possible fixes

**Fix A (recommended): Run vision_tower + projector on rank 0 only,
pass scattered hidden state into stage 0's forward.**

The current `multimodal_loss` already calls `lm(input_ids,
vision_embeds=..., image_mask=...)` — but under PP, the call goes
into `pipeline_llm`'s wrapper, which routes to stage 0's submodule.
That submodule's forward receives `(tokens, blocks=None, *,
vision_embeds=..., image_mask=..., inputs_embeds=None, **kwargs)`.

`KimiLinearAttnResModel.forward` (the underlying class) line 262-267
already handles the scatter:

```python
elif self.embed_tokens is not None:
    h = self.embed_tokens(tokens)
    if vision_embeds is not None and image_mask is not None:
        h = h.clone()
        h[image_mask] = vision_embeds.reshape(-1, vision_embeds.size(-1)).to(h.dtype)
```

But: stage 0's submodule (after `pipeline_module_split`) **only has
embed_tokens IF it's the first stage**. Verify that
`KimiLinearAttnResModel.forward` is still the entry point on stage 0
under PP — it should be, but `pipeline_module_split` swaps in `None`
for any module that didn't fall on this stage's layer slice. Stage 0
keeps `embed_tokens`; stages 1-3 get `embed_tokens = None`. The
existing forward handles this via the `elif` chain (line 257-267):
the `inputs_embeds is not None` branch is the PP middle-stage branch;
the `embed_tokens is not None` branch is the PP first-stage branch
(also non-PP).

**So fix A is mostly: just make sure the multimodal kwargs make it
through `pipeline_llm`'s call dispatch.** That's where the work is.
`pipeline_llm` and `pipeline_llm_with_cache_adapter` need to accept
and forward the `vision_embeds` + `image_mask` kwargs into stage 0's
submodule call.

Search for the dispatch:

```bash
grep -n 'forward(' /root/torchtitan_attention_residual/torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py | head -20
```

The cache adapter's per-stage `forward` likely takes `(tokens,
blocks=None)` only. **Extend it to accept and pass `vision_embeds` +
`image_mask` on the first stage; ignore them on middle stages.**

**Fix B: Pre-compute the scattered hidden state in the trainer; pass
`inputs_embeds=h` to PP stage 0.**

* Trainer does: `h = embed_tokens(input_ids)` then scatter, OUTSIDE
  the LM forward.
* But then stage 0's `embed_tokens` is unused — the trainer must use
  `pipeline_module_split`'s "stage 0 submodule" without embed_tokens
  ownership.
* Cleaner separation of concerns but introduces FSDP/PP plumbing
  complications (stage 0's `embed_tokens` becomes dead weight).
* Probably worse than fix A.

### What to verify after Gap 2

```bash
# 1-microbatch PP=2 smoke (no cache adapter; just test scatter survives PP)
NGPU=4 PP=2 LOCAL_BS=1 GLOBAL_BS=2 STEPS=5 \
    bash phase5_vlm_multimodal_sft/launch_pp_adapter.sh
# expect step 1 loss ~10-12 (matches FSDP fresh-init loss); no shape errors.
```

---

## 8. Gap 3 — Variable-length sequences under PP (medium, 1-3 days)

### The problem

`PipelineSchedule` (Interleaved1F1B) sends fixed-shape tensors over P2P.
The shape is whatever `partial_block` is when stage 0's forward
returns: `(batch, seq, hidden)`. If different microbatches have
different `seq` lengths (caption tokens vary 5-60 chars), P2P will
either:

* (best case) error out immediately with shape mismatch
* (worse) silently corrupt with mismatched recv buffer pre-allocated by
  the PP scheduler from the first batch's shape

Neither is acceptable.

### Options

**Option A (use this): pad-to-max in collate.**

`phase5_vlm_multimodal_sft/multimodal_dataset.py:collate_with_pad` already pads to
`max_len = max(b['input_ids'].size(0) for b in batch)`. **But that's
per-microbatch max.** Under PP with N microbatches per batch, the
scheduler runs forward on each microbatch sequentially (interleaved
across stages). Each microbatch has its OWN `max_len`. P2P sends use
the FIRST microbatch's shape as the buffer template.

**Fix**: change `collate_with_pad` to pad to a **GLOBAL max_len**
that's deterministic across all microbatches:

```python
GLOBAL_MAX_LEN = 196 + 60 + 2   # n_vision + max_caption + bos + eos
```

Drop captions longer than 60 (already happens via `max_caption_tokens`
arg). Pad shorter captions up to `GLOBAL_MAX_LEN`. All microbatches
have shape `(B, 258, hidden)`.

**Trade-off**: wastes some compute on padding but the alternative
(dynamic-shape PP) is a much bigger refactor.

**Option B: dynamic-shape PP**.

Out of scope. PyTorch's PP API doesn't support this cleanly; you'd
need to bypass it.

### Subtle gotcha 1: image_mask under microbatch split

PP's scheduler splits the global batch into N microbatches. Each
microbatch goes through stage 0 → stage 1 → ... independently.
`image_mask` is shape `(B_micro, T)`. **Verify** that:

* Each microbatch's `image_mask` correctly identifies its 196 image
  positions
* `B_micro = global_batch_size / num_microbatches` is the right
  micro-batch dim
* The mask is sent ALONGSIDE the input_ids to stage 0 (not just
  computed there from input_ids)

If image_mask is recomputed inside stage 0 from input_ids (current
code path: `image_mask = (input_ids == IMAGE_TOKEN_ID)`), then it's
deterministic per-microbatch and doesn't need to cross P2P. Good.

### Subtle gotcha 2: pad-token attention mask

Padded positions need to be masked out of the LM's attention. Kimi
Linear's KDA + MLA architecture: does it use a built-in attention
mask, or assume packed sequences? **Check `kimi_linear/model.py`** for
the attention path. If it doesn't honor a pad mask, you'll need to
either:

* Set padded labels to `-100` so loss ignores them (already done in
  `collate_with_pad`)
* OR explicitly pass an attention mask through stage forward

Loss is the dominant concern (CE only on non-`-100` positions); the
attention itself attending to pad tokens shouldn't break correctness
as long as those positions don't contribute to grads of real tokens.
But it WILL waste compute. Profile and decide.

---

## 9. Gap 4 — Cache adapter cross-modality smoke (key milestone, 3-5 days)

### Setup

Set `TORCHTITAN_ATTNRES_CACHE=1`. Launch:

```bash
NGPU=4 PP=4 V=2 LOCAL_BS=1 GLOBAL_BS=12 STEPS=2000 INIT=fresh \
    bash phase5_vlm_multimodal_sft/launch_pp_adapter.sh
```

Notes:
* `V=2` = `pipeline_parallel_layers_per_stage=2` (8 virtual stages on
  16-layer 436M; every block boundary aligns with a stage boundary
  since `layers_per_block=4` and `V*PP=8`, see Phase 3 layout calculus)
* `GLOBAL_BS=12` so `num_microbatches=12`, ≥ V*PP=8, satisfies
  Interleaved1F1B's lookahead requirement

Run **simultaneously** the FSDP=4 baseline at matched seed:

```bash
NGPU=4 PP=1 LOCAL_BS=3 GLOBAL_BS=12 STEPS=2000 INIT=fresh \
    bash phase5_vlm_multimodal_sft/launch_train.sh   # Arm 1 launcher with same data
```

### Validation criterion

At matched steps (1, 10, 100, 500, 1000, 1500, 2000):

```
|loss_pp_adapter[step] − loss_fsdp_baseline[step]|  ≤  noise_band
```

Where `noise_band` ≈ 0.13 nats (Phase 3 measured FSDP seed-vs-seed
spread on Llama3 175M; remeasure for Kimi 436M if conservative).

If alignment passes: ship it as the headline result.

If alignment fails: **root-cause the divergence** and write up the
failure mode. That itself is publishable. Possible root causes are
listed in § 10.

### Per-rank cache size accounting

At the end of one batch's forward sweep, log per-rank cache size:

```python
# inside RankLocalCache or via a debug hook in pipeline_adapter
total_blocks = sum(len(blocks) for blocks in cache._blocks.values())
print(f"rank={rank} mb-keyed cache slots: {total_blocks}")
```

Expected (PP=4 V=2, num_blocks=8, M=12 microbatches):
* rank 0: 5 blocks/mb × 12 mb = 60 block-slots steady-state
* rank 1: 6 × 12 = 72
* rank 2: 7 × 12 = 84
* rank 3: 8 × 12 = 96
* See Phase 3 handoff `handoff_status_20260421.md` for the math

Per-block size: `B × T × D × 2 (bf16) = 1 × 258 × 1168 × 2 = 0.6 MB`.
Total rank-3 cache: 96 × 0.6 MB = **58 MB**. Fits trivially. (Phase 3
175M was 25-50 MB; Kimi 436M cache is ~2x larger per block but still
fits.)

---

## 10. Likely bugs (the real handoff content)

The prior agent's deep-dive analysis surfaced several places where the
naïve port from Arm 1's FSDP-only multimodal to Arm 2's PP+adapter
multimodal will likely fail. Use this as a debug checklist when
alignment doesn't pass on first attempt.

### Bug 10.1 — `pipeline_llm` doesn't route multimodal kwargs to stage 0

**Symptom**: stage 0's forward gets called with `vision_embeds=None,
image_mask=None`. Vision scatter never runs. Loss computed on
`embed_tokens(IMAGE_TOKEN_ID)` (which is whatever Llama tokenizer's
embedding for token 32000 happens to be — random-init or wherever
Phase 4 left it after seeing 0 actual images during pretraining).

**Detection**: log `vision_embeds is None` at top of
`KimiLinearAttnResModel.forward` on stage 0. If it's None, you have
this bug.

**Fix**: extend `pipeline_llm` (and the cache adapter's `forward`
wrapper) to accept and forward `vision_embeds` and `image_mask` to
stage 0 only. Stages 1-3 should not see these kwargs.

### Bug 10.2 — image_mask and input_ids out of sync after microbatch split

**Symptom**: PP scheduler splits the global batch into N microbatches.
If `image_mask` is computed in the trainer (from full input_ids) and
passed alongside, the slicing might end up using `B_global // N` for
input_ids but full-batch image_mask, or vice versa.

**Detection**: log `input_ids.shape, image_mask.shape, vision_embeds.shape`
at top of stage 0 forward. They must be consistent: B_micro for both
mask and ids; vision_embeds is `(B_micro, 196, 1168)`.

**Fix**: Compute image_mask **inside** stage 0's forward from
input_ids: `image_mask = (input_ids == IMAGE_TOKEN_ID)`. Don't pass
image_mask through the kwargs path; this is what `multimodal_model.py`
does today and it's safe.

### Bug 10.3 — vision_embeds doesn't survive microbatch split

**Symptom**: vision_embeds is `(B_global, 196, 1168)`. PP splits the
batch dim, so each stage 0 forward call receives `(B_micro, 196, 1168)`.
But the SPLIT only happens for the model inputs the PP scheduler knows
about (`input_ids` typically). Custom kwargs like `vision_embeds` may
NOT get split — they'd reach stage 0 with the full B_global, while
input_ids has been sliced to B_micro. Crash on `image_mask` shape
mismatch in scatter.

**Detection**: same as 10.2 — shape check at stage 0 forward entry.
This is the most likely failure mode.

**Fix**: **Pass `pixel_values` through the input dict** that PP knows
how to split. Compute vision_tower + projector INSIDE stage 0's
forward. Specifically:

```python
# new flow
input_dict = {"input": input_ids, "pixel_values": pixel_values}
# pp scheduler splits "input" AND "pixel_values" along batch dim
# stage 0 forward:
def forward(input, pixel_values, ...):
    with torch.no_grad():
        vision_features = vision_tower(pixel_values).last_hidden_state
    vision_embeds = projector(vision_features)
    h = embed_tokens(input)
    image_mask = (input == IMAGE_TOKEN_ID)
    h[image_mask] = vision_embeds.reshape(-1, vision_embeds.size(-1)).to(h.dtype)
    # then run stage 0's layer slice...
```

Where is vision_tower? It must be **attached to stage 0's submodule**
(not the trainer). Add it as a child of the stage 0 submodule via
the parallelize_fn or as an external hook. **This is the key
non-trivial piece.**

### Bug 10.4 — projector not a stage 0 module → its grads don't accumulate

**Symptom**: projector parameters' grads are computed but never
applied. Loss curve looks similar to FSDP at first then drifts.

**Detection**: at `optimizer.step()`, log `projector.fc1.weight.grad.norm()`.
If it's 0 or None, projector grads aren't reaching the optimizer.

**Fix**: ensure the projector is wrapped as part of stage 0's submodule
so its forward+backward sit inside stage 0's autograd subgraph. Then
the projector AdamW (separate optimizer in `train_mm.py:128-145`) will
see its grads as expected. If projector is "external" (not in any
stage's module tree), its forward graph is detached from stage 0's
backward and grads stay zero.

**Alternative**: include projector params in stage 0's optimizer (the
LM's optimizer, not the separate projector AdamW). Simpler.

### Bug 10.5 — vision tower called once per microbatch instead of once per batch

**Symptom**: throughput much lower than expected. Each PP microbatch
re-runs the entire SigLIP forward.

**Detection**: tps significantly below FSDP=1 baseline. Or profile.

**Fix**: cache vision_features at the global-batch level. In stage 0's
forward, if we're processing microbatch i of N, look up cached
vision_features for this batch and slice to the i-th microbatch. Keyed
by something stable across the PP forward sweep (the chunk_id from PP
scheduler — see `pipeline_adapter.py:46-54` for how the cache adapter
keys per-microbatch state).

**Acceptable simpler fix**: just let it re-run; vision_tower is small
(~92M frozen params, no_grad) compared to LM forward. Profile first;
optimize only if it's a real bottleneck.

### Bug 10.6 — attn_res_proj zero-init breaks under multimodal at step 0

**Symptom**: at step 0 (random init or fresh ckpt load), the
zero-init `attn_res_proj` makes block residual weights uniform.
Vision-position hidden states (large magnitude from random projector)
get equal weight as text-position hidden states (smaller magnitude
from embed_tokens). Could cause gradient explosion via softmax
saturation.

**Detection**: at step 1, log `attn_res_proj(block_outputs).abs().max()`
on stage 3 (last stage with all blocks visible). If softmax inputs
are wildly different magnitudes between vision and text positions,
this bug is live.

**Fix**: probably none needed — paper's zero-init guarantees uniform
softmax at step 0, which means uniform weight regardless of magnitude.
The result of softmax(0,0,...,0) is `1/(N+1)` for all entries. Vision
vs text position magnitudes affect the WEIGHTED SUM, not the weights.
Uniform-weight aggregation of hidden states with different magnitudes
gives a hidden state of intermediate magnitude — fine.

But: the projector starts random, so vision positions have hidden
norms wildly different from text. After 1 grad step, attn_res_proj
shifts off zero, weights de-uniform, and now vision/text positions
get DIFFERENT weights based on… something. Watch for grad_norm spikes
in early training.

**If this bug bites**: warm up the projector first (1000-step
projector-only pretrain with LM frozen, similar to LLaVA stage 1).
But that complicates Arm 2's "single ckpt init at step 0" claim;
prefer to just verify in the smoke that it doesn't bite.

### Bug 10.7 — cache adapter's mb-keyed cache leaks memory across batches

**Symptom**: per-rank GPU memory grows unboundedly over training
steps. Eventually OOM.

**Detection**: log `len(rank_cache._blocks)` after each batch.
Should be 0 at the start of each batch (eviction triggered by
`_install_step_drop_patch` at the end of each `pp_schedule.step` call).

**Fix**: Phase 3 already handles this via `_install_step_drop_patch`
in `pipeline_adapter.py`. Verify the patch is still installed under
multimodal — `pipeline_llm_with_cache_adapter` should call it. If it
doesn't, reinstall.

### Bug 10.8 — recv buffer shape mismatch on first variable-length microbatch

**Symptom**: at step 1, P2P recv on stage 1 fails with shape mismatch
(typical message: "Tensors must have the same shape"). The PP
scheduler pre-allocates recv buffers based on the FIRST microbatch's
shape; a later microbatch with longer/shorter padded length crashes.

**Detection**: this is loud — first or second microbatch crashes
with NCCL/P2P error.

**Fix**: see § 8 (Gap 3) — pad-to-global-max instead of pad-to-batch-max.

### Bug 10.9 — image_mask ALL-False on middle PP stages

**Symptom**: stage 0 sends `partial_block` (already with vision
scattered) to stage 1. Stage 1 doesn't have `embed_tokens`, doesn't
have `image_mask`, doesn't need it (vision is already in `h`). But if
the stage 1 forward path tries to construct image_mask from
`input_ids`, it'll fail because `input_ids` is None on stage 1.

**Detection**: stage 1 forward logs `image_mask is None and tokens is None`.

**Fix**: middle stages should not touch image_mask. The kwargs flow
should be: `vision_embeds + image_mask` → stage 0 only, scatter happens
inside stage 0's forward, downstream stages just receive scattered
`partial_block` and don't need to know vision was ever there.

### Bug 10.10 — gradient flow mismatch at vision-position residuals

**Symptom**: under cache adapter, the LM's last stage runs
`block_attn_res(block_list, partial_block, final_attn_res_proj,
final_attn_res_norm)`. The block_list contains hidden states from
earlier blocks, including their vision-position values. Gradient
flowing backwards from final_attn_res_proj's softmax-pooled output
must reach BOTH the vision-position scatter point AND the projector.

Under FSDP=1 single-root forward (Arm 1), this is automatic — autograd
tracks the full graph.

Under PP+adapter, the `block_list` is reconstructed via the cache
adapter (some blocks are `recv_delta` slices from upstream stages,
some are local detached cache entries with `_LocalCacheCapture`
wrapping). **The captured-grad mechanism in `_LocalCacheCapture` was
designed for text-only; need to verify it correctly accumulates grads
through vision-position values.**

**Detection**: grad_norm on `projector.fc1.weight` should be similar
between FSDP and PP-adapter at matched step. If it's significantly
smaller in PP-adapter, the vision-residual grad path is broken.

**Fix**: depends on root cause. Most likely: the `_LocalCacheCapture`
just wraps the entire hidden state slice including vision positions,
and grads flow through cleanly. But verify.

### Bug 10.11 — recv_delta slicing assumes contiguous block boundary

**Symptom**: at higher PP ranks, the cache adapter receives a delta
tensor of shape `(num_new_blocks, B, T, D)`. Each block is a
**slab of T tokens**. With variable-length sequences (post-pad), T is
fixed, but the **content** at vision positions vs text positions
differs. The delta unstack should work cleanly (it's just a stride-1
copy), but verify no assumption about "all positions are text-like".

**Detection**: `unstack_blocks` on a recv_delta returns tensors of
unexpected shape, or autograd graph shows a strange topology around
vision positions.

**Fix**: probably none needed; `unstack_blocks` is a pure tensor op.
But run a single-step debug with `torch.autograd.set_detect_anomaly(True)`
to be sure no NaN sneaks in.

### Bug 10.12 — interaction with FSDP1 replicate fallback

**Symptom**: under PP=4 V=2 + FSDP2=1, all params are replicated per
rank (no FSDP sharding). But the parallelize_kimi_linear function
might still wrap the LM in FSDP-like containers. Verify that
`parallelism.data_parallel_shard_degree=1` actually disables FSDP.

**Detection**: `nvidia-smi` memory should be 4× smaller per rank
under FSDP2 sharding vs replicate. Under PP=4 V=2 + FSDP=1, memory
per rank should equal full-model size, not sharded.

**Fix**: configure correctly in the launcher. Phase 3's launcher had
this right; copy that config.

---

## 11. Suggested launcher template (`launch_pp_adapter.sh`)

```bash
#!/usr/bin/env bash
# Phase 5 Arm 2: PP=4 V=2 + cache adapter cross-modality smoke.
#
# Tests whether the Phase-3-validated cache adapter preserves loss
# invariance under mixed vision+text sequences.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

NGPU="${NGPU:-4}"
PP="${PP:-4}"
V="${V:-2}"
STEPS="${STEPS:-2000}"
LOCAL_BS="${LOCAL_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-12}"   # >= V*PP=8 microbatches for Interleaved1F1B
SEQ_LEN="${SEQ_LEN:-258}"      # 196 vision + 60 caption + bos + eos
LR="${LR:-1e-5}"               # full-param fine-tune from ckpt; small LR

INIT="${INIT:-fresh}"          # fresh | weak_ckpt
INIT_CKPT="${INIT_CKPT:-}"

if [[ "$INIT" == "weak_ckpt" && -z "$INIT_CKPT" ]]; then
    echo "ERROR: INIT=weak_ckpt requires INIT_CKPT path" >&2
    exit 1
fi

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/arm2_pp${PP}_v${V}_${INIT}}"
mkdir -p "$OUT_DIR"

CKPT_ARGS=""
if [[ "$INIT" == "weak_ckpt" ]]; then
    CKPT_ARGS="--checkpoint.enable \
               --checkpoint.initial_load_path $INIT_CKPT \
               --checkpoint.initial_load_model_only"
fi

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}:${PYTHONPATH:-}" \
TORCHTITAN_ATTNRES_CACHE=1 \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="$NGPU" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_vlm_multimodal_sft.train_mm \
    --mm.json /root/hf_cache/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --mm.images /root/hf_cache/LLaVA-Pretrain \
    --mm.vision-model google/siglip-base-patch16-224 \
    --mm.tokenizer NousResearch/Meta-Llama-3.1-8B \
    --mm.cache-dir /root/hf_cache \
    --module kimi_linear --config kimi_linear_436m_block_attn_res_n4 \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "$STEPS" \
    --training.local_batch_size "$LOCAL_BS" \
    --training.global_batch_size "$GLOBAL_BS" \
    --training.seq_len "$SEQ_LEN" \
    --optimizer.lr "$LR" \
    --lr_scheduler.warmup_steps 100 \
    --lr_scheduler.total_steps "$STEPS" \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree "$PP" \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage "$V" \
    --parallelism.data_parallel_shard_degree 1 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    $CKPT_ARGS \
    --metrics.save_tb_folder tb \
    --dump_folder "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/train.log"
```

Also need a matching FSDP baseline launcher with the **exact same**
data shuffle seed, hparams, and ckpt init, for the alignment
comparison. Easiest is to set `--training.seed=N` consistently and use
`launch_train.sh` with `INIT_CKPT` matching.

---

## 12. Order of operations

1. **Day 1**: § 3 env setup + § 3.6 + § 3.7 sanity smokes.
2. **Day 2**: § 6 Gap 1 (5 min) + read § 7 + § 10 thoroughly.
3. **Days 3-5**: § 7 Gap 2 (vision scatter under PP). Bug 10.1, 10.3,
   10.4 most likely to bite first. Land 1-microbatch PP=2 smoke
   passing.
4. **Days 6-8**: § 8 Gap 3 (var-len padding). Bug 10.8 will surface
   immediately when going past 1 microbatch.
5. **Days 9-13**: § 9 Gap 4 (cache adapter on). Bugs 10.6, 10.7,
   10.10, 10.11 are the new content. Run Strategy A (fresh init)
   alignment test.
6. **Days 14-15**: Run Strategy B (weak ckpt init) alignment test.
7. **Days 16-21**: write up results. If alignment passes — ship as
   the headline. If alignment fails — root-cause + ship the failure
   mode as the result.

---

## 13. References

* **AttnRes paper**: arXiv 2509.13863 (Block AttnRes scaling-law
  table is Table 2)
* **Kimi Linear paper**: MoonshotAI technical report 2024 (Kimi
  KDA + MLA architecture; 48B-A3B reference config)
* **Phase 3 handoff**: `phase3_attnres_pp_integration/handoff_status_20260421.md` —
  per-rank cache distribution + the math behind P-1 delta blocks
* **Phase 4 retrain context**: `phase4_kimi_attnres_lm_pretrain/README.md`, especially the
  "Continuation-pretrain to 100K steps" section
* **Cache adapter design doc**: `phase3_attnres_pp_integration/adapter_design.md`
  (state machine + invariants; the local-cache-capture grad-bridge
  story is here)
* **Existing phase5 code**: `phase5_vlm_multimodal_sft/multimodal_*.py` + `train_mm.py`
  (Arm 1's working FSDP-only multimodal — the launching point for
  Arm 2's PP extension)

---

## 14. Quick reality check before starting

If you read this and felt "this is too many bugs" — that's correct,
and that's the point. Megatron's open-source multimodal recipes do
NOT solve this; they replicate vision tower and run PP only on the
LM with full-shape send/recv (pad to global max). The cache-adapter
twist on top of multimodal is the genuinely new content, and the bugs
in § 10 are all things the prior agent's analysis surfaced as likely-
but-unconfirmed.

If you confirm AND fix all of them and arrive at FSDP/PP loss
alignment within seed-noise — that's the result.

If you confirm only some, root-cause why the others didn't matter, or
why the alignment fails on a specific bug — that's also the result.

What's NOT a result: a smoke that runs but you can't account for what
each bug above did or didn't do. Treat § 10 as your debug checklist;
each item should end up either "fixed" or "confirmed not present and
why".
