# Phase 4: Kimi Linear + AttnRes faithful scale-up

## Goal

Port **MoonshotAI/Kimi-Linear** (48B-A3B MoE) to torchtitan-idiom so
AttnRes can be woven in and the Phase-3 PP cache adapter reused
verbatim. This is the platform for the paper's Table-2 scaling-law
sweep (194M → 528M activated params) and the eventual 48B-A3B
upscale target.

## Why a new `experiments/kimi_linear/` instead of extending `attn_res/`

`attn_res/` already has a DSv3-shaped `dsv3_16b_attn_res` flavor that
combines MLA + MoE + AttnRes, and it would be tempting to bolt a KDA
layer onto that scaffold. We explicitly do NOT go that route:

- **Fidelity over reuse.** DSv3 MLA and Kimi Linear MLA are
  architecturally similar but have differences that matter at
  training-time fidelity: `mla_use_nope=true`, specific head dims
  (`qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`),
  `q_lora_rank=null`, init scales, norm placements. Mixing the two
  MLA implementations would silently introduce DSv3 conventions that
  drift from the Kimi Linear spec, and the drift would not show up
  on the outside.
- **KDA is novel.** It's not an Attention subclass of the `attn_res`
  package's `BaseAttention.Config` abstraction without extending
  that abstraction; cleaner to house KDA next to its sibling layers
  in a dedicated experiment.
- **MoE+MLA+KDA interleave pattern is Kimi-specific.** The per-layer
  "KDA:MLA = 3:1, all layers MoE except first N dense" scheduling is
  baked into the config (`kda_layers`, `full_attn_layers`,
  `first_k_dense_replace`). Implementing that faithfully as its own
  model is cleaner than generalizing `AttnResModel`'s layer-config
  pattern.
- **`attn_res/` stays as the baseline/Llama3/DSv3 experiment bed**
  for sanity checks, ablations, and the PP adapter's CPU test suite.
  `kimi_linear/` is the production target.

Both experiments reuse the same `pipeline_llm_with_cache_adapter`
from Phase 3, so the PP adapter has zero Kimi-specific code.

## Architecture, per MoonshotAI's `modeling_kimi.py`

### Layer stack

```
┌───────────────────────────────────────────────────────┐
│  for layer_idx in 0..num_hidden_layers-1:             │
│                                                       │
│    x = input_layernorm (RMSNorm)                      │
│    if layer_idx + 1 in kda_layers:        ← KDA       │
│       attn_out = KimiDeltaAttention(x)                │
│    elif layer_idx + 1 in full_attn_layers: ← MLA      │
│       attn_out = KimiMLAAttention(x)                  │
│    x = residual + attn_out                            │
│                                                       │
│    x = post_attention_layernorm (RMSNorm)             │
│    if layer_idx >= first_k_dense_replace:             │
│       ffn_out = KimiSparseMoeBlock(x)     ← MoE       │
│    else:                                              │
│       ffn_out = KimiMLP(x)                ← dense FFN │
│    x = residual + ffn_out                             │
└───────────────────────────────────────────────────────┘
```

### 48B-A3B reference config highlights (from HF `config.json`)

| knob                                  | value |
|---------------------------------------|-------|
| `num_hidden_layers`                   | 27    |
| `hidden_size`                         | 2304  |
| `intermediate_size` (dense)           | 9216  |
| `kda_layers` (1-indexed)              | 20 layers: {1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26} |
| `full_attn_layers` (MLA, 1-indexed)   | 7 layers: {4,8,12,16,20,24,27} |
| **KDA : MLA ratio**                   | **20 : 7 ≈ 3 : 1** ✓ matches paper |
| `first_k_dense_replace`               | 1 (layer 0 is dense MLP; all others MoE) |
| MoE: `num_experts`                    | 256 |
| MoE: `num_experts_per_token`          | 8 |
| MoE: `num_shared_experts`             | 1 |
| MoE: `moe_intermediate_size`          | 1024 |
| MoE: `moe_router_activation_func`     | sigmoid |
| MoE: `use_grouped_topk`               | true (1 group, top 1 group) |
| MLA: `q_lora_rank`                    | null (no Q compression) |
| MLA: `kv_lora_rank`                   | 512 |
| MLA: `qk_nope_head_dim`               | 128 |
| MLA: `qk_rope_head_dim`               | 64 |
| MLA: `v_head_dim`                     | 128 |
| MLA: `mla_use_nope`                   | true |
| KDA: `num_heads`                      | 32 |
| KDA: `head_dim`                       | 128 |
| KDA: `short_conv_kernel_size`         | 4 |
| vocab                                 | 163840 |
| `tie_word_embeddings`                 | false |

Dependencies: `fla-core >= 0.5.0` (provides `chunk_kda`,
`fused_recurrent_kda`, `fused_kda_gate`, `ShortConvolution`,
`FusedRMSNormGated`). Confirmed installed and importable on this box.

## Scaling-law sweep targets (paper Table 2)

| size  | L_b | L (=2·L_b) | H   | d_model | d_ff | lr       | batch | tokens |
|-------|-----|-----------|-----|---------|------|----------|-------|--------|
| 194M  | 12  | 24        | 12  | 896     | 400  | 2.99e-3  | 192   | 38.7B  |
| 241M  | 13  | 26        | 13  | 960     | 432  | 2.80e-3  | 256   | 45.4B  |
| 296M  | 14  | 28        | 14  | 1024    | 464  | 2.50e-3  | 320   | 62.1B  |
| 436M  | 16  | 32        | 16  | 1168    | 528  | 2.20e-3  | 384   | 87.9B  |
| 528M  | 17  | 34        | 17  | 1264    | 560  | 2.02e-3  | 432   | 119.0B |
| **48B-A3B** (upscale) | 27 | — | 32 | 2304 | 9216 | — | — | — |

Paper variants per size: `Baseline` (no AttnRes), `Block AttnRes (N≈8)`,
`Full AttnRes`. All three use identical backbone hyperparameters so
the loss delta is attributable purely to AttnRes.

## AttnRes weave (no new math vs. Phase 3)

Same pattern as `attn_res/model.py:AttnResLlama3Model`, just applied
to `KimiLinearModel` blocks:

- Per **block start** layer (where `layer_idx % layers_per_block == 0`):
  add an `RMSNorm` + zero-initialized pseudo-query vector `w_l ∈ R^d`.
- AttnRes forward produces a weighted (softmax-over-sources) average
  of this layer's output and all prior block-start layers' outputs,
  then adds it to the residual stream.
- Pseudo-queries zero init → initial attention weights uniform →
  training begins equivalent to standard residuals (paper §5 claim).
- `_return_only_new_blocks` flag gates the forward to return only
  THIS stage's newly-committed blocks vs. full accumulated stack —
  directly consumed by the Phase-3 PP cache adapter.

Block AttnRes (paper's headline variant, N≈8) means exactly 8 block
boundaries distributed across L layers. E.g. 194M model has L=24,
N=8 → layers_per_block=3.

## PP cache adapter reuse (zero Kimi-specific code)

`pipeline_llm_with_cache_adapter` in `attn_res/pipeline_adapter.py`
only needs from the model:

1. `_return_only_new_blocks: bool` attribute (toggle)
2. Forward returns either `(partial_out, new_blocks_tensor)` or
   `(partial_out, full_blocks_tensor)` based on (1)
3. `_layers_per_block` or equivalent so `BlockLayoutTables` can be
   built

All three are satisfied by our AttnRes subclass. No PP adapter
changes. The layout tables (Phase-3 `BlockLayoutTables`) handle any
`(P, V, num_blocks, n_layers, layers_per_block)` tuple.

Validated in Phase 3 on 4-GPU PP=4 V=2 @ Llama3 175M. Kimi Linear's
transformer block is architecturally a different Attention type
(KDA/MLA), but the block **output contract** (a block-final hidden
state that AttnRes aggregates) is identical to Llama3, so the
adapter path reuses verbatim.

## File layout shipped

```
torchtitan/experiments/kimi_linear/
├── __init__.py                       ModelSpec registration
├── README.md                         Experiment scope + reproduction
├── reference/                        Verbatim fork from HF (NOT imported)
│   ├── modeling_kimi.py              1028 lines, blueprint
│   ├── configuration_kimi.py         140 lines
│   └── config.json                   48B-A3B reference config
├── model.py                          Torchtitan-idiom port:
│                                       - KimiDeltaAttention (KDA via fla-core)
│                                       - KimiMLAAttention
│                                       - KimiMoEGate + KimiSparseMoeBlock
│                                       - KimiMLP (dense FFN)
│                                       - KimiDecoderLayer
│                                       - KimiLinearModel
├── attn_res_model.py                 AttnRes subclass of KimiLinearModel
├── config_registry.py                5 scaling-law flavors + debug + 48B
└── tests/
    ├── __init__.py
    └── test_layers.py                CPU smoke for KDA/MLA/MoE/block shapes
```

## Phases of this implementation

**Phase 4a (this session):** plan doc + directory skeleton + reference
files fetched. NO model training.

**Phase 4b:** port `model.py` — KDA layer (using fla-core),
MLA layer (self-contained, not reusing DSv3 MLA), MoE gate + sparse
block, dense MLP, decoder layer, model. CPU smoke test: 2-layer
small config forward pass produces right-shaped output. Still NO
training.

**Phase 4c:** AttnRes subclass (`KimiLinearAttnResModel`) + the 5
scaling-law flavors + debug flavor. CPU tests for AttnRes weave on a
tiny config.

**Phase 4d:** 4-GPU PP smoke of `kimi_linear_debug_attn_res` with
Phase-3 `pipeline_llm_with_cache_adapter`, confirming the adapter
path works unchanged on this new architecture. Still no long
training.

**Phase 4e (far future):** rent 8+ GPU nodes, run the Table-2
scaling-law sweep. 48B-A3B needs multi-node + careful FSDP/MoE
sharding strategy — explicitly out of scope for current 4-GPU box.

## Out of scope

- HF weight conversion (there's an open-weights Kimi-Linear-48B-A3B-Base
  checkpoint on HF; porting those weights to torchtitan state-dict
  layout is a separate task).
- Kimi's specific tokenizer (`tokenization_kimi.py`). Training with
  Llama3 tokenizer is fine for our ablation study; matching Kimi's
  tokenizer only matters for HF-weight-loading or Kimi-vs-released
  comparisons.
- GenerationMixin / inference path. We're validating training-time
  loss, not deploying inference.
- Kimi's RoPE scaling (uses plain theta=10000 — no YaRN / linear
  scaling in the 48B-A3B config we're targeting).

## Sanity gates before the eventual training run

1. `python -m pytest torchtitan/experiments/kimi_linear/tests/` green
2. Debug flavor forward-pass end-to-end on CPU produces sensible
   logit shapes
3. Debug flavor forward on 1 GPU (no PP) produces finite loss at
   init ≈ log(vocab_size) ≈ log(163840) ≈ 12.0
4. Debug flavor on 4-GPU PP=2 V=2 (lps=1 for debug size L=4)
   completes 50 steps without RuntimeError
5. Debug AttnRes flavor + cache adapter ON completes 50 steps with
   same loss trajectory as adapter-OFF within bf16 noise

## Phase 4 LM retrain — 3 attempts → final val ~3.05 → marked complete

**Goal**: get the AttnRes-Kimi-436M LM val loss low enough to support
Phase 5 multimodal without LM-bottlenecking (original Phase 4 ckpt was
val 3.73, Phase 5 multimodal smoke stalled at caption loss ~3.8 from
that baseline). Three attempts, third succeeded, fourth failed
gracefully and we accept the third's result.

### Attempt 1 — `continuation_100k` (FAILED, disk-fill)

`launch_continuation_100k.sh`: resume from Phase 4 step-12500 with
constant LR=3e-4 for 87.5K more steps. Hit two issues:

* Conservative LR (1.4× the original cosine min 2.2e-4) was too low
  to escape the local min the step-12500 ckpt sat in. Val drift
  over 7500 steps: 3.73 → 3.71 (only -0.02 nats). Plateau.
* Disk filled at step ~10000 from KEEP_K=5 × 15 GB ckpts; ckpt
  write at step 10000 partial-failed and crashed all 4 ranks with
  zip-archive corruption.

Train log: `runs/kimi_436m_block_attn_res_fsdp_100k/train.log`.
Conclusion: continuation from a deeply-decayed ckpt with conservative
LR doesn't recover; need from-scratch with bigger effective batch.

### Attempt 2 — `from-scratch paperhparams + grad_accum=8` (SUCCEEDED)

`launch_from_scratch_paperhparams.sh`: fresh weights, paper LR=2.2e-3,
paper schedule (warmup 500, cosine decay_ratio 0.8, min_lr_factor 0.1),
**effective bs=96 via grad_accum=8** (LBS=3 × 4 ranks × 8 accum).
seq_len=2048 (HW cap), 12500 effective steps = 2.46B tokens (28% of
chinchilla-optimal 8.72B for 436M).

**Result**: clean run, val descent table:

| step | val | Δ |
|------|-----|---|
| 1     | 12.23 | (random init baseline) |
| 1000  | 4.02  | −8.21 |
| 2000  | 3.70  | −0.32 |
| 4000  | 3.47  | −0.23 (−0.115/1K) |
| 6000  | 3.34  | −0.13 |
| 8000  | 3.23  | −0.11 |
| 10000 | 3.13  | −0.10 |
| 11000 | 3.09  | −0.04 |
| 12000 | 3.07  | −0.025 |
| 12500 (final) | **~3.05** | −0.02 (extrapolated; final-step val not measured) |

Train log: `runs/kimi_436m_block_attn_res_fsdp/train_original_12500.log`.
Final ckpt is the **Phase 4 final** for downstream Phase 5/6 use.

Why grad_accum=8 was the key: paper LR=2.2e-3 calibrated for paper
bs=384. Original Phase 4 used bs=12 (1/32 of paper) at the same LR;
Adam noise was much higher than ideal. Effective bs=96 via grad_accum
(noise reduced sqrt(8)≈2.83×) brought Adam back into a regime where
paper LR + cosine schedule converges cleanly.

### Attempt 3 — `break-3.0 resume` (FAILED, post-min-LR destabilization)

`launch_paperhparams_break3.sh`: full-ckpt resume from step-12500 with
peak LR=3e-4 (1.36× post-cosine min 2.2e-4) and 200-step smooth warmup
ramp (warmup_steps=12700 — at step 12500 this gives LR = 12500/12700 ×
peak = 2.95e-4, no jump from 0). Goal: extend training another 17.5K
steps → 5.9B tokens = 68% chinchilla.

**Result**: model pushed OUT of step-12500 basin without finding a
better one.

| step | val | vs pre-resume baseline |
|------|-----|-----------------------|
| 12000 (pre-resume) | 3.07 | — |
| 13000 | 3.35 | **+0.28 ↑** |
| 14000 | 3.33 | +0.26 (no recovery in 1000 steps) |

Killed at step 14020 after 1500 steps confirmed plateau in elevated
basin. Train log:
`runs/kimi_436m_block_attn_res_fsdp/train_resume_break3_attempt2_killed.log`.

(There was an earlier attempt that died at step 12960 from a
HF Hub C4 streaming network glitch — not training-related — log:
`train_resume_break3_attempt1_hf_fail.log`.)

**Root cause**: model spent the last ~2000 steps of Attempt 2 at
LR=cosine-min (2.2e-4), so Adam's `v_t` calibrated for tiny gradient
magnitudes. Even the modest 1.36× LR ramp produced updates Adam
couldn't absorb cleanly — model drifted to a worse basin and the
new (smaller) cosine schedule didn't have enough headroom to recover.
General lesson: **don't resume from a fully-cosine-decayed ckpt with
a fresh LR schedule.** Stop Attempt 2 earlier (mid-cosine, e.g.
step 10000) if you want to extend, or use SGD + restart from scratch
instead.

## Phase 4 status: COMPLETE (val 3.05, accepted as final)

* **Final ckpt**: from Attempt 2 step-12500 (val ~3.05 extrapolated)
* **Phase 4 contribution to Phase 5/6**: this ckpt is the LM init
  consumed by `phase5_vlm_multimodal_sft/launch_train.sh` and `phase5_vlm_multimodal_sft/launch_pp_adapter.sh`
* No further LM training planned in Phase 4. Re-running with
  chinchilla-comfortable budget (e.g. STEPS=46500 → 9B tokens, ~6 days
  wallclock) would land val ~2.55-2.60 and is the path to take if
  Phase 5/6 surfaces LM bottlenecking again. Out of current scope.
