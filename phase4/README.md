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

## Continuation-pretrain to 100K steps (phase4d → phase5 bridge)

Started 2026-04-27 (`launch_continuation_100k.sh`). Baseline: Phase 4
overnight ckpt at step-12500, val loss = **3.73** on C4. Goal: train
the LM through enough additional steps that Phase 5 multimodal has a
non-LM-bottlenecked starting point.

### Why this run exists

Phase 5 multimodal smoke (single-stage full-param fine-tune of
AttnRes-Kimi-436M + frozen SigLIP + trainable MLP projector on
LLaVA-Pretrain-558K) ran 2K steps and showed loss descent stalling
near 3.8. Diagnosis: the LM itself only saw ~320M tokens during
Phase 4 (12500 steps × global_bs 12 × seq_len 2048), far short of
chinchilla-optimal ~9B for a 436M model. Captions inherit the LM's
linguistic ceiling; the multimodal experiment can't validate AttnRes
on a robust LM until the LM is robust.

### Target val loss tiers

| Tier | Val loss | Multimodal usefulness |
|------|----------|----------------------|
| Stretch | ≤ 2.8 | Pythia-410M-class LM, captions fluent + can ground objects |
| **Target (recommended stop)** | **≤ 3.0** | GPT-2-355M-class LM, captions basically fluent — primary stop trigger |
| Acceptable floor | ≤ 3.2 | Captions less repetitive but still primitive — fallback if 3.0 unreachable |
| Current baseline | 3.73 | LM-bottlenecked, captions primitive/repetitive |

Theoretical scaling-law extrapolation: 100K steps × 24K tokens =
2.5B tokens (8× the Phase 4 baseline). Loss ∝ N^(-α), α≈0.075:
3.73 × 8^(-0.075) ≈ 3.17 nats theoretical. With ~30% small-bs
plateau discount, realistic landing zone is **3.0-3.3**.

### Stop criteria — DO NOT stop the run unless one of these triggers

This is a long autonomous run (~46h). The agent monitors and applies
these rules without asking; the user can override at any time.

1. **PRIMARY (success)** — `val_loss ≤ 3.0` confirmed at any
   `--validator.freq` checkpoint (every 2500 steps). Stop the run,
   keep the latest ckpt, return to Phase 5 multimodal with this
   ckpt as initial weights.

2. **PLATEAU (real)** — val loss has not improved by ≥ 0.05 nats
   over **20K consecutive steps** (i.e. 8 consecutive validator
   checkpoints, since validator runs every 2500 steps). Stop the
   run — the model has settled into a small-bs local minimum that
   constant LR + same optimizer state can't escape. Return to
   Phase 5 with the best val ckpt seen so far (typically val
   3.2-3.4 in this scenario).

3. **PLATEAU (transient, do NOT stop)** — single-checkpoint val
   regression, or 2-3 consecutive non-improvements followed by
   another drop, is normal small-bs noise. Only "8 consecutive
   non-improvements with total drift < 0.05 nats" counts as real
   plateau.

4. **DIVERGENCE** — train loss spike > 5.5 sustained for 100+
   steps, OR grad_norm > 5.0 sustained, OR NaN. Stop, debug
   before relaunch.

5. **NEITHER (keep running)** — val still descending, even slowly.
   The full 100K is the budget; do not pre-empt.

### Resume strategy (if PLATEAU triggers)

If the run stops on plateau at val 3.2-3.4:

- The best ckpt is still **substantially better** than baseline
  (3.73 → 3.2 = 0.5-nat improvement = ~40% perplexity reduction).
- Phase 5 multimodal restart with that ckpt should converge faster
  and to a lower caption loss than the original phase5 (which
  stalled at 3.8 from baseline 3.73).
- Plateau means: bs is the bottleneck, not training duration. Future
  work could try (a) gradient accumulation to ↑ effective bs, (b)
  larger seq_len if memory allows, or (c) move to bigger-memory
  hardware (H100/H200) to fit paper bs=384 directly.
