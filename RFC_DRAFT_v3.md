# [RFC] Block Attention Residuals for torchtitan

## Author

**Yiqiao Qiu** — Software Engineer, AWS
[LinkedIn](https://www.linkedin.com/in/yiqiao-qiu-12013723a/)

## Problem

Standard residuals `h_{l+1} = h_l + f_l(h_l)` give every layer equal
weight; hidden-state magnitude grows linearly with depth and
shallow-layer signal is diluted. At larger scales this shows up as
training dynamics skewed toward late layers and reduced loss-per-FLOP
efficiency.

[Attention Residuals (Kimi Team, 2026)](https://arxiv.org/abs/2603.15031)
replaces the fixed add with softmax attention over preceding block
outputs, using a per-layer learned pseudo-query. The paper reports
**AttnRes ≈ baseline × 1.25 compute** at matched model size and <4 %
end-to-end training overhead under pipeline parallelism. No open-source
framework has integrated it yet.

## Solution

**Block AttnRes**: partition `L` layers into `N` blocks, standard
residuals within a block, softmax attention at block boundaries. Each
layer's `block_attn_res(blocks, partial, proj, norm)` returns the next
sub-layer's input as `softmax(w_l · RMSNorm(V)) · V` over the stacked
block representations. Pseudo-queries are **zero-initialized**, so step 0
is numerically equivalent to standard residuals (the softmax is uniform)
and the model can begin training without any warmup schedule change.

Block boundaries align with PP stage boundaries, which is the PP-friendly
property the paper exploits: `O(N d)` cross-stage traffic vs `O(L d)` for
Full AttnRes. The PP cross-stage caching adapter (per paper §4.1) is
implemented and validated on 8× 5090 PCIe — it lives in **PR #2**, not
this PR, so PR #1 can land as a self-contained single-GPU experiment
first. See Plan below.

## Placement

Self-contained experiment under `torchtitan/experiments/attn_res/`. No
core modifications:

- `AttnResLlama3Model` and `AttnResLlama3TransformerBlock` subclass the
  core `Llama3Model` / `Llama3TransformerBlock` and override `forward` to
  route through `block_attn_res` when AttnRes kwargs are provided. When
  those kwargs are absent the subclass is bitwise-identical to the core
  forward.
- A separate `ModelSpec` is registered (`attn_res.model_registry`), so
  `--module attn_res` routes to this experiment without touching
  `--module llama3`.
- Config registry declares `llama3_175m_baseline` and
  `llama3_175m_attn_res` that share every hyperparameter except
  `model_spec`, so the measured delta reflects only the AttnRes change.

Follows the `transformers_modeling_backend` precedent for extending a
model family without modifying `core`.

## Evidence (single RTX 5090, FSDP2, BF16)

### Model

12-layer Llama3 dense (dim 768, n_heads 12, n_kv_heads 4, SwiGLU FFN,
tied embeddings, vocab 128,256). Total physical parameters
`174,017,280` (the 98.5M tied embed/output counted once via
`model.parameters()`), hence the `175M` in the flavor name.
torchtitan's `size:` log applies its weight-tying convention
(`torchtitan/models/utils.py:430-432`: `nparams -= nparams_embedding`
when tying is enabled) and reports only the non-embedding part:

| Flavor | torchtitan `size:` (non-embedding, tied convention) | Δ vs baseline |
| --- | ---: | ---: |
| `llama3_175m_baseline` | 75,516,672 | — |
| `llama3_175m_attn_res` (N=6) | 75,555,072 | +38,400 |

AttnRes adds per-layer pseudo-query + RMSNorm on pre-attn and pre-MLP
residual reads plus a final cross-block aggregation:
`12 × 2 × (768 + 768) + 2 × 768 = 38,400` parameters. That is
`0.05 %` of the transformer stack, negligible.

### Training config (identical for both runs)

| Setting | Value |
| --- | --- |
| dataset | C4 (`allenai/c4`, English, HF streaming) |
| tokenizer | `NousResearch/Meta-Llama-3.1-8B` (mirrors Llama-3.1 tokenizer, vocab 128,256) |
| seq_len | 2048 |
| local_batch_size | 8 |
| global_batch_size | 16 |
| grad_accum | 2 |
| steps | 20,000 (≈ 650 M tokens) |
| lr | 3e-4, cosine, warmup 500, decay ratio 0.8 |
| optimizer | AdamW |
| precision | BF16 mixed (params/grads BF16, reduce fp32) |
| FSDP | FSDP2 (fully_shard) |
| seed | torchtitan default (not set explicitly); identical between runs |

The full configs are in
[`experiments/attn_res/config_registry.py`](https://github.com/QIU023/torchtitan/blob/attention_residual_dev/torchtitan/experiments/attn_res/config_registry.py).

### Loss vs. step

| step | baseline | AttnRes (N=6) | Δ |
| ---: | ---: | ---: | ---: |
| 500 | 6.141 | 6.015 | **−0.127** |
| 5000 | 4.357 | 4.270 | −0.088 |
| 10000 | 4.324 | 4.219 | −0.105 |
| 15000 | 3.737 | 3.686 | −0.051 |
| 20000 | 3.685 | 3.619 | **−0.066** |

AttnRes is below baseline at every logged milestone. The step-500 gap
(−0.127) is the "first-block of cross-block attention kicks in"
transient. Delta shrinks over training (−0.127 at step 500 → −0.066
final), consistent with the paper's smaller asymptotic gap on
larger-scale runs.

### `num_blocks` ablation (step 20,000)

| N | Final loss | Δ vs baseline | tps | TFLOPS | MFU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.685 | — | 70,660 | 48.02 | 15.39 % |
| 3 | 3.655 | −0.030 | 52,664 | 35.80 | 11.48 % |
| 6 | **3.619** | **−0.066** | 49,412 | 33.59 | 10.77 % |
| 12 | 3.623 | −0.061 | 26,437 | 17.97 | 5.76 % |

N=6 and N=12 are statistically indistinguishable (gap within smoothing
noise); N=3 clearly underperforms. N=6 is the preferred operating
point because it matches N=12 in loss at ~2× throughput — the stacked
value tensor `[N+1, B, T, D]` becomes memory-bandwidth-bound during
`block_attn_res` for large N.

**On N=8.** The paper names N=8 as the sweet spot at L≥32 (54-layer
Kimi Linear, 6 layers/block). At L=12 the sweet-spot region widens:
N=6 and N=12 both work, which matches the paper's prediction that
Full-AttnRes (N=L) does not degrade at shallow L. N=8 was not run
because 8 does not divide 12 cleanly (N=6 and N=4 are the natural
divisors below 12). We will include N=8 when we scale to L=16 in the
PP follow-up.

### Single-GPU throughput and memory overhead

On a single RTX 5090 with no communication to hide behind, Block AttnRes
adds visible compute:

| Metric | baseline | AttnRes N=6 | delta |
| --- | ---: | ---: | ---: |
| tokens / sec / GPU | 70,660 | 49,412 | **−30.1 %** |
| MFU (BF16, 5090 peak 312 TF) | 15.39 % | 10.77 % | −4.62 pp |
| Peak memory | 29.11 GiB (92.83 %) | 30.05 GiB (95.84 %) | +0.94 GiB / +3.01 pp |

This is higher than the paper's **<4 % PP overhead** number because
that number measures end-to-end throughput under interleaved 1F1B
where AttnRes compute overlaps with PP communication. On a single GPU
there is no communication to overlap with, so AttnRes compute shows up
directly. The target audience for this PR is the algorithm integration
itself — the PP throughput story requires the cross-stage caching
adapter (PR #2).

Activation memory retries were observed on all AttnRes runs (tight
bs=8 at seq=2048 on 32 GiB). The delta in peak memory (+0.94 GiB)
matches the paper's predicted per-layer activation increase from 3 d
to 5.5 d.

Profile traces / memory snapshots are not captured for this PR — they
become genuinely useful at PP scale in PR #2 and we plan to include
them there.

### Checkpoint compatibility

- **Core Llama-3 checkpoint into `AttnResLlama3Model`**: loads cleanly.
  All AttnRes-specific params (pseudo-queries + their RMSNorms) are
  missing from the checkpoint; torchtitan's state-dict loader
  tolerates missing keys when they are initialized on the model side.
  Pseudo-queries are already initialized to zero, so loading a base
  checkpoint leaves the model in the "AttnRes degenerates to uniform
  softmax = standard residual" state at step 0. Fine-tuning from a
  core checkpoint into AttnRes is therefore safe.
- **`AttnResLlama3Model` checkpoint into core `Llama3Model`**: fails
  with unexpected keys unless loaded with `strict=False`. This is
  intended (an AttnRes checkpoint is not a Llama-3 checkpoint); we
  document the asymmetry rather than silently dropping keys.

## Plan

### PR #1 (this RFC — ready)

`experiments/attn_res/` containing:

- `attn_res.py`: primitive, `AttnResConfig`, `AttnResProjection`
  (zero-initialized), `stack_blocks` / `unstack_blocks`.
- `model.py`: `AttnResLlama3Model` and `AttnResLlama3TransformerBlock`
  subclasses.
- `__init__.py`, `config_registry.py`: flavors
  `debugmodel_attn_res`, `175M_attn_res`, and paired trainer configs.
- `tests/`: CPU unit tests for the primitive (zero-init equivalence,
  softmax invariants, stack/unstack round-trip, gradient flow) and an
  end-to-end debug-model forward+backward.
- `README.md`: motivation, file inventory, design notes, run
  instructions, ownership.
- Integration-test workflow badge: happy to follow whichever pattern the
  maintainers prefer (`integration_test_Xgpu_<name>.yaml`, 1-GPU or
  8-GPU). Default plan is 1-GPU for PR #1 (the evidence above runs on
  one device) with an 8-GPU workflow added when PR #2 lands.

### PR #2 — Cross-stage caching adapter for AttnRes under PP

**Status**: implementation in progress.

Scope: a `pipelining_fn` hook (`pipeline_llm_with_cache_adapter`)
registered on `ModelSpec` that wraps each `PipelineStage.submod`
with a `CrossStageCacheAdapter`. When `TORCHTITAN_ATTNRES_CACHE=1`
and the schedule is `Interleaved1F1B`, the adapter replaces the
naive "send the full accumulated block stack at every hop" with a
static delta layout: at each hop the producer ships only the blocks
the receiver's rank does not already hold, computed at setup time
from `(PP, VP, num_blocks, n_layers, layers_per_block)` by
`BlockLayoutTables`. Cached blocks on the receiving rank are merged
with the incoming delta before being handed to the wrapped model.

All AttnRes PP code lives under `torchtitan/experiments/attn_res/`
(`pipeline_adapter.py` ≈ 900 lines, `layout.py` ≈ 270 lines;
comparable to `experiments/transformers_modeling_backend/pipeline.py`
≈ 419 lines). Zero modifications to torchtitan core or to
`torch.distributed.pipelining`.

**Current validation (8× RTX 5090 PCIe, 175M dense, PP=8 VP=2, M=4)**:

- Forward delta correct: each stage emits the shape predicted by
  the static layout table; torch's `_shape_inference` and runtime
  match.
- 1000-step loss curve matches naive PP within bf16 tolerance
  (Δ at step 1000 = 0.007). Numerically A/B-aligned.
- Adapter remains opt-in: unsetting the env flag falls back to the
  Phase-2 naive PP path with no behavioral change.

**Bandwidth framing (accurate).** Paper §4.1 guarantees the
cross-stage cache reuse from virtual stage `v ≥ 2` onward: `v=0`
is naive growth; starting at `v ≥ 2` each rank reuses its cached
blocks across virtual stages, so per-hop bytes drop to ∼`P · Np · d`
instead of ∼`v · P · Np · d`. The sustained-state benefit shows up
only with `VP ≥ 2`, which is why PR #2 is measured under
Interleaved1F1B with `VP=2`.

**Schedule coverage**: Interleaved1F1B only. Any other schedule
falls back to naive PP with a warning; adding
`ScheduleInterleavedZeroBubble` / `LoopedBFS` is a mechanical
extension of the layout tables and out of scope for PR #2's initial
landing.

**Model-size coverage for the PR**: correctness validated at 175M;
the 1.5–2B scale-up run for the PCIe-overhead headline plot is the
next step in this PR's validation track.

## Open questions

PP adapter optimization work is ongoing; specific open items are
tracked inside the experiment rather than in this RFC until they
stabilize. Comments on PR #1 (single-GPU AttnRes correctness) are
welcome in the meantime.

## Reference

- Paper: [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)
- Reference impl (README + PDF only): [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- Kimi infra engineer's implementation notes: [zhihu](https://www.zhihu.com/question/2016993095078684011)
- Branch: [QIU023/torchtitan@attention_residual_dev](https://github.com/QIU023/torchtitan/tree/attention_residual_dev)
- Owner: @QIU023
