# [RFC] Block Attention Residuals for torchtitan

## Problem

Standard residuals `h_{l+1} = h_l + f_l(h_l)` give every layer equal weight;
hidden-state magnitude grows linearly with depth and shallow-layer signal
is diluted. At larger scales this shows up as training dynamics skewed
toward late layers and reduced loss-per-FLOP efficiency.

[Attention Residuals (Kimi Team, 2026)](https://arxiv.org/abs/2603.15031)
replaces the fixed add with softmax attention over preceding block
outputs, using a per-layer learned pseudo-query. The paper reports
**AttnRes ≈ baseline × 1.25 compute** at matched model size. No
open-source framework has integrated it yet.

## Solution

**Block AttnRes**: partition `L` layers into `N` blocks, standard
residuals within a block, softmax attention at block boundaries. Each
layer's `block_attn_res(blocks, partial, proj, norm)` produces the next
sub-layer's input as `softmax(w_l · RMSNorm(V)) · V` over the stacked
block representations. Pseudo-queries zero-initialized so step 0 is
numerically equivalent to standard residuals.

Block boundaries align with PP stage boundaries (PP-friendly: `O(N d)`
cross-stage traffic vs `O(L d)` for Full AttnRes). The real engineering
win is the **cross-stage caching adapter** that keeps per-stage send
size constant in stage id.

## Placement

Self-contained experiment under `torchtitan/experiments/attn_res/`. No
core modifications: subclasses `Llama3Model` / `Llama3TransformerBlock`
for the forward path, and provides a custom `ModelSpec.pipelining_fn`
(`pipeline_llm_with_cache_adapter`) that wraps stages with the adapter.
Follows the `transformers_modeling_backend` precedent.

## Evidence (single GPU, Phase 2)

Llama3-150M dense (12 layers, 75 M params), BF16 FSDP, C4-en, 20 k
steps, identical config except `model_spec`:

| step | baseline | AttnRes | Δ |
|---:|---:|---:|---:|
| 500 | 6.141 | 6.015 | **−0.127** |
| 5000 | 4.358 | 4.270 | −0.088 |
| 10000 | 4.324 | 4.219 | −0.104 |
| 15000 | 3.737 | 3.686 | −0.051 |
| 20000 | 3.685 | 3.619 | **−0.066** |

AttnRes is below baseline at every logged milestone. `num_blocks`
ablation at 150M favors N=6 (Δ=−0.066) over N=3 (Δ=−0.030); N=12 in
progress.

## Plan

- **PR #1 (this RFC)**: `experiments/attn_res/` with primitive, Llama3
  subclass, unit tests, and the single-GPU evidence above. Ready.
- **PR #2 (follow-up, in flight)**: cross-stage caching adapter on
  `8 × RTX 5090 PCIe, PP=8, Llama3 1-2 B, interleaved 1F1B`. Target:
  step-time overhead < 5 % over PCIe (intentionally the cheap
  interconnect). Reported: loss parity with naive PP, per-stage send
  size constant in stage id, NCCL comm trace, memory 5.5 d vs 3 d per
  layer, full scale-up loss curve at 1-2 B dense pretraining.

**Status**: standard `torch.distributed.pipelining` assumes a fixed
activation tensor shape across stages, but Block AttnRes's per-stage
send payload is `(partial, new_blocks_committed_this_stage)` where the
second tensor's leading dim grows with `stage_id` (naive path) or is
constant but matched across stages under the adapter. A first cut
using `torch.autograd.Function` for grad send-back proved brittle under
interleaved 1F1B recomputation, so the adapter is being reimplemented
around a custom effective-PP path that does explicit NCCL P2P outside
autograd, keyed on integer `(microbatch, producer_stage, block_idx)`
tags. Scale-up 1-2 B benchmark runs once that lands.

## Open questions for maintainers

1. **Adapter hook surface.** Wrapping `stage.submod` via a custom
   `pipelining_fn` requires walking `schedule._stages` (private torch
   attr). Is there a cleaner canonical extension?
2. **Variable-shape activations between stages.** Our cross-stage
   tensor has a leading dim that depends on `stage_id`. Is there
   precedent / a recommended pattern for this in torchtitan or
   `torch.distributed.pipelining`, beyond bypassing the built-in P2P?
3. **VP chunk keying.** Cache per `(microbatch_id, virtual_stage_id)`
   or per logical-depth block index?

## Reference

- Paper: [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)
- Reference impl: [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- Kimi infra engineer's implementation notes:
  [zhihu](https://www.zhihu.com/question/2016993095078684011)
- Branch: [QIU023/torchtitan@attention_residual_dev](https://github.com/QIU023/torchtitan/tree/attention_residual_dev)
- Owner: @QIU023
