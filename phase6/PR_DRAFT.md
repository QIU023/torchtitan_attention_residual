# Draft PR description for upstream torchtitan

This file is consumed verbatim into the eventual `pytorch/torchtitan`
PR description when the Kimi-NextGen-AttnRes model release becomes
the merge trigger. Update as new phase 6 items land.

---

## Title

```
Add Block AttnRes + cross-stage cache adapter, with multimodal trainer
```

## Summary

This PR brings two related pieces into torchtitan:

1. **Block Attention Residuals (AttnRes)** — the residual-stream
   replacement from [*Attention Residuals* (Kimi Team, 2026), arXiv:2603.15031](https://arxiv.org/abs/2603.15031).
   The math is `h_{l+1} = sum_i softmax(<w_l, h_i>) h_i` over earlier
   *block boundary* hidden states, replacing the standard
   `h_{l+1} = h_l + f_l(h_l)`. Pseudo-queries `w_l` are zero-initialized
   so initial softmax is uniform, making training start as if standard
   residuals are still in effect.

2. **Cross-stage caching pipeline adapter** — under
   `Interleaved1F1B` PP, a per-rank cache that ships only the
   newly-committed AttnRes block delta across each stage hop instead
   of the full accumulated block stack. Loss-invariant vs naive PP
   by construction (zero math difference; only network-bytes
   difference). Activated by `TORCHTITAN_ATTNRES_CACHE=1`.

Both pieces compose with FSDP2, FSDP+PP, FSDP+PP+TP, and the
multimodal trainer in this same folder.

## Multimodal trainer (phase 5/6)

`phase5/train_mm.py` (workspace location; lands at
`torchtitan/extensions/multimodal/` in the upstream PR) is a Trainer
subclass that:

* Wraps SigLIP frozen vision tower + small trainable MLP projector
  on the PP first stage only.
* Injects `vision_embeds` into the input dict via
  `post_dataloading_process` so the standard PP/FSDP forward-backward
  path handles both uniformly.
* FSDP2-wraps the projector on the dp/batch mesh so its grads
  reduce-scatter across DP ranks. Without this wrap, replicated-but-not-
  shared projector copies diverged silently.
* Resolves the image-sentinel token id from a per-tokenizer registry
  (`phase5/sentinel_registry.py`) with startup collision check, instead
  of hardcoding a number.

## Verified configurations

| Backbone | Parallelism | Multimodal | Loss alignment vs FSDP-only |
|---|---|---|---|
| Llama-3 175M (text-only) | PP=4 V=2 + cache adapter | No | max\|Δ\|=0.013 nats (Phase 3) |
| Kimi-Linear AttnRes 436M | PP=4 V=2 + cache adapter | No | passing (Phase 4) |
| Kimi-Linear AttnRes 436M | PP=4 V=2 + cache adapter | LLaVA-Pretrain | median \|Δ\|=0.024 nats / max 0.252 (warmup transient), 2000 steps from a multimodal-trained ckpt (Phase 6 A1, this PR) |

Full ablation report: `phase6/cache_adapter_ablation.md`. The closed-form
bytes-saved formula gives `≈ (N+1)/2` ratio when the AttnRes block count
N is small relative to the virtual stage count S; concretely for
`L=16, N=4, S=8` the ratio is 4× per stage hop.

## What this PR does NOT do

* Stitch AttnRes onto Qwen2 / Llama-3 base models. The reviewers asked
  to gate large-scale validation on the Kimi K3 release; this PR is
  infra only and does not claim quality on non-Kimi backbones.
* Implement the full scaling-law sweep from paper Table 2. The
  `kimi_linear_*_block_attn_res_*` flavor configs ARE registered for
  reproducibility but no GPU-time-burning sweep results are bundled.
* Vision-tower FSDP shard for >1B vision encoders. Spec'd in the
  multimodal trainer (`wrap_vision_tower(parallel_dims)` API surface)
  but deferred until a >4 GB-per-rank vision encoder lands.

## Test coverage

Run `pytest torchtitan/torchtitan/experiments/{attn_res,kimi_linear}/tests/
phase5/tests/`:

* `attn_res/tests/test_attn_res.py` — primitive / projection / stack-unstack
* `attn_res/tests/test_attn_res_dsv3.py` — DSv3 MoE composition (4 currently
  fail on CPU due to a pre-existing torchtitan moe.py CPU NotImplementedError;
  unrelated to this PR)
* `attn_res/tests/test_pipeline_adapter.py` — naive + delta mode dispatch,
  rank-local cache, capture-count audit
* `kimi_linear/tests/test_attn_res_model.py` — pseudo-query zero-init
  invariant (paper §5 requirement)
* `kimi_linear/tests/test_layers.py` — KDA / MLA / MoE / decoder block
  shape regression
* `kimi_linear/tests/test_model_spec.py` — flavor → ModelSpec dispatch
* `kimi_linear/tests/test_pipeline_adapter.py` — adapter passes through
  the kimi_linear-specific block contract unchanged
* `kimi_linear/tests/test_multimodal_model.py` — multimodal scatter
  forward shape
* `phase5/tests/test_pp_vision_plumbing.py` (4 tests) — vision_embeds +
  image_token_id kwarg survival through CrossStageCacheAdapter on stage 0
  / middle stage; collate fixed-len under variable per-row caption length
* `phase5/tests/test_variable_image_count.py` (6 tests, this PR) —
  uniform / mixed (zero, half, full) / all-zero / PP shape-inference /
  caller-supplied image_mask / overflow detection
* `phase5/tests/test_sentinel_registry.py` (9 tests, this PR) — registry
  hit per tokenizer family, fallback, collision check under/over
  threshold, strict raise, reserved-skip, unknown-role-rejected

Total: 97 + 19 = 116 CPU tests passing + 4 DSv3 pre-existing CPU
NotImplementedErrors (not introduced by this PR).

## Phase-by-phase provenance

| Phase | Purpose | Key artifact |
|---|---|---|
| 2 | DSv3-baseline FSDP bringup | `attn_res/model.py` |
| 3 | PP=4 V=2 + cache adapter on Llama3 175M | `attn_res/pipeline_adapter.py`, `attn_res/layout.py` |
| 4 | Kimi-Linear AttnRes 436M faithful reimplementation | `kimi_linear/{model,attn_res_model,parallelize}.py` |
| 5 | Multimodal trainer (LLaVA-Pretrain) — FSDP arm + PP+adapter arm with synthetic C4 | `phase5/{train_mm,multimodal_dataset,multimodal_model}.py` |
| 6 | Pre-merge infra completeness — fix projector grad sync; variable image count; sentinel registry; cache-adapter ablation; cross-parallelism determinism | this folder |

## Reviewers

Original RFC: pytorch/torchtitan#3029. Maintainer ask: gate merge on
Kimi K3 / NextGen-AttnRes release, which provides external large-scale
validation. This PR delivers the framework support so the merge is a
one-line model registration when the model lands.
