# PR #5 — Block Attention Residuals inference overlay (Kimi + Qwen3 carriers)

**Target repo**: `sgl-project/sglang`
**New modules**:

- `python/sglang/srt/layers/attn_res.py` — algorithm core (model-agnostic).
- `python/sglang/srt/models/attn_res_overlay.py` — Kimi-Linear carrier overlay.
- `python/sglang/srt/models/qwen3_attn_res_overlay.py` — Qwen3 carrier overlay.
- `python/sglang/srt/models/attn_res_vl_overlay.py` — VLM (SigLIP + projector + AttnRes LM) carrier.
- `python/sglang/srt/configs/kimi_attn_res_vl.py` — HF config for the VL carrier.
- `python/sglang/srt/multimodal/processors/attn_res_vl.py` — multimodal processor.

**Effort**: L (~600-line algorithm + 2 carrier overlays + 1 VL carrier + processor).
**Risk**: high — new abstraction layer in `layers/`, dual-arch loader hook, fp32 MoE bias patch, RMSNorm contiguous shim.
**Track**: **Research / RFC** — needs paper-level academic legitimisation to land cleanly.

---

## Suggested PR title

> [RFC] Block Attention Residuals inference overlay (Kimi Linear + Qwen3 carriers)

---

## Summary

Block Attention Residuals (Kimi Team, 2026, [arXiv:2603.15031](https://arxiv.org/abs/2603.15031))
is a generic residual-stream overlay that replaces the standard
pre-norm residual with a learned aggregation over committed prior
blocks plus the current partial block. Same family of ideas as
ByteDance Hyper-Connections and DeepSeek mHC.

This PR adds an inference path for AttnRes-trained models in SGLang.
Two carriers are validated: Kimi Linear MoE (1.4B-active) and Qwen3
dense. The algorithm core (`layers/attn_res.py`) is model-agnostic;
carriers are thin wrappers (~150 lines each) that expose an
`EntryClass` for SGLang's registry to pick up via
`architectures: ["XxxBlockAttnResForCausalLM"]`.

## Motivation

Training-side AttnRes is documented in our fork's torchtitan
experiment (`QIU023/torchtitan@attention_residual_dev::experiments/attn_res/`)
with paper-Table-1 reproduction on 174M dense Llama3. The inference
path is **currently fork-only** — there is no public AttnRes inference
implementation in any production engine (vLLM, SGLang, TensorRT-LLM,
llama.cpp). Upstreaming the SGLang overlay closes that gap.

## Why research-track

Three reviewer concerns expected:

1. **Algorithm legitimacy**: a learned per-block aggregation looks
   like a one-off variant unless backed by a paper. The original
   Kimi paper (arXiv:2603.15031) + downstream Kimi K-series release
   are the legitimacy anchor. PR opens *after* the K-series release
   when AttnRes is "production-grade upstream method" not "one fork's
   experiment".
2. **Loader assumptions**: dual-arch hint
   (`architectures: ["KimiAttnResVLForConditionalGeneration",
   "KimiLinearForCausalLM"]`) triggers SGLang's MLA dispatch; fp32 MoE
   bias patch + RMSNorm contiguous shim are small but cross-cutting.
3. **Two-phase computation** (paper §4.1): the RS+merge+AG seq-shard
   fusion path is a separable feature (filed as PR #6).

## Patch surface

```
python/sglang/srt/layers/attn_res.py                                 ~250 LOC
python/sglang/srt/models/attn_res_overlay.py                         ~150 LOC
python/sglang/srt/models/qwen3_attn_res_overlay.py                    ~80 LOC
python/sglang/srt/models/attn_res_vl_overlay.py                      ~120 LOC
python/sglang/srt/configs/kimi_attn_res_vl.py                         ~60 LOC
python/sglang/srt/multimodal/processors/attn_res_vl.py                ~50 LOC
python/sglang/srt/utils/hf_transformers/common.py                  +small hook
python/sglang/srt/model_executor/model_runner.py                   +backend dispatch
```

## Suggested staging

Land in 3 PRs (after the RFC discussion concludes):

1. **algorithm-only**: `layers/attn_res.py` standalone — has its own
   unit tests, no model wiring. Lowest reviewer surface.
2. **Kimi-Linear carrier**: `models/attn_res_overlay.py` + the
   model_runner hook. Carries the AttnRes-Kimi-Linear LM weights.
3. **VL carrier + processor**: `models/attn_res_vl_overlay.py`,
   `configs/kimi_attn_res_vl.py`,
   `multimodal/processors/attn_res_vl.py`. Builds on (1) + (2),
   includes PR #2's base64 data-URL support as a day-1 feature.

(PR #6 ships as a separate documented feature.)

## Filing checklist

- [ ] Wait for Kimi K-series production model to land publicly so
      the algorithm has a deployment artifact to point at.
- [ ] File RFC issue first with the 3-PR staging plan + design
      decisions list (per-block aggregation API, eager-fallback hook
      registry, dual-arch dispatch).
- [ ] Reference paper [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)
      and our torchtitan experiment for the training-side counterpart.
- [ ] PR #5a (algorithm-only) ready to file once RFC closes.
