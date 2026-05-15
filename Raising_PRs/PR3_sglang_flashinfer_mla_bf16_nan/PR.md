# PR #3 — flashinfer_mla bf16 NaN on high-magnitude residual streams (issue + fp32 fallback knob)

**Target repos**:
- **Issue**: `sgl-project/sglang` (primary) + `flashinfer-ai/flashinfer` (cross-link).
- **Patch (if accepted)**: `sgl-project/sglang` :: a new per-layer eager-fallback hook OR `flashinfer-ai/flashinfer` :: kernel-level fp32-scoring path.

**Fork reference**:
- `e8e7134ee` — `[AttnRes] fp32 MLA fallback: extend-only + write cache for native decode`
- `334990612` — `[AttnRes] fp32 MLA eager fallback to fix flashinfer_mla NaN on Blackwell`

**Effort**: file the issue (1 hour), patch shape pending upstream API decision.
**Risk**: medium — touches the engine's MLA dispatch path.

---

## Filing order

File the **issue first** (no patch). Let flashinfer / SGLang
maintainers weigh in on whether the fix belongs at:

1. **flashinfer kernel level**: a `--mla-fp32-scoring` flag that runs
   `Q @ K + softmax` in fp32, `V` multiply in bf16. Cleanest, but
   requires flashinfer team buy-in.
2. **SGLang per-layer hook**: third-party model classes install an
   eager-fallback callable for specific layers. More invasive but
   more general.

Our fork implements (2) (extend / prefill via fp32 eager SDPA, decode
via flashinfer_mla since per-step input is bounded). Either is
acceptable to us; let upstream choose.

---

## Issue body (suggested)

### Title

> `flashinfer_mla` returns NaN on Blackwell (RTX 5090, SM 12.0) with bf16 MLA + high-magnitude residual stream

### Repro

```
Model:    Kimi Linear AttnRes (1.4B-active, MLA attention head_dim=128)
Hardware: RTX 5090 SM 12.0 (Blackwell consumer), 32 GB VRAM
SGLang:   commit X (post 2026-05)
dtype:    bf16
Symptom:  NaN logits at deepest MLA layer when prefill input max ≈ 77
          (the AttnRes residual stream's natural magnitude at converged
          training).
```

Confirmed via:

- Same model + weights through `torch` eager forward: works.
- Alternative SGLang attention backends on Blackwell: `triton` OOMs,
  `fa3` needs SM 80-90, `torch_native` doesn't support MLA layout.
- Lowering activation magnitudes (untrained / early-step ckpt): works.

### Root cause hypothesis

flashinfer_mla's softmax scoring is bf16-only on Blackwell. High-
magnitude residuals saturate the bf16 exponent range; the softmax
denominator becomes 0; division produces NaN.

The bug is upstream in flashinfer — Blackwell's bf16 MLA kernel needs
either:

- a fp32 scoring intermediate (preferred — cheapest), or
- numerical-stability bounds on the input (harder — requires
  application-level cooperation).

### Our workaround (commits `e8e7134ee` + `334990612`)

In `python/sglang/srt/models/attn_res_overlay.py`, register a
per-layer eager-fallback so MLA layers run fp32 SDPA on prefill /
extend, while DECODE keeps native flashinfer_mla (per-step input is
bounded and stays in bf16's stable range). Cache layout preserved so
prefill→decode handoff is correct.

Performance cost on RTX 5090:

- prefill: ~2× slower (fp32 eager vs fused bf16 kernel)
- decode: unchanged (still flashinfer_mla)
- aggregate throughput: bf16 baseline 44.6 tok/s → with fp32 fallback
  41-42 tok/s on Kimi Linear AttnRes 1.4B-active.

### Suggested upstream form

Pick one (we'll PR whichever the maintainers prefer):

1. **flashinfer kernel knob**: `mla_fp32_scoring=True` in the kernel
   builder; routes the softmax intermediate to fp32. Bypasses the
   Blackwell saturation issue at the kernel level.
2. **SGLang per-layer eager-fallback hook**: a `register_eager_fallback`
   API that third-party model classes call during `init_layers`. Our
   fork's `attn_res_overlay.py` already exercises this shape.

### Filing checklist

- [ ] File the SGLang issue with the repro + workaround link.
- [ ] Cross-link a flashinfer issue with the same repro + the kernel-
      level fix proposal.
- [ ] If maintainers choose option (1), wait for the flashinfer fix;
      our SGLang fork's `attn_res_overlay` workaround stays until the
      kernel ships.
- [ ] If maintainers choose option (2), submit our `attn_res_overlay`
      hook as a separate PR (refactored into a general
      `EagerFallbackRegistry` instead of AttnRes-specific code).
