# PR #10 — `Fp8MoEMethod` user-visible warning on silent bf16 fallback

**Target repo**: `sgl-project/sglang`
**Target file**: `python/sglang/srt/layers/quantization/fp8.py` (`Fp8Config.get_quant_method`).
**Effort**: XS (~10 lines of logging).
**Risk**: low (logging only, no behaviour change).
**Status**: **Tentative** — only worth filing if PR #8's downstream ICA stays unresolved long enough that "fp8 weight-only on dense Linear + ignored MoE" becomes the canonical Blackwell config.

---

## Suggested PR title

> [fp8] Log a one-line INFO when `Fp8Config.get_quant_method` returns
> `UnquantizedFusedMoEMethod` due to `ignored_layers`

---

## Summary

When `Fp8Config.get_quant_method` returns `UnquantizedFusedMoEMethod`
for a `FusedMoE` layer because of `ignored_layers` (e.g. via
`SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts`), log a one-line INFO so
users understand their effective quantization scheme.

## Motivation

Today this fallback is **silent**. A user setting
`Engine(quantization="fp8", ...)` and `SGLANG_FP8_IGNORED_LAYERS` to
work around PR #8's downstream ICA (illegal memory access in the fp8
fused-MoE kernel on Blackwell consumer cards) has no indication that:

1. fp8 quantization applies only to dense Linear (q / k / v / o,
   gate / up / down).
2. The expensive MoE layer is still bf16.
3. The expected "fp8 weight-only" memory + throughput numbers will
   look different from the H100 reference.

Without the log line, users debug for hours wondering why their fp8
config produces ~85% of the bf16 throughput (vs the expected memory
+ throughput win).

## Patch

```python
# python/sglang/srt/layers/quantization/fp8.py

def get_quant_method(self, layer, prefix):
    ...
    if isinstance(layer, FusedMoE) and self._is_layer_ignored(prefix):
        logger.info(
            f"fp8: layer '{prefix}' is in ignored_layers; "
            f"falling back to UnquantizedFusedMoEMethod (bf16). "
            f"Set SGLANG_FP8_IGNORED_LAYERS env to control."
        )
        return UnquantizedFusedMoEMethod(...)
    ...
```

## When to file this PR

**Wait** until either:

1. PR #8's downstream ICA is **fixed** — then no one runs the "ignore
   MoE" config and this PR becomes pointless.
2. PR #8's downstream ICA stays open **6+ months** — at which point
   "fp8 dense + ignored MoE" is the de-facto Blackwell-consumer fp8
   config and the silent-fallback footgun is real enough to warrant a
   user-visible warning.

Until one of those happens, this PR sits in the inventory as
"file if needed".

## Filing checklist

- [ ] Check current status of PR #8's downstream ICA.
- [ ] If ICA fixed: close this PR proposal as obsolete.
- [ ] If ICA unresolved 6+ months: file this PR.
- [ ] PR body should reference the PR #8 followup-issue link.
