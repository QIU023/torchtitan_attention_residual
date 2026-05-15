# PR #2 — base64 data-URL support in `attn_res_vl` image loader

**Target repo**: `sgl-project/sglang`
**Target file**: `python/sglang/srt/multimodal/processors/attn_res_vl.py`
**Fork reference**: commit `850ebb715` on `QIU023/sglang@attention_residual_inference`.
**Effort**: ~20 min (6-line patch + tiny test).
**Risk**: low — additive code path; existing file-path and URL handling unchanged.
**Filing dependency**: blocks on PR #5 (`attn_res_vl` processor itself only exists in fork as part of the AttnRes VL overlay).

---

## Suggested PR title

> [multimodal/attn_res_vl] Accept base64 data-URL image inputs

---

## Summary

`attn_res_vl` processor's image loader currently accepts only filesystem
paths and remote URLs. This PR adds a third branch handling base64
data-URL inputs (`data:image/png;base64,...`), matching the OpenAI
vision API spec and removing the need for callers to write image bytes
to disk first.

## Motivation

RL rollouts and async streaming pipelines commonly pass image bytes
inline (no temp file, no remote URL). Without data-URL support, every
caller has to:

1. Decode the inline bytes themselves.
2. Write to a temp file.
3. Pass the path to SGLang.
4. Clean up.

Direct data-URL parsing collapses (1)-(4) into 3 lines inside the
processor. The OpenAI / Anthropic vision API specs already standardise
this format; SGLang should match.

## Patch

```python
# python/sglang/srt/multimodal/processors/attn_res_vl.py

if isinstance(item, str):
    if item.startswith("data:image/") and ";base64," in item:
        _, _, payload = item.partition(",")
        return Image.open(BytesIO(b64.b64decode(payload))).convert("RGB")
    return Image.open(item).convert("RGB")  # existing path / URL behaviour
```

## Test plan

Add a unit test under
`python/sglang/test/srt/multimodal/test_attn_res_vl_image_loader.py`
that asserts a known 1×1 RGB PNG round-trips through base64 / data-URL
and decodes to the right RGB values.

## Why filing depends on PR #5

The `attn_res_vl.py` processor file does not exist in upstream sglang
today; it lands as part of PR #5 (Block AttnRes inference overlay).
File this PR after #5 merges, or fold it into #5 as one of the
processor's day-1 features.
