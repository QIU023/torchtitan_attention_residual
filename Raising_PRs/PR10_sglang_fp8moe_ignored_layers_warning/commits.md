# Backing commits — PR #10 fp8 MoE silent-fallback warning

## Discovered in

**Phase 11** — fp8 quantization sweep follow-up to PR #8. While
verifying the Blackwell smoke workaround
(`SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts`), noticed the fallback
was completely silent — no INFO / WARN line indicating MoE was
running bf16 under an `Engine(quantization="fp8")` call. This makes
the "fp8 dense + bf16 MoE" performance regime invisible to debugging.

## Fork source

**None.** This is a propose-only PR; no fork commit implements the
logging change. The proposed patch is small enough (~10 lines) to
write directly against upstream `Fp8Config.get_quant_method`.

## Status

**Tentative.** File only if PR #8's downstream ICA stays unresolved
long enough that "fp8 dense + bf16 MoE" becomes the canonical
Blackwell-consumer fp8 config.

## Filing recipe (when triggered)

```bash
# Only run this once PR #10 is decided ready to file.

git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout -b sglang-fp8-moe-fallback-warning upstream/main

# Hand-write the ~10-line patch in
# python/sglang/srt/layers/quantization/fp8.py inside
# Fp8Config.get_quant_method (see PR.md "Patch" section).

git add python/sglang/srt/layers/quantization/fp8.py
git commit -m "[fp8] Log INFO when MoE silently falls back to bf16 via ignored_layers

When Fp8Config.get_quant_method returns UnquantizedFusedMoEMethod for
a FusedMoE layer due to ignored_layers (e.g. via
SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts), log a one-line INFO so
users understand their effective quantization scheme.

Today the fallback is silent; users hitting PR #8's downstream ICA
(<link>) and using the ignored-MoE workaround have no indication
that fp8 only applies to dense Linear and MoE runs bf16. The
~85% bf16-baseline throughput then looks like a regression."

git push origin sglang-fp8-moe-fallback-warning
```

## Filing trigger criteria

File PR #10 when ANY of:

1. PR #8 lands but its downstream ICA stays open for ≥ 6 months.
2. Multiple users (issues / discord) hit the silent-fallback footgun
   on Blackwell consumer cards.
3. A SGLang maintainer asks for it during a different fp8-related
   PR review.

## Notes for the PR opener

- Pure logging change; no behaviour difference. Easy land.
- The log level choice is INFO (not WARN) since the fallback is
  user-requested via env var, not an error.
- Include a one-line note pointing users to PR #8's followup issue
  so they understand WHY the workaround exists.
