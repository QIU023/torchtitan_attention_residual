# PR #3 — filing instructions (issue-first, no PR)

## Status

🟠 **Issue ready to file; patch deferred pending upstream API decision.**

## What gets filed

**Two cross-linked issues**, no code PR yet:

1. **Primary issue on `sgl-project/sglang`** — the user-visible failure
2. **Cross-link issue on `flashinfer-ai/flashinfer`** — the kernel-level root cause

The bodies share the same repro and root-cause analysis; the action
each maintainer team is asked to take differs (sglang: choose between
kernel-level fix vs per-layer hook API; flashinfer: optionally add
`mla_fp32_scoring` knob).

## Where to file

| Repo | URL | What to file |
|---|---|---|
| sgl-project/sglang | https://github.com/sgl-project/sglang/issues/new | **Primary** issue (use body in [PR.md](PR.md) "Issue body (suggested)" section) |
| flashinfer-ai/flashinfer | https://github.com/flashinfer-ai/flashinfer/issues/new | **Cross-link** issue (mirror of the same body, with the kernel-level fix proposal as the recommended action) |

After both are filed, edit each to add the other's URL in a "Cross-linked" line.

## Title

```
[Blackwell SM 12.0] flashinfer_mla returns NaN on bf16 MLA + high-magnitude residual stream
```

## Body

Use [PR.md](PR.md) → "Issue body (suggested)" section verbatim. The body
already contains:

- Repro environment (model, hardware, sglang commit, dtype, symptom)
- Negative cross-checks (torch eager works; alternative backends OOM /
  unsupported)
- Root cause hypothesis (bf16 softmax saturates on Blackwell consumer
  cards)
- Our workaround (per-layer eager fp32 fallback at extend; cache layout
  preserved for native bf16 flashinfer_mla decode)
- Performance cost (44.6 → 41-42 tok/s aggregate, ~6% throughput hit)
- Two suggested upstream shapes (kernel-level fp32 scoring knob vs
  SGLang per-layer eager-fallback registry API)

## Why issue-first

The upstream-actionable shape is **not yet decided**. Filing a patch
would presuppose one of two API directions; the issue lets maintainers
weigh in and pick before any code is written. Estimated patch size
after direction is chosen: ~150 lines.

## Fork-side workaround reference

Our fork's `attn_res_overlay.py` implements the per-layer-hook variant.
Production verified by the Phase 11 GRPO infra:

- Prefill: fp32 eager MLA (`ATTNRES_MLA_FP32_FALLBACK=1`)
- Decode: `decode_attention_backend=torch_native`
- Numerical-stability companions: `ATTNRES_FP32_NORM=1`,
  `ATTNRES_INPUT_CLAMP=N`

These were verified on Kimi Linear AttnRes 1.4B-active; bench results
documented in `phase11_rlhf_grpo_infra/rlhf/run_grpo_kimi_attnres_with_trace.sh`
"workarounds_active" block.

## Cross-link with other PRs in this batch

- **PR #5** (AttnRes inference overlay) — the overlay this workaround lives
  inside; depends on Kimi K-series release for upstream legitimacy. PR #3
  blocks #5 from being framed as "works out of the box" until either the
  kernel-level fix lands flashinfer-side, or the per-layer hook API is
  approved sglang-side.
- **PR #7** (KDA causal_conv1d fp16) — independent kernel issue, also
  surfaced during the same Phase 11 fp16/fp8 sweep.
