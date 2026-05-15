# PR #9 — AttnRes block-aggregation einsum → manual broadcast+sum (cuBLAS strided batched bf16 + fp8 dequant bypass)

**Target repo**: `sgl-project/sglang`
**Target file**: `python/sglang/srt/layers/attn_res.py` (overlay-side fix).
**Cross-link**: cuBLAS / driver-side root cause may belong on NVIDIA's side; file a separate reproducer issue (no patch) for that.

**Fork reference**: bundled inside commit `a6c46168a` on `QIU023/sglang@attention_residual_inference` (alongside PR #7 and PR #8).
**Effort**: ~1 hour (3-call manual broadcast+sum patch; already implemented in fork).
**Risk**: low — bf16 path identical FLOPs (verified 44.6 → 44.6 tok/s); fp8 path goes from crashed to working.
**Filing dependency**: requires PR #5 (overlay itself) to land first — the file `layers/attn_res.py` only exists upstream after #5.

---

## Suggested PR title

> [layers/attn_res] Replace 3 block-aggregation einsums with manual broadcast+sum to bypass cuBLAS strided batched bf16 GEMM failure under fp8 dequant

---

## Summary

Replaces three `torch.einsum("n..., n...d -> ...d", w, V)` calls inside
`block_attn_res()`, the vectorised `block_attn_res_phase1()`, and the
per-query fallback, with manual `(weights.unsqueeze(-1) * V).sum(dim=0)`.

The einsums naturally decompose into a `cublasGemmStridedBatchedEx`
call with `CUDA_R_16BF` accumulation — which fails with
`CUBLAS_STATUS_EXECUTION_FAILED` on RTX 5090 whenever an upstream
tensor came out of an fp8 dequant path. Pure bf16 path works
unchanged. Manual broadcast+sum bypasses the GEMM entirely.

## Symptom (without this PR)

```
Engine(quantization="fp8", model="<AttnRes-Kimi-Linear>")

cuBLAS error: CUBLAS_STATUS_EXECUTION_FAILED
  cublasGemmStridedBatchedEx CUDA_R_16BF
  at <one of the 3 einsums in layers/attn_res.py>
```

Reproducer: any model with an fp8-quantized layer feeding into
`torch.einsum("n..., n...d -> ...d", w, V)` with N ≤ 16 and D ≥ 512 on
RTX 5090.

## Patch

```python
# python/sglang/srt/layers/attn_res.py

# OLD (3 call sites, all the same shape):
out = torch.einsum("n..., n...d -> ...d", weights, V)

# NEW: manual broadcast + sum
out = (weights.unsqueeze(-1) * V).sum(dim=0)
```

bf16 verification:

| Path | tok/s |
|---|---|
| bf16 baseline (before fix) | 44.6 |
| bf16 with broadcast+sum | 44.6 |

No measurable throughput change on bf16; fp8 weight-only path goes
from crashing to working (38.9 tok/s coherent 8/8 vs bf16 44.6).

## Why this is partly an upstream cuBLAS / driver concern

The cuBLAS error itself is `CUBLAS_STATUS_EXECUTION_FAILED on
cublasGemmStridedBatchedEx CUDA_R_16BF` **only when an upstream tensor
came out of an fp8 dequant path** — same shapes, same strides, same
dtype, same kernel. Pure bf16 input works; bf16-after-fp8-dequant
fails. Two hypotheses:

1. cuBLAS Blackwell bf16 GEMM has an alignment requirement that fp8
   dequant violates.
2. SGLang's fp8 dequant return path produces a stride / storage layout
   that cuBLAS rejects.

The overlay-side `.contiguous()` defense was tried first and does NOT
fix it (rules out simple contiguity). Manual broadcast+sum sidesteps
the GEMM but doesn't explain *why* cuBLAS rejects this specific
combination.

**File a separate reproducer issue with NVIDIA** (or sglang as a
cross-link) including:

- the exact RTX 5090 driver version,
- the cuBLAS version,
- a minimal repro that produces an fp8-dequanted bf16 tensor of shape
  `(N, ..., D)` with N=8, D=1024 and tries the strided GEMM,
- the failure mode.

This PR is the overlay-side workaround; the deeper cuBLAS-side issue
is independent.

## Filing dependency

The file `python/sglang/srt/layers/attn_res.py` does not exist upstream
yet — it lands as part of PR #5 (the AttnRes inference overlay).
**File this PR after #5 lands**, OR fold the broadcast+sum change into
#5's initial commit so the overlay ships fp8-compatible from day 1
(cleaner — one PR, one working overlay).

## Filing checklist

- [ ] Wait for PR #5 (or be folded into it).
- [ ] PR body links the cuBLAS reproducer issue.
- [ ] bf16 regression: `bench_one_batch.py` throughput unchanged.
- [ ] fp8 smoke: coherent 8/8 generation on hf_step3100.
