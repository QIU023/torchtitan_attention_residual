# PR #9 — filing instructions (cuBLAS issue first; sglang patch blocked on PR #5)

## Status

🟠 **Re-scoped 2026-05-17**. The overlay-side bypass patch in
`attn_res.py` cannot land upstream until **PR #5** (the AttnRes
inference overlay itself) lands — the file doesn't exist upstream yet.
What *can* be filed independently right now is the **cuBLAS root cause
issue**.

## What gets filed

1. **NOW**: an upstream issue against `pytorch/pytorch` (cuBLAS wrapping)
   describing the strided-batched bf16 GEMM failure when an input
   tensor is downstream of an fp8 dequant.
2. **LATER, after PR #5 lands**: the overlay-side `einsum →
   broadcast+sum` bypass either as a follow-up PR to sglang or folded
   into PR #5's initial commit.

## Where to file the cuBLAS issue

| Repo | URL | Why |
|---|---|---|
| pytorch/pytorch | https://github.com/pytorch/pytorch/issues/new | cuBLAS strided-batched GEMM is wrapped by `torch.einsum`; pytorch is the right surface for a user-reproducible bug report |
| sgl-project/sglang (optional cross-link) | https://github.com/sgl-project/sglang/issues/new | only if the maintainers want awareness that this affects sglang's fp8 + bf16 codepath |

Skip a direct NVIDIA cuBLAS bug report unless a pytorch maintainer
explicitly asks to escalate — pytorch issue thread is the right first
contact since users hit this through `torch.einsum`.

## Title (cuBLAS issue)

```
[cuBLAS Blackwell] cublasGemmStridedBatchedEx CUDA_R_16BF fails with CUBLAS_STATUS_EXECUTION_FAILED when an input tensor is downstream of an fp8 dequant
```

## Body (cuBLAS issue, copy-paste)

```markdown
## Description

`cublasGemmStridedBatchedEx` with `CUDA_R_16BF` accumulation fails with
`CUBLAS_STATUS_EXECUTION_FAILED` on Blackwell (RTX 5090, SM 12.0) when
one of the input tensors came out of an fp8 dequantization path —
even though both inputs end up as plain bf16 tensors of compatible
shapes and strides before the GEMM call.

The same call with pure bf16 inputs (no upstream fp8 dequant) succeeds.

## Repro environment

- **GPU**: NVIDIA RTX 5090 (SM 12.0, 32 GB)
- **Driver**: 591.86 (or any 580+)
- **CUDA**: 12.4 / 12.6 / 13.0 (all reproduce)
- **PyTorch**: 2.6.0+cu124 through 2.11.0+cu130 (all reproduce)
- **Trigger pattern**: `torch.einsum("n..., n...d -> ...d", w, V)`
  where `w` is bf16 and `V` is bf16-after-fp8-dequant, with N ≤ 16
  and D ≥ 512.

## Symptom

```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling
`cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha,
A, ..., CUDA_R_16BF, ..., CUDA_R_16BF, ...)`
```

Repro pseudocode (extract the dequant step from any sglang fp8
weight-only model that has an einsum head):

```python
import torch

D = 1024
N = 8

# Simulate a tensor that just came out of fp8 dequant
fp8_weight = torch.empty((N, D), dtype=torch.float8_e4m3fn, device="cuda")
fp8_weight.uniform_(-0.5, 0.5)
scale = torch.ones(1, device="cuda", dtype=torch.float32)
V = (fp8_weight.to(torch.float32) * scale).to(torch.bfloat16)

# This is the bf16 input that triggers the failure
assert V.dtype == torch.bfloat16
assert V.is_contiguous()

# Weights are pure bf16 — also fine
weights = torch.randn((N, 32, 64), dtype=torch.bfloat16, device="cuda")

# The crash:
out = torch.einsum("nij, nd -> ijd", weights, V)  # ← CUBLAS_STATUS_EXECUTION_FAILED
```

Same call with `V = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")`
(no upstream fp8 dequant) succeeds.

## What we've ruled out

- **Contiguity**: explicit `.contiguous()` on both inputs before the
  einsum does NOT fix.
- **Alignment**: `data_ptr() % 256 == 0` for both inputs.
- **Dtype**: both `.dtype == torch.bfloat16`.
- **Shape / stride**: re-verified, no surprises.

The pure-bf16 vs bf16-after-fp8-dequant difference is the only
discriminator.

## Hypotheses

1. cuBLAS Blackwell bf16 strided-batched GEMM has an alignment / layout
   requirement that the fp8 dequant kernel's output layout violates.
2. SGLang's fp8 dequant path produces a stride or memory layout that
   cuBLAS rejects, but the public PyTorch wrapper hides the difference.

Either way, the failure mode is fragile and not surfaced as a clear
error from the user's perspective.

## Our workaround (sglang downstream)

In our research fork's `layers/attn_res.py` (overlay code that lives
above sglang's base layers), we replaced the three einsums with manual
`(weights.unsqueeze(-1) * V).sum(dim=0)`. This bypasses
`cublasGemmStridedBatchedEx` entirely. bf16 throughput is unchanged
(44.6 → 44.6 tok/s on Kimi-Linear AttnRes); fp8 weight-only path goes
from crashing to working (38.9 tok/s coherent 8/8).

## Asking maintainers

- Is this a known cuBLAS limitation on Blackwell consumer cards?
- Should there be a documented constraint on fp8-dequant-output strides
  before they can feed a strided-batched bf16 GEMM?
- A clearer error from cuBLAS (or a fallback in `torch.einsum`) when
  this pattern is detected would unblock other users hitting the same
  combination.
```

## Follow-up sglang PR (after #5 lands)

```
[layers/attn_res] Replace 3 block-aggregation einsums with manual
broadcast+sum to bypass cuBLAS strided batched bf16 GEMM failure
under fp8 dequant
```

Body: see [PR.md](PR.md) "Summary" / "Symptom" / "Patch" sections. The
fork commit `a6c46168a` already contains the patch (3 call sites in
`block_attn_res`, `block_attn_res_phase1`, per-query fallback). Cherry-
pickable as a standalone commit once `python/sglang/srt/layers/attn_res.py`
exists upstream.

## Cross-link

- **PR #5** (AttnRes inference overlay) — file dependency
- **PR #7** (KDA causal_conv1d fp16) — same `a6c46168a` bundle commit
- **PR #8** (fp8 MoE Blackwell shmem) — same `a6c46168a` bundle commit
- The cuBLAS issue itself is independent of PR #5 timing
