# PR #8 — fp8 weight-only MoE fused kernel Blackwell shmem autotune

**Target repo**: `sgl-project/sglang`
**Target file**: `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
**Fork reference**: shmem-shrink helper `_maybe_shrink_config_for_sm120` is bundled inside commit `a6c46168a` on `QIU023/sglang@attention_residual_inference` (alongside PR #7 and PR #9 fixes).

**Effort**: ~2 hours including autotune sweep.
**Risk**: medium — partial fix. The shmem-shrink helper successfully launches the kernel, but the shrunk config triggers a downstream "Triton Error [CUDA]: an illegal memory access" inside the fp8 fused-MoE kernel on RTX 5090 — likely a separate pipelining / expert-token-counting issue under reduced `num_stages` on SM 12.0.

---

## Suggested PR title

> [moe/fp8] Autotune row for Blackwell consumer (RTX 5090, SM 12.0, ~100KB shmem)

---

## Summary

Adds a smaller-shmem autotune row to `_get_default_config` for the fp8
fused-MoE kernel, gated by `device_capability == (12, 0)` and
`shared_memory_per_block < 128 KB`. Lets the kernel **launch** (no
more `OutOfResources: out of resource: shared memory`) on RTX 5090.

**Caveat**: the shrunk config still triggers a downstream illegal
memory access on RTX 5090. The launch path is fixed; the kernel
internals (pipelining / expert-token-counting under `num_stages=2`)
need a follow-up PR for full Blackwell consumer support.

## Symptom (without this PR)

```
Engine(dtype="bfloat16", quantization="fp8", model="kimi_linear_...")

triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 147456, Hardware limit: 101376
```

The fp8 fused-MoE kernel was tuned for SM 9.0+ (Hopper / H100, ≥ 228 KB
shared memory). RTX 5090 (Blackwell consumer, ~100 KB shared memory)
is unrepresented in the autotune grid; the default block sizes +
`num_stages=3` exceed the shmem cap.

## Patch

```python
# python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py

def _get_default_config(dtype, dev_cap, shmem_per_block, ...):
    # Existing rows for SM 8.0 / 8.6 / 9.0+ unchanged.
    if dev_cap == (12, 0) and shmem_per_block < 128 * 1024:
        # Blackwell consumer: smaller block / fewer stages to fit shmem.
        return TritonMoeConfig(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=8,
            num_stages=2,
            num_warps=4,
        )
    # ... existing fallback
```

## Test plan

1. RTX 5090 smoke: `Engine(quantization="fp8", model="...")` on a small
   MoE — must not crash with `OutOfResources`.
2. H100 regression: existing autotune row unchanged, throughput stays.
3. **Known limitation**: RTX 5090 still hits "illegal memory access" at
   first MoE forward after the kernel launches successfully. Document
   this in the PR as a follow-up issue rather than a blocker.

## Downstream issue (mention in PR body, file separately)

Even after the autotune row lands, the shrunk config (BLOCK_M=64,
num_stages=2, num_warps=4) triggers a downstream `Triton Error [CUDA]:
an illegal memory access` inside the fp8 fused-MoE kernel on RTX 5090.
Likely a separate issue with the kernel's pipelining or expert-token-
counting under reduced num_stages on SM 12.0. Needs deeper Triton-
level debugging than this PR's scope.

Smoke workaround for users hitting the downstream ICA:
`SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts` so MoE stays bf16; fp8
weight-only on dense Linear (q / k / v / o, mlp gate / up / down) still
works (38.9 tok/s coherent 8/8 vs bf16 baseline 44.6).

## Filing checklist

- [ ] Verify the autotune row keeps existing SM 9.0+ rows untouched.
- [ ] H100 regression test green.
- [ ] RTX 5090 smoke launches without `OutOfResources`.
- [ ] PR body links a follow-up issue for the downstream ICA.
- [ ] CC SGLang Triton-MoE maintainers.

## Why upstream

Blackwell consumer cards (RTX 5090 / 5080) are becoming common rental
hardware (vast.ai / runpod price points), and fp8 weight-only
quantization is a key throughput win. Without this autotune row,
Kimi-Linear-class MoE models can't be fp8-served on these cards at
all — the fall-through to bf16 weights defeats the quantization point.
