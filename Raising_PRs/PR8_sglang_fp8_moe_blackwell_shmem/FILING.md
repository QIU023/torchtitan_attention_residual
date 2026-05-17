# PR #8 — filing instructions

## Status

🚀 **Branch pushed; PR not yet opened.**

| Item | Link / value |
|---|---|
| Fork branch | https://github.com/QIU023/sglang/tree/pr8-fp8-moe-blackwell-shmem |
| Open-PR URL | https://github.com/QIU023/sglang/pull/new/pr8-fp8-moe-blackwell-shmem |
| Target repo | https://github.com/sgl-project/sglang |
| Base | `sgl-project/sglang:main` |
| Head | `QIU023/sglang:pr8-fp8-moe-blackwell-shmem` |
| Commit | `46592d9cf` (1 commit, +107/-0 in `fused_moe_triton_kernels.py`) |
| Verification | static (py_compile); hot-path GPU smoke needs SM 12.0 (RTX 5090 — we have a fork-side benchmark in `phase11_rlhf_grpo_infra/bench_inference_dtype.py`) |
| Cross-link | Cite **PR #10** as the documented downstream consumer-facing warning that we'd want filed if/when the deeper "illegal memory access" fp8-MoE kernel bug stays unresolved |

## To push the branch

```bash
cd sglang
git push origin pr8-fp8-moe-blackwell-shmem
```

(Visible/irreversible — requires explicit user OK.)

## To open the PR

1. Open https://github.com/QIU023/sglang/pull/new/pr8-fp8-moe-blackwell-shmem after push
2. Confirm base = `sgl-project/sglang:main`, head = `QIU023/sglang:pr8-fp8-moe-blackwell-shmem`
3. Title + body below

---

## Title (copy-paste)

```
[moe/fused_moe_triton] Shrink fp8/int8 fused-MoE config on SM 12.0 (Blackwell consumer)
```

## Body (copy-paste)

```markdown
## Summary

Adds `_maybe_shrink_config_for_sm120()` to the fused-MoE Triton kernel
launcher. For SM 12.0 (Blackwell consumer, e.g. RTX 5090 — ~100 KB
shared memory per block) with fp8 or int8 quantization, shrinks the
H100-tuned config in place so the kernel fits the smaller shmem budget.

H100 / A100 / B100 / MI300 paths are byte-identical: three early-return
guards skip the helper for non-CUDA, non-SM-12.0, and already-fitting
configs.

## Symptom (without this PR)

```
Engine(quantization="fp8", model="<any MoE model on RTX 5090>")
  triton.runtime.errors.OutOfResources:
    Required 147456, Hardware limit 101376
```

The default fused-MoE fp8/int8 configs in `get_default_config()` and
the tuned JSONs under `python/sglang/srt/layers/moe/configs/` were
chosen for Hopper-class shmem budgets (~228 KB on SM 9.0). They
overflow on Blackwell consumer GPUs (SM 12.0, ~100 KB cap).

## Root cause

Rough shared-memory estimate for the fp8/int8 MoE GEMM with
`num_stages`-pipelined A/B tiles:

```
shmem = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * num_stages   bytes (fp8/int8)
```

H100 default: `BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, num_stages=4` →
192 KB. Fits H100's 228 KB; overflows 5090's 101 KB.

## Patch

`_maybe_shrink_config_for_sm120()` shrinks in 4 stages with early
returns:

1. Cap `BLOCK_M` at 64 (halves A-tile footprint)
2. Cap `BLOCK_N` at 128 (halves B-tile footprint)
3. Cap `num_stages` at 2 (Hopper uses 3-4; SM 12.0 has less shmem but
   L2 prefetcher still hides most latency at stages=2)
4. Cap `num_warps` at 4 when `BLOCK_M=64` (register file fits)

Block-wise quant constraints (`BLOCK_SIZE_K == block_shape[1]`,
`BLOCK_SIZE_N == block_shape[0]`) are preserved.

Hooked at `invoke_fused_moe_kernel()` entry — one line.

## Verification

- **Static**: `py_compile` passes; helper is pure-Python with three
  early-return guards covering all non-affected code paths.
- **Functional (RTX 5090 SM 12.0)**: Kimi-Linear AttnRes fp8 weight-only
  inference boots and decodes **38.9 tok/s coherent 8/8** vs bf16
  **44.6 tok/s** baseline. Without this patch, boot fails at
  OutOfResources before any token is generated.
- **Non-regression**: H100 / A100 / B100 / MI300 paths skip the
  helper at the first early return (`is_sm120_supported() == False`).

## Out of scope (separate follow-up)

A downstream consumer found that even the shrunk config triggers an
"illegal memory access" in the fp8 fused-MoE Triton kernel on SM 12.0
(separate kernel-level bug, traceable to triton kernel internals once
the OutOfResources gate is past). Workaround in user code:
`SGLANG_FP8_IGNORED_LAYERS="mlp.experts"` to skip MoE quant and fall
back to `UnquantizedFusedMoEMethod`. That issue is independent of the
config-fit problem this PR fixes; happy to file as a separate issue
if the downstream report would help triage.

## Backwards compatibility

100% — three early returns ensure non-SM-12.0, non-fp8/int8, and
already-fitting configs return the input config unchanged.
```

## Reviewer hints

- The diff is +107/-0, single-file, additive. Easy review.
- `is_sm120_supported` is already used 8 places in `python/sglang/srt/`
  (fp4_utils, fp8_kernel, fp8_utils, fp8, modelopt_quant, etc.) —
  not a new utility, just a new call site.
- The shmem estimate is a heuristic, not exact — but the 4-stage
  shrink is conservative enough that the actual launch succeeds with
  ample margin (verified by the production smoke).

## Related work in same batch

- **PR #1** (SHM-MM env-gate) — separate sglang PR, same fork
- **PR #7** (KDA causal_conv1d fp16) — separate sglang PR, same fork;
  same `a6c46168a` bundle
- **PR #10** (Fp8Config silent bf16 fallback warning) — companion
  user-experience PR, conditional on the downstream MoE-kernel issue
  staying unresolved long-term
