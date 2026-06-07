## Motivation

`_causal_conv1d_fwd_kernel` and `_causal_conv1d_update_kernel` in
`sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` fail to JIT-compile when
the model is launched with `--dtype float16` for any KDA / Mamba-conv-using model:

```
triton.compiler.errors.CompilationError: at line ...:
    AssertionError("Mismatched type for col0 between then block
                    (<['256'], bf16>) and else block (<['256'], fp16>)")
```

Root cause: the kernels read prior tokens from the `conv_states` cache and the cache's
element dtype is **independent of the model dtype**: SGLang stores it as
`bfloat16` by default (`SGLANG_MAMBA_CONV_DTYPE`). Inside the kernel:

- The `HAS_INITIAL_STATES → load_init_state == True` branch loads `col*` from
  `conv_states` → values carry the cache dtype (`bf16`).
- The `else` branch initializes `col*` via `tl.zeros(..., dtype=x_ptr.dtype.element_ty)`
  → values carry the model dtype (`fp16`).

Triton's SSA join over the two `if/else` branches requires identical element types;
when the model dtype differs from the cache dtype the union fails at compile time
and the engine never reaches the launch site, hard-crashing `--dtype float16`
inference for the whole Kimi-Linear / hybrid-linear-attention family.

The bf16-default training/inference path is unaffected (both `x` and `conv_states`
are bf16, so the type-join is trivial).

## Modifications

Introduce a `col_dtype: tl.constexpr` taken from the model dtype and cast both
branches to it. The unification target is `x_ptr.dtype.element_ty` (the model
dtype) — not the cache dtype — because the same `col*` variables are later
overwritten with values loaded from `x` further down the kernel
(e.g. `col0 = matrix_x`), which already carry the model dtype. Standardizing on
the model dtype avoids a second downstream join.

Concretely, in both `_causal_conv1d_fwd_kernel` and
`_causal_conv1d_update_kernel`:

```python
col_dtype: tl.constexpr = x_ptr.dtype.element_ty
if load_init_state:
    # load from conv_states (cast to x dtype to keep col* uniform)
    if KERNEL_WIDTH == 2:
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(col_dtype)
    if KERNEL_WIDTH == 3:
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(col_dtype)
        ...
else:
    # prior-tokens are zeros
    if KERNEL_WIDTH >= 2:
        col0 = tl.zeros((BLOCK_N,), dtype=col_dtype)
    if KERNEL_WIDTH >= 3:
        col1 = tl.zeros((BLOCK_N,), dtype=col_dtype)
    ...
```

`+38 / -25`, one file. The bf16+bf16 default path is **byte-identical**: `.to(bf16)`
of a bf16 load is a no-op, and `tl.zeros(..., dtype=bf16)` is exactly what the
unpatched else-branch produced.

| | bf16 model (default) | fp16 model + bf16 conv_states (this PR fixes) |
|---|---|---|
| `_causal_conv1d_fwd_kernel` | byte-identical | compiles + correct output |
| `_causal_conv1d_update_kernel` | byte-identical | compiles + correct output |
| Default Triton compile | passes (today) | passes (this PR) |

## Accuracy Tests

`KERNEL_WIDTH=4` matches Kimi-Linear KDA's `short_conv_kernel_size=4`
([Kimi-Linear-48B-A3B-Instruct config.json](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json)),
so the smoke exercises the exact production code path.

**Direct kernel smoke matrix** (no SGLang Engine boot needed — runs both the
prefill `_causal_conv1d_fwd_kernel` and the decode `_causal_conv1d_update_kernel`
directly with synthetic inputs), verified on RTX 4070Ti (SM 8.9) with
torch 2.11.0+cu130, triton 3.6.0:

```text
Prefill (_causal_conv1d_fwd_kernel):
  baseline_bf16_bf16             PASS
  bug_repro_fp16_x_bf16_state    PASS   ← production scenario (this PR fixes)
  inverted_bf16_x_fp16_state     FAIL   ← see "Follow-up" below
  all_fp16                       PASS

Decode (_causal_conv1d_update_kernel):
  baseline_bf16_bf16             PASS
  bug_repro_fp16_x_bf16_state    PASS   ← production scenario (this PR fixes)
  inverted_bf16_x_fp16_state     PASS   ← decode path is symmetrically clean
  all_fp16                       PASS
```

Smoke scripts:
[`smoke_kernel_direct_fp16.py`](https://github.com/QIU023/torchtitan_attention_residual/blob/main/Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_direct_fp16.py)
(prefill) +
[`smoke_kernel_decode_fp16.py`](https://github.com/QIU023/torchtitan_attention_residual/blob/main/Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_decode_fp16.py)
(decode). Both are GPU-required but zero-network: they import only the patched
kernel module and feed synthetic 64×8 batches.

**End-to-end research-fork verification** (Kimi-Linear AttnRes inference under
`--dtype float16`): coherent 8/8 on the smoke prompt set at **44.5 tok/s** vs
the bf16 baseline's **44.6 tok/s**; documented in our research repo's Phase 11
benchmark output.

## Speed Tests and Profiling

Kernel-internal change with no extra ops on the bf16 path → no measurable
throughput change on the default bf16+bf16 path (verified end-to-end:
44.6 → 44.6 tok/s on Kimi-Linear AttnRes 1.4B-active).

The fp16+bf16 path goes from **not compiling** (hard crash at first KDA layer)
to **44.5 tok/s** on the same model — within 0.2% of the bf16 baseline.

### Interaction with `fp8` weight-only quantization

`causal_conv1d` consumes activations only; SGLang's `--quantization fp8` is
weight-only and leaves activations at the requested model dtype:

- `--quantization fp8 --dtype bfloat16` → kernel sees `x=bf16`, `conv_states=bf16`,
  baseline path, unaffected by this PR.
- `--quantization fp8 --dtype float16` → kernel sees `x=fp16`, `conv_states=bf16`,
  exactly the bug this PR fixes; covered by the same smoke.

So fp8 inference picks the fix up for free.

### Follow-up (out of scope for this PR)

The prefill kernel has a structurally analogous SSA type-join in the **write-back**
path (`tl.store(conv_states_ptrs_target, new_conv_state, mask)` where
`new_conv_state` is sourced from `tl.load(x_ptrs, ...)` in the
`state_len <= seqlen` branch but from `tl.where(mask, conv_state, loaded_x)` in
the `load_init_state` branch). Triggering it requires the inverse-dtype
configuration (`x.dtype=bf16` + `conv_states.dtype=fp16`), which SGLang's defaults
(`SGLANG_MAMBA_CONV_DTYPE=bfloat16` regardless of model dtype) do not produce in
current code. Keeping this PR scoped to the production-hit site; a symmetric fix
for write-back is a clean follow-up if anyone ever wants to override the cache
dtype.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). *(Verified locally against sglang's pinned hook versions on the patched file: `isort 7.0.0`, `black 26.1.0`, `ruff 0.15.1` with sglang's `--select=F401,F821`, `codespell 2.4.1` — all clean.)*
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). *(No CI-registered test in this PR — the smoke scripts linked above exercise the exact failure mode and are runnable in isolation. Happy to add a registered test under `test/registered/unit/` if maintainers prefer.)*
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). *(Direct kernel smoke matrix above; end-to-end bf16 vs fp16 throughput parity 44.6 ≈ 44.5 tok/s.)*
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
