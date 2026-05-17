# PR #7 — KDA `causal_conv1d_triton` fp16 dtype type-join fix

**Target repo**: `sgl-project/sglang`
**Target file**: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
**Fork reference**: commit `a6c46168a` on `attention_residual_inference` branch (verified fp16 path 44.5 tok/s coherent 8/8 vs bf16 baseline 44.6).
**Effort**: ~1 hour (one Triton kernel branch + one regression test).
**Risk**: low — single-kernel dtype unification; bf16 path is regression-clean (fork-verified).

---

## Suggested PR title

> [mamba/causal_conv1d] Fix `_causal_conv1d_fwd_kernel` fp16 dtype type-join error on `HAS_INITIAL_STATES` branch

---

## Suggested PR body

### Summary

Fixes a Triton compilation failure in `_causal_conv1d_fwd_kernel` when
the model is loaded with `dtype="float16"`. The kernel has a branch
(`if HAS_INITIAL_STATES: if load_init_state: ...`) where one side
loads from a buffer that was promoted to bf16 elsewhere in the
prologue, while the other side carries the user's fp16 model dtype.
The Triton `if/else` join then fails to merge the two branch types.

After this fix, fp16 inference works for the whole Kimi-Linear /
hybrid-linear-attention family.

### Symptom

Booting an SGLang Engine with `dtype="float16"` for any KDA-using
model (Kimi-Linear and friends — `moonshotai/Kimi-Linear-*`) crashes
at first KDA forward with:

```
triton.compiler.errors.CompilationError: at 105:8:
AssertionError("Mismatched type for col0 between then block
                (<['256'], bf16>) and else block (<['256'], fp16>)")
```

### Root cause

In `_causal_conv1d_fwd_kernel` the `HAS_INITIAL_STATES` branch loads
the kernel-prologue's residual buffer (cast to bf16 a few lines
earlier for a different code path) — the other branch carries the
user-requested fp16 dtype. The `if/else` join can't merge.

### Fix sketch

Pick one of:

1. **Common-cast before join**: cast both branches to a unified dtype
   immediately before the merge point. Simplest, no semantic change.
2. **`tl.where` rewrite**: replace the two-branch `if/else` with a
   single `tl.where(load_init_state, init_load, zero_init)`. Eliminates
   the join issue structurally. Slightly more invasive but cleaner.

Either works; option 1 is the minimal patch.

### Test plan

Add a regression test under
`python/sglang/test/srt/layers/attention/test_causal_conv1d_dtype.py`:

```python
import pytest, torch
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fwd_triton,
)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_causal_conv1d_dtype_round_trip(dtype):
    B, T, D = 2, 64, 256
    x = torch.randn(B, T, D, dtype=dtype, device="cuda")
    w = torch.randn(D, 4, dtype=dtype, device="cuda")
    init_state = torch.randn(B, D, 3, dtype=dtype, device="cuda")
    y = causal_conv1d_fwd_triton(x, w, initial_states=init_state)
    assert y.dtype == dtype
    assert not torch.isnan(y).any()
```

(Currently `fp16` parametrization fails at Triton compile; bf16 / fp32
pass. After patch, all three pass.)

### Why upstream

Today, fp16 inference is silently broken for the whole Kimi-Linear /
hybrid-linear-attention model family. Doesn't matter at training
(everyone trains in bf16) but appears whenever someone tries fp16
inference for memory / throughput on cards where fp16 is preferable
to bf16 (or where the deployment target only supports fp16).

Affected models include but are not limited to:

- `moonshotai/Kimi-Linear-48B-A3B-Base` (and its derivatives in the
  fork's `experiments/kimi_linear/`)
- Any future `KimiDeltaAttention`-using carrier

### Fork verification

Verified in our research fork (`QIU023/sglang@attention_residual_inference`
commit `a6c46168a`):

- fp16 path: **44.5 tok/s, coherent 8/8** on the smoke prompt set.
- bf16 baseline (unchanged): **44.6 tok/s**.
- fp32: unchanged.

No regression observed on existing bf16 / fp32 paths.

### Direct-kernel smoke (RTX 4070Ti, SM 8.9, torch 2.11.0+cu130, triton 3.6.0)

To isolate the patch from full-Engine boot infrastructure (which on
some SM tiers depends on `sgl_kernel` wheels not always shipped for
that arch), a kernel-level smoke covers both prefill
(`_causal_conv1d_fwd_kernel`) and decode (`_causal_conv1d_update_kernel`)
across the dtype matrix that real Kimi-Linear/KDA inference exercises.
KERNEL_WIDTH = 4 = `short_conv_kernel_size` from the Kimi-Linear
HF config.

```text
Prefill (_causal_conv1d_fwd_kernel):
  baseline_bf16_bf16             PASS
  bug_repro_fp16_x_bf16_state    PASS   ← production scenario (this PR)
  inverted_bf16_x_fp16_state     FAIL   ← see "Follow-up" below
  all_fp16                       PASS

Decode (_causal_conv1d_update_kernel):
  baseline_bf16_bf16             PASS
  bug_repro_fp16_x_bf16_state    PASS   ← production scenario (this PR)
  inverted_bf16_x_fp16_state     PASS   ← decode path is symmetrically clean
  all_fp16                       PASS
```

The smoke scripts live in the filing folder:
`Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_{direct,decode}_fp16.py`.

### Interaction with fp8 weight-only quantization

`causal_conv1d` only consumes activations; SGLang's fp8 quant is
weight-only and leaves activations at the requested model dtype.
That means:

- `--quantization fp8 --dtype bfloat16` → kernel sees `x=bf16`,
  `conv_states=bf16` — baseline path, unaffected by this PR.
- `--quantization fp8 --dtype float16` → kernel sees `x=fp16`,
  `conv_states=bf16` — exactly the bug this PR fixes, verified by
  the prefill smoke above.

So fp8 inference picks the fix up for free; no separate fp8 work is
required for this kernel.

### Follow-up (out of scope for this PR)

The prefill kernel has a structurally analogous SSA type-join in the
write-back path (lines around `tl.store(conv_states_ptrs_target,
new_conv_state, mask)`) where `new_conv_state` is sourced from
`tl.load(x_ptrs, ...)` in the `state_len <= seqlen` branch but from
`tl.where(mask, conv_state, loaded_x)` in the `load_init_state`
branch. Triggering it requires the inverse-dtype configuration
(`x.dtype=bf16` + `conv_states.dtype=fp16`), which SGLang's defaults
(`SGLANG_MAMBA_CONV_DTYPE=bfloat16` regardless of model dtype) do
not produce in current code. Keeping this PR scoped to the
production-hit site; a symmetric fix for write-back is a clean
follow-up if anyone ever wants to override the cache dtype.

### Reference downstream usage

Hit while bringing up SGLang Engine for AttnRes-Kimi-Linear inference
under varied dtype configs for a quantization sweep. Without this fix,
the fp16 row of the sweep is unreachable.

---

## Filing checklist

- [ ] Fork branch up to date with sglang `main`.
- [ ] Single-commit PR titled per above.
- [ ] Description includes the symptom traceback + root cause.
- [ ] Regression test added covering fp16 / bf16 / fp32.
- [ ] Verify bf16 throughput unchanged via existing benchmarks
      (e.g. `python/sglang/bench/bench_one_batch.py`).
