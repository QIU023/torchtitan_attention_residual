# PR #25 — `torch._grouped_mm` rejects every valid layout for an operand whose contraction dim is 0 (breaks compiled MoE backward on an empty expert group)

**Target**: `pytorch/pytorch`, `aten/src/ATen/native/GroupedMMUtils.h`
(`check_valid_strides_and_return_transposed`), plus the duplicated logic in
`torch/_meta_registrations.py`.
**Found on**: `2.14.0.dev20260802+cu130`, CUDA 13.0, sm_120.
**Status**: reproduction is airtight and self-contained. **The C++ patch is
NOT compile-verified** -- see "What is and is not verified".
**Risk**: low. The change only adds an early return on a path that currently
raises unconditionally; no tensor with elements takes it.

---

## Suggested PR title

> [grouped_mm] accept an operand with a zero-length contraction dim instead of rejecting every possible stride

## Suggested PR body

### Summary

`torch._grouped_mm` cannot be called with a 2D operand whose inner dimension
is 0. Not "it is slow" or "it returns garbage" -- there is **no stride vector
that passes validation**, including the one `torch.empty` itself produces for
that shape.

This is reachable from ordinary MoE training: in the weight-gradient form of
a grouped GEMM the operand's contraction length is the total routed token
count, so it is zero exactly when no expert in the call received any tokens
(a single empty group among busy ones does not produce it -- see "Empty
groups are already fine" below). Eager autograd happens not to emit the call
in that case; inductor emits it unconditionally, so the failure shows up as
"this model trains fine eagerly and dies under `torch.compile`".

### Reproduction

Five lines, no model, no `torch.compile`, no distributed:

```python
import torch
F, E, D = 224, 4, 256
a    = torch.empty(F, 0, device="cuda", dtype=torch.bfloat16)   # contraction dim = 0
w    = torch.randn(E, 0, D, device="cuda", dtype=torch.bfloat16)
offs = torch.zeros(E, device="cuda", dtype=torch.int32)
torch._grouped_mm(a, w, offs=offs)
# RuntimeError: strides should be multiple of 16 bytes
```

and with the natural row-major stride for a zero-column matrix:

```python
base = torch.empty(0, device="cuda", dtype=torch.bfloat16)
torch._grouped_mm(base.as_strided((F, 0), (0, 1)), w, offs=offs)
# RuntimeError: Invalid strides/sizes, got [0, 1] for strides and [224, 0] for sizes
```

### The check is unsatisfiable for this shape

`check_valid_strides_and_return_transposed` (`GroupedMMUtils.h:25`) accepts a
matrix on one of two branches. With `sizes = [224, 0]`, `end_dim = 1`, and
`alignment = 16 / 2 = 8` for bf16:

| stride | branch 1 `s[0]==1 && s[1] >= max(1, size[0]=224)` | branch 2 `s[1]==1 && s[0] >= max(1, size[1]=0)` | result |
|---|---|---|---|
| `(0, 1)` natural row-major | `s[0]!=1`, no | `0 >= 1`, no | `Invalid strides/sizes` |
| `(1, 1)` what `torch.empty(224,0)` gives | `1 >= 224`, no | yes -> then `1 % 8 == 0`, no | `strides should be multiple of 16 bytes` |
| `(8, 1)` fabricated | -- | yes -> `8 % 8 == 0`, yes | **ok** |

Branch 2 demands `stride[end_dim - 1] >= 1` *and* a multiple of the alignment,
while the row stride of a 0-column matrix is naturally 0. The `max(1, ...)`
guards anticipate a degenerate *size* but not a tensor with no elements at all.

### The kernel itself is fine

Handing it a fabricated but aligned stride shows the validation is the only
obstacle:

```python
torch._grouped_mm(base.as_strided((F, 0), (8, 1)), w, offs=offs)
# ok, output shape [224, 256]
```

Correct shape, correct (zero) result. So this is a validation bug, not a
missing kernel capability.

### Empty *groups* are already fine

Worth separating, because "grouped_mm breaks on empty experts" would be too
broad a claim. Empty groups work today, eager and compiled:

| group sizes | total rows | eager | compiled |
|---|---|---|---|
| `[128,128,128,128]` | 512 | ok | ok |
| `[256,256,0,0]` | 512 | ok | ok |
| `[512,0,0,0]` | 512 | ok | ok |
| `[0,256,0,256]` | 512 | ok | ok |
| `[0,0,0,0]` | 0 | ok | ok |

Only the zero-length *contraction* dimension fails, and that shape arises in
backward rather than forward.

### Suggested fix

An operand with no elements has no memory to align and no layout to validate --
every stride vector describes the same empty tensor. Return early, inferring
the orientation from whichever stride is unit:

```cpp
inline bool check_valid_strides_and_return_transposed(const Tensor& mat) {
  IntArrayRef tensor_strides = mat.strides();
  IntArrayRef tensor_sizes = mat.sizes();
  int end_dim = mat.dim() - 1;
  int alignment = 16 / mat.element_size();
  bool is_cpu = mat.device().is_cpu();

+ // An operand with no elements has nothing to align and no layout to check:
+ // every stride vector describes the same tensor. Reachable from the backward
+ // of a grouped GEMM whose call has no routed tokens at all: the weight
+ // gradient contracts over the total token count. Infer the orientation from
+ // whichever stride is unit.
+ if (mat.numel() == 0) {
+   return tensor_strides[end_dim] != 1 && tensor_strides[end_dim - 1] == 1;
+ }
+
  TORCH_CHECK(is_cpu || uint64_t(mat.data_ptr()) % 16 == 0, ...);
```

Placing it before the `data_ptr()` check also avoids asserting alignment on a
pointer that may be null for an empty allocation.

The returned bool keeps today's convention -- branch 1 (`stride[end_dim-1]==1`)
means transposed, branch 2 (`stride[end_dim]==1`) means not -- so `(0,1)` and
`(1,1)` return `false` (row-major) and `(1,0)` returns `true`.

### The same logic is duplicated in the meta registration

`torch/_meta_registrations.py:8470-8488` reimplements the identical two-branch
check and needs the matching guard, otherwise the meta function and the kernel
disagree about what is callable.

**One question for a maintainer there.** The failing shape is genuinely
data-dependent (a routed token count), so under dynamic shapes `mat.numel()`
may be a `SymInt` and `== 0` would install a guard. Should that use
`guard_size_oblivious`, or is a guard acceptable here given the shape is
already specialized by the time inductor emits the call? Not asserting an
answer -- this is the part where local reproduction cannot settle it.

### Test plan

* `test_grouped_mm` case: for each of `(0,1)`, `(1,1)` and the contiguous
  layout, `torch._grouped_mm` on a `[M, 0]` operand returns zeros of shape
  `[M, N]` rather than raising.
* Backward case: a grouped GEMM where one group is empty, `.backward()` on a
  contiguous gradient, under `torch.compile`. Fails before, passes after.

### What is and is not verified

* **Verified locally**: the reproduction, the branch analysis, that a
  fabricated aligned stride makes the same call succeed with the correct
  output shape, and that empty groups are otherwise fine in both modes.
* **Not verified**: the C++ patch itself. This machine has PyTorch from a pip
  wheel and no build tree, so the diff is written against the installed
  header's source and reviewed by hand, not compiled or run. Say so in the PR
  rather than implying otherwise.
* End-to-end evidence that this single check is the whole obstacle: a local
  shim that re-strides exactly as this patch would
  (`phase13_k3like_48b_posttrain/matrix_scripts/gmm_shim.py` in the public
  logbook) takes a compiled 18-cell parallelism matrix from 15/18 to 18/18
  (`SEED_MATRIX_2026-08-04.md`, compiled section); the full analysis is
  `GROUPED_MM_EMPTY_GROUP_2026-08-04.md`.

### Provenance

One sentence, no more: found while bringing up `torch.compile` across a
parallelism matrix for a Kimi-K3-family MoE model; three EP configurations
failed under compile and passed eagerly, and the cause reduced to the five
lines above. Public logbook:
`phase13_k3like_48b_posttrain/GROUPED_MM_EMPTY_GROUP_2026-08-04.md`.

---

## Notes for the filer

- Lead with the five-line reproduction. It needs no model, no distributed, no
  compile, and it is the whole argument.
- Do **not** claim the C++ patch is tested. It is not. Offer to iterate on CI.
- Keep "empty groups already work" in the body. Without it a reviewer will
  reasonably assume the report is about zero-token experts generally, which is
  a different and already-working thing.
- An earlier version of our own logbook blamed our routing code for this and
  was retracted. Do not repeat the claim that the operator is "intolerant of
  degenerate shapes" -- it is intolerant of exactly one shape.

---

## Branch state (2026-08-07)

Fix branch pushed: `QIU023/pytorch` branch `fix_grouped_mm_empty_contraction`
(commit `6f971e8`, based on upstream main `6a34faa`), applied by hand -- the
hand-written patch in this folder did not `git apply` and is kept only as the
design record. `torch/_meta_registrations.py` passes `py_compile`; the C++
side remains CI-verified-only, as the body states. Open the PR at:
`https://github.com/pytorch/pytorch/compare/main...QIU023:pytorch:fix_grouped_mm_empty_contraction`
The local `pytorch/` submodule is a shallow clone carrying this branch; it
cannot serve as the running torch (both venvs install wheels) -- its role is
the PR branch plus CI.
