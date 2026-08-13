# PR #25 — `torch._grouped_mm` rejects every valid layout for an operand whose contraction dim is 0 (breaks compiled MoE backward on an empty expert group)

**Target**: `pytorch/pytorch`, `aten/src/ATen/native/GroupedMMUtils.h` (`check_valid_strides_and_return_transposed`), plus the duplicated logic in `torch/_meta_registrations.py`.
**Found on**: `2.14.0.dev20260802+cu130`, CUDA 13.0, sm_120.
**Status**: reproduction is airtight and self-contained. **The C++ patch is NOT compile-verified** -- stated plainly in the body's closing caveat.
**Risk**: low. The change only adds an early return on a path that currently raises unconditionally; no tensor with elements takes it.

**Format note**: single-line paragraphs (tables, lists and code blocks excepted) so the body copies verbatim.

---

## Suggested PR title

> [grouped_mm] accept an operand with a zero-length contraction dim instead of rejecting every possible stride

## Suggested PR body

--- PASTE BEGIN ---

### Summary

`torch._grouped_mm` cannot be called with a 2D operand whose inner dimension is 0: no stride vector passes `check_valid_strides_and_return_transposed`, including the one `torch.empty` itself produces for that shape.

This is reachable from ordinary MoE training. In the weight-gradient form of a grouped GEMM the operand's contraction length is the total routed token count, so it is zero exactly when no expert in the call received any tokens. Eager autograd happens not to emit the call in that case; inductor emits it unconditionally, so the model trains eagerly and dies under `torch.compile`.

### Reproduction

```python
import torch
F, E, D = 224, 4, 256
a    = torch.empty(F, 0, device="cuda", dtype=torch.bfloat16)   # contraction dim = 0
w    = torch.randn(E, 0, D, device="cuda", dtype=torch.bfloat16)
offs = torch.zeros(E, device="cuda", dtype=torch.int32)
torch._grouped_mm(a, w, offs=offs)
# RuntimeError: strides should be multiple of 16 bytes
```

With `sizes = [224, 0]` and bf16 alignment 8, the two accepting branches in `GroupedMMUtils.h:25` are both unsatisfiable:

| stride | branch 1 `s[0]==1 && s[1] >= max(1, 224)` | branch 2 `s[1]==1 && s[0] >= max(1, 0)` | result |
|---|---|---|---|
| `(0, 1)` natural row-major | no | `0 >= 1`, no | `Invalid strides/sizes` |
| `(1, 1)` what `torch.empty(224,0)` gives | no | yes, then `1 % 8 != 0` | `strides should be multiple of 16 bytes` |
| `(8, 1)` fabricated | -- | yes | **ok, correct [224, 256] zeros** |

The `(8, 1)` row shows the kernel itself is fine -- the validation is the only obstacle. The `max(1, ...)` guards anticipate a degenerate size but not a tensor with no elements: the natural row stride of a 0-column matrix is 0.

Empty *groups* already work in both modes (`[512,0,0,0]`, `[0,0,0,0]`, etc.); only the zero-length contraction dimension fails, and that shape arises in backward.

### Fix

An operand with no elements has no memory to align and no layout to validate, so return early, inferring the orientation from whichever stride is unit (keeping the existing transposed-bool convention). Placed before the `data_ptr()` check, since an empty allocation may not carry an aligned pointer. The duplicated stride check in `meta_grouped_mm` gets the matching guard so the meta function and the kernel agree on what is callable.

One question on the meta side: the failing shape is data-dependent (a routed token count), so under dynamic shapes `mat.numel() == 0` would install a guard. Should that use `guard_size_oblivious`, or is a guard acceptable given the shape is already specialized by the time inductor emits the call?

### Test plan

* `test_grouped_mm`: `(0,1)`, `(1,1)` and contiguous layouts of a `[M, 0]` operand return zeros of shape `[M, N]` rather than raising.
* Backward of a grouped GEMM with all groups empty, under `torch.compile`: fails before, passes after.

Caveat, stated plainly: the C++ change is written against the installed header from a pip wheel and reviewed by hand -- I could not compile it locally, so CI is its first build. The reproduction, the branch analysis, and the fabricated-stride control above are all verified. End to end, a shim that re-strides exactly as this patch would takes a compiled 18-cell MoE parallelism matrix from 15/18 to 18/18 ([shim](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/matrix_scripts/gmm_shim.py), [analysis](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/GROUPED_MM_EMPTY_GROUP_2026-08-04.md)).

--- PASTE END ---

---

## Notes for the filer

- Lead with the five-line reproduction. It needs no model, no distributed, no compile, and it is the whole argument.
- Do **not** claim the C++ patch is tested. It is not. Offer to iterate on CI.
- Keep "empty groups already work" in the body. Without it a reviewer will reasonably assume the report is about zero-token experts generally, which is a different and already-working thing.
- An earlier version of our own logbook blamed our routing code for this and was retracted. Do not repeat the claim that the operator is "intolerant of degenerate shapes" -- it is intolerant of exactly one shape.

## Branch state (2026-08-07)

Fix branch pushed: `QIU023/pytorch` branch `fix_grouped_mm_empty_contraction` (commit `f5c3171` after the comment compression, based on upstream main `6a34faa`), applied by hand -- the hand-written patch in this folder did not `git apply` and is kept only as the design record. Inline comments are 1-2 lines; the full WHY lives in the commit message and PR body. `torch/_meta_registrations.py` passes `py_compile`; the C++ side remains CI-verified-only, as the body states. Open the PR at: `https://github.com/pytorch/pytorch/compare/main...QIU023:pytorch:fix_grouped_mm_empty_contraction`

The local `pytorch/` submodule is a shallow clone carrying this branch; it cannot serve as the running torch (both venvs install wheels) -- its role is the PR branch plus CI.
