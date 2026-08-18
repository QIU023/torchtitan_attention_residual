# pytorch/pytorch PR: `dtype` on `get_total_norm`

Companion to `ISSUE_pytorch.md` -- that one opens the design question, this one is the
change. File the issue first and reference it; janeyx99 said "issue/PR", and the issue is
where the naming decision gets made.

**Target**: `pytorch/pytorch`, `torch/nn/utils/clip_grad.py` (one function)
**Tag**: @janeyx99
**Patch**: `get_total_norm_dtype_pytorch.patch` -- against `pytorch/main`, `git apply
--check` clean, `py_compile` clean
**Test/probe**: `probe_get_total_norm_dtype.py` -- runs on stock or patched torch
**Risk**: none at `dtype=None`, which is every existing call.
**Do NOT file without a human's go-ahead.**

## Read off pytorch/main before drafting

| claim | source |
|---|---|
| public symbol is an alias | `torch/nn/utils/__init__.py`: `_get_total_norm as get_total_norm`; `clip_grad.py`'s `__all__` has only the three `clip_grad*` names |
| three norm call sites | `_get_total_norm` body: `torch._foreach_norm`, the `vector_norm` list comprehension, the norm-of-norms over `torch.stack` |
| `clip_grad_norm_` also scales | it calls `_clip_grads_with_norm_`, whose body is `torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))` |
| `vector_norm` already has the argument, with this meaning | its docstring: "If specified :attr:`x` is cast to :attr:`dtype` prior to doing the accumulation" |
| **DTensor already takes the non-foreach branch** | `_foreach_supported_types = [torch.Tensor]` and `_has_foreach_support` tests `type(t) in ...` -- an exact-type test, so a subclass never matches |
| **`torch._foreach_norm` accepts an explicit `dtype=None`** | measured on torch 2.8: `omitted -> bfloat16`, `dtype=None -> bfloat16`, `dtype=torch.float32 -> float32`. This was the open question; the unconditional passthrough is fine |

The DTensor row is why the downstream cleanup is a deletion and not a rewrite: torchtitan's
private copy hard-codes `any(isinstance(t, DTensor) ...)` to force the per-tensor path, and
upstream sends DTensors down that same path already. Same behaviour, so the copy has nothing
in it but the dtype.

## Title

    get_total_norm: add a dtype argument for the accumulation

## Body

--- PASTE BEGIN ---

`get_total_norm` accumulates in the input tensors' dtype, so with bf16 tensors it depends on
how the tensors were grouped and not only on their values: each group's norm is rounded
before the groups are combined. 394 bf16 tensors of 64 elements, seed 0, one grouping per
row, partials combined in float64:

    float64 reference     158.057787
    one group             158.000000     0.037% off
    split 100/294         157.896327     0.102%
    split 200/194         158.041925     0.010%
    split 300/94          157.648343     0.259%

Spread across groupings 2.49e-03 relative. Under pipeline or expert parallelism the grouping
is where the model was cut, so the same gradients report a different total norm on different
layouts.

This adds `dtype`, passed to the three norm calls the function already makes.
`torch.linalg.vector_norm` and `torch._foreach_norm` both accept it already, so nothing
below this function changes. With `dtype=torch.float32` the same four groupings agree to
1.00e-07 relative. `dtype=None` is current behaviour, including the empty-input return.

DTensor inputs need no special case: `_foreach_supported_types` is matched by exact type, so
they already take the `vector_norm` branch, and passing `dtype` there preserves the
`_NormPartial` placement rather than materializing anything.

Deliberately not added to `clip_grad_norm_`, which also scales the gradients through
`_clip_grads_with_norm_` -- there "which ops use this dtype" would be a real question. A
caller wanting an fp32 norm composes `get_total_norm(..., dtype=torch.float32)` with
`clip_grads_with_norm_(...)`.

Fixes #NNNNN.

--- PASTE END ---

## The change

`get_total_norm_dtype_pytorch.patch`, +14/-3 across one function: the keyword, a docstring
entry, and four passthroughs (`torch.tensor(0.0)`, the foreach branch, the per-tensor
branch, the norm-of-norms). The per-tensor branch reflows to four lines because the one-line
form goes past 88 characters with the kwarg.

`torch.tensor(0.0, dtype=None)` is float32, so the empty case is unchanged at the default.
The norm-of-norms would inherit `dtype` from its inputs anyway; passing it makes the return
dtype a guarantee rather than a consequence.

Verified against a clean copy of `main`'s file: `git apply --check` clean, result
byte-identical to the intended file, `py_compile` clean.

## The test, and two things it took to get right

`probe_get_total_norm_dtype.py`. Run it bare against a stock torch, or point `--module` at a
patched `clip_grad.py` to exercise the change without touching an install:

    python probe_get_total_norm_dtype.py
    python probe_get_total_norm_dtype.py --module /path/to/patched/clip_grad.py

Measured on torch 2.8.0+cpu with the equivalent edits applied (2.8 spells the annotation
`Optional[bool]`, so the main-targeted patch does not apply to it verbatim):

| accumulation | spread across the four groupings |
|---|---|
| bfloat16 (today) | 2.490e-03 |
| float32 (patched) | 1.004e-07 |

24811x tighter, and every assertion passes: returned dtype, default unchanged, empty case
with and without `dtype`, and `foreach=False` agreeing with the foreach branch.

**The partials must be combined in higher precision than the per-group norm, or there is
nothing to see.** The first version of this probe stacked the bf16 partials and normed them
in bf16; at magnitude ~158 the bf16 grid spacing is 2.0, so all four groupings snapped to
158.000000 and the table read as if nothing was wrong. That is not what PP or EP does -- the
groups are ranks and the cross-rank combine is a separate step -- but it is an easy way to
write a probe that refutes a real defect.

**The patched assertion is a tolerance, not equality.** fp32 grouping differences shrink,
they do not vanish: each group's norm is still rounded, at 2^-23 instead of 2^-8. A bitwise
assertion would fail on a correct patch. The probe asserts the spread drops below 1e-6
relative and prints both numbers so the claim is checkable.

## Still to confirm at filing time

* **Where the test goes.** `get_total_norm`'s existing coverage in the pytorch test suite
  was not located; find it rather than guessing a file. The probe is our harness, not a
  drop-in for theirs.
* Whether a non-floating-point `dtype` should raise here or be left to `vector_norm`'s own
  error.

## What this does to the torchtitan diff

Today: `_get_total_norm_fp32`, +54/-3, plus three call-site replacements.
After: the helper is deleted and the three call sites gain one kwarg.

    -   total_norm = _get_total_norm_fp32(
    -       grads, norm_type, error_if_nonfinite, foreach
    +   total_norm = torch.nn.utils.get_total_norm(
    +       grads, norm_type, error_if_nonfinite, foreach, dtype=torch.float32
        )

x3 -- one in `clip_grad_norm_`, two in `_clip_grad_norm_with_ep`.

**Do not make the torchtitan PR wait for this.** They are the same fix but not the same
change: the torchtitan one is a correctness bug that exists today and is already under
review, while this one is an API addition that has to survive a naming discussion. Coupling
them means the grad norm stays partition-dependent for however long the API takes.

The sequence that costs nothing either way:

1. torchtitan lands the private helper (it is self-contained and version-independent);
2. this lands in pytorch;
3. a three-line torchtitan follow-up deletes the helper.

Step 3's gate is torchtitan's minimum torch, not this PR's merge -- but torchtitan's
from-source install already requires a PyTorch nightly (README: "This method requires the
nightly build of PyTorch"), so that is about one nightly after step 2, not a release cycle.

The one case for the other order is a torchtitan maintainer preferring not to carry a copy
of a torch function at all. That is a real position and it is theirs to take, so the
question belongs on the torchtitan thread rather than being decided here. The PR body
already offers it ("happy to propose that upstream, at which point this helper reduces to a
one-line call"), which is what invited janeyx99's comment in the first place.

## Local test environment

`C:\Users\78532\.venvs\torch2` -- python 3.9, torch 2.8.0+cpu, CPU only. Separate from the
base conda env on purpose: that one has torch 1.9 with `torchvision` 0.10 and `torchtext`
0.10 pinned to it, and the disk is at 95%. 2.8 is the newest wheel for python 3.9; the GPU
box runs 2.14-dev, where the main-targeted patch should apply directly.
