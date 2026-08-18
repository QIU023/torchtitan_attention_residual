# pytorch/pytorch PR: `dtype` on `get_total_norm`

Companion to `ISSUE_pytorch.md` -- that one opens the design question, this one is the
change. File the issue first and reference it; janeyx99 said "issue/PR", and the issue is
where the naming decision gets made.

**Target**: `pytorch/pytorch`, `torch/nn/utils/clip_grad.py` (one function)
**Tag**: @janeyx99
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

That last row is why the downstream cleanup is a deletion and not a rewrite: torchtitan's
private copy hard-codes `any(isinstance(t, DTensor) ...)` to force the per-tensor path, and
upstream sends DTensors down that same path already. Same behaviour, so the copy has nothing
in it but the dtype.

## Title

    get_total_norm: add a dtype argument for the accumulation

## Body

--- PASTE BEGIN ---

`get_total_norm` accumulates in the input tensors' dtype, so with bf16 tensors the result
carries three to four significant digits -- and because the rounding happens per group, it
depends on how the tensors were grouped rather than only on their values. 394 synthetic
bf16 tensors, same values, partitioned four ways:

    float32 exact         121.222923
    one group             121.000000
    split 100/294         121.153351
    split 200/194         121.050613
    split 300/94          121.011543

Under pipeline parallelism that grouping is the model cut, so the same gradients report a
different total norm depending on where the model was split.

This adds `dtype`, passed to the three norm calls the function already makes.
`torch.linalg.vector_norm` and `torch._foreach_norm` both accept it already, so nothing
below this function changes. `dtype=None` is current behaviour, including the empty-input
return.

DTensor inputs need no special case: `_foreach_supported_types` is matched by exact type, so
they already take the `vector_norm` branch, and passing `dtype` there preserves the
`_NormPartial` placement rather than materializing anything.

Deliberately not added to `clip_grad_norm_`, which also scales the gradients through
`_clip_grads_with_norm_` -- there "which ops use this dtype" would be a real question. A
caller wanting an fp32 norm composes `get_total_norm(..., dtype=torch.float32)` with
`clip_grads_with_norm_(...)`.

Fixes #NNNNN.

--- PASTE END ---

## The diff

Signature:

    @_no_grad
    def _get_total_norm(
        tensors: _tensor_or_tensors,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    +   dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

Docstring, after the `foreach` entry:

    dtype (torch.dtype, optional): if specified, the tensors are cast to
        :attr:`dtype` prior to doing the accumulation, and the returned norm
        has this dtype. Use it when the tensors are in a low-precision dtype,
        where the accumulation error otherwise depends on how they are
        grouped. Default: ``None``

Body, four edits:

    -   return torch.tensor(0.0)
    +   return torch.tensor(0.0, dtype=dtype)

    -   norms.extend(torch._foreach_norm(device_tensors, norm_type))
    +   norms.extend(torch._foreach_norm(device_tensors, norm_type, dtype=dtype))

    -   [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
    +   [torch.linalg.vector_norm(g, norm_type, dtype=dtype) for g in device_tensors]

    -   torch.stack([norm.to(first_device) for norm in norms]), norm_type
    +   torch.stack([norm.to(first_device) for norm in norms]), norm_type, dtype=dtype

`torch.tensor(0.0, dtype=None)` is float32, so the empty case is unchanged at the default.
The norm-of-norms would inherit `dtype` from its inputs anyway; passing it makes the return
dtype a guarantee rather than a consequence.

## The test that encodes the bug

Partition-independence is the property, not "fp32 is closer" -- a test that only asserts
accuracy would pass on a change that improves precision and leaves the grouping dependence.

    def test_get_total_norm_dtype_is_partition_independent(self):
        torch.manual_seed(0)
        tensors = [torch.randn(64, dtype=torch.bfloat16) for _ in range(394)]
        whole = torch.nn.utils.get_total_norm(tensors, 2.0, dtype=torch.float32)
        self.assertEqual(whole.dtype, torch.float32)
        for cut in (100, 200, 300):
            halves = torch.stack([
                torch.nn.utils.get_total_norm(tensors[:cut], 2.0, dtype=torch.float32),
                torch.nn.utils.get_total_norm(tensors[cut:], 2.0, dtype=torch.float32),
            ])
            self.assertEqual(torch.linalg.vector_norm(halves, 2), whole)

    def test_get_total_norm_dtype_default_unchanged(self):
        tensors = [torch.randn(64, dtype=torch.bfloat16) for _ in range(8)]
        self.assertEqual(torch.nn.utils.get_total_norm(tensors, 2.0).dtype, torch.bfloat16)

The same two assertions on the bf16 default are what fail today: the halves recombine to a
different value than the whole.

## To confirm at filing time, not before

* **Where the test goes.** `get_total_norm`'s existing coverage was not located; find it
  rather than guessing a file.
* **`torch._foreach_norm(t, ord, dtype=None)` with an explicit `None`.** The schema is
  `ScalarType? dtype=None` so it should be accepted, and our torchtitan patch only ever
  passes `torch.float32`. If it rejects `None`, the foreach branch needs a conditional
  kwarg. One line to check, and the whole diff depends on it.
* Whether a `dtype` that is not floating point should raise here or be left to
  `vector_norm`'s own error.

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
