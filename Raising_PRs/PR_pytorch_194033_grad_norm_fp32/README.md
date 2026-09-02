# pytorch PR-194033: get_total_norm dtype + DTensor _foreach_norm strategy -- backup of the rebased branch

Branch `get-total-norm-dtype` on the fork, rebased 2026-09-02 onto pytorch main
`a6fc3c2959` (710 commits, no conflicts). NOT pushed yet -- the
force-push waits for the user. The three patches here are the branch verbatim.

1. `0001-*` DTensor: `_foreach_norm.Scalar` shares vector_norm's strategy but has no dim
2. `0002-*` get_total_norm: a `dtype` argument for the accumulation
3. `0003-*` the reviewer-requested regression tests (added tonight): `test_nn.py::test_get_total_norm_dtype`
   (bf16 grads, dtype=float32, one call vs norm-of-group-norms, foreach x p in {1,2}, CPU/CUDA) and
   `test_math_ops.py::test_foreach_norm_dtype` (DTensor `_foreach_norm(..., dtype=)` -- position 2 is dtype, not dim).

Validated on nightly 2.15.0.dev20260902+cu130 (separate venv, the two source patches overlaid): 8 passed / 2 passed.
The p=1 reference comparison uses rtol 1e-4: one-pass sum vs norm-of-norms differ by fp32 accumulation order.
