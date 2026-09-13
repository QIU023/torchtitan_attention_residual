# pytorch PR-194033: get_total_norm dtype + DTensor _foreach_norm strategy -- backup of the rebased branch

## 2026-09-13: rebased again and Jane's review addressed (local; force-push pending the user's approval)

Branch rebased onto main `b8c2b362c4` (435 commits; the test commit conflicted in `test/test_nn.py` only because upstream dropped `@skipIfMPS` from `test_clip_grad_norm`, kept dropped). Head `1ad4f1623f`: `f23d666f01` DTensor strategy (Jane's comment wording), `ddc50a05b4` dtype argument (Jane's docstring), `1ad4f1623f` tests rewritten -- `test_get_total_norm_dtype` asserts exact values (`atol=0, rtol=0`) on bf16 `[255, 32, 1]`, whole vs split, and the current bf16 split dependence at p=2 (258 vs 256); foreach skips on XLA / MPS via `subtest(decorators=[skipXLA, skipMPS])`; the DTensor dtype case folded into the existing `test_foreach_norm` as a subTest. Validated on `venv_ptnightly` (2.15.0.dev20260902 with the source changes): 8 + 6 passed; on the unpatched nightly (`venv_bfx9`) both fail. Replies: `REPLY_2026-09-13.md`; patches: `patches_2026-09-13/`.

Branch `get-total-norm-dtype` on the fork, rebased 2026-09-03 onto pytorch main
`6b2b0ddbdd` (first rebased 09-02 onto `a6fc3c2959`, 710 commits, no conflicts; re-rebased
09-03 over 34 more, no conflicts, the combined diff is byte-identical before and after).
Force-pushed 2026-09-03: PR head `64ec4af36b`, commits `d03377a101` (DTensor strategy),
`f502519356` (dtype argument), `64ec4af36b` (tests). The first push of the day (`31559d860f`) was
re-pushed after `tools/linter/adapters/pyfmt_linter.py` flagged two over-long list comprehensions
(clip_grad.py, test_math_ops.py; the old head's lint job had failed on the clip_grad.py one). AST
identical before and after; RUFF clean. CI on a fork PR waits for a maintainer's approval
(check-suites show `action_required`), so nothing runs until someone clicks approve. The three patches here are
the branch as it was before the 09-03 rebase; the content is the same.

1. `0001-*` DTensor: `_foreach_norm.Scalar` shares vector_norm's strategy but has no dim
2. `0002-*` get_total_norm: a `dtype` argument for the accumulation
3. `0003-*` the reviewer-requested regression tests (added tonight): `test_nn.py::test_get_total_norm_dtype`
   (bf16 grads, dtype=float32, one call vs norm-of-group-norms, foreach x p in {1,2}, CPU/CUDA) and
   `test_math_ops.py::test_foreach_norm_dtype` (DTensor `_foreach_norm(..., dtype=)` -- position 2 is dtype, not dim).

Validated on nightly 2.15.0.dev20260902+cu130 (venv_ptnightly, the two source patches overlaid):
8 passed / 2 passed, rerun 2026-09-03 before the push with the same result. Without the overlay
both tests fail on exactly the two defects (`unexpected keyword argument 'dtype'`;
`'>=' not supported between 'torch.dtype' and 'int'`), so they pin what the reviewer asked for.
The p=1 reference comparison uses rtol 1e-4: one-pass sum vs norm-of-norms differ by fp32 accumulation order.
