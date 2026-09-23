# pytorch PR-194033: get_total_norm dtype + DTensor _foreach_norm strategy -- backup of the rebased branch

## 2026-09-23: rebased onto main `54144378a7`, Jane's 09-22 request addressed locally (not pushed)

Jane (2026-09-22, comment 5783420331) asked for a rebase, "fix the errors", and thoughts on the claude[bot] review's point 3 and the docstring item under 4. The errors were only the two `inductor_aoti_fallback` CPU shards, which Dr. CI marks unstable and unrelated; the PR was `mergeable_state: dirty` (863 commits behind). Rebase conflict: one import line in `test/test_nn.py` (upstream dropped `skipCUDAIfRocm`); resolved, and the `skipXLA` import went away with the change below.

Head `0ff051d35d` (four commits on `54144378a7`), not pushed:
1. `10997dca05` DTensor strategy (unchanged).
2. `174fa99d19` dtype argument: the docstring now states the constraint (a floating point dtype the inputs promote to, `torch.promote_types(input.dtype, dtype) == dtype`, the check `linalg.vector_norm` and the foreach CUDA kernel make; ATen: `LinearAlgebra.cpp` `check_linalg_norm_dtype`, `cuda/ForeachReduceOp.cu` L402).
3. `c0f7a04ff4` tests: `test_get_total_norm_dtype` is `@onlyOn(["cpu", "cuda"])`. The bot proposed `@onlyNativeDeviceTypes`, but `NATIVE_DEVICES` includes `mps` and `mtia` (common_utils.py L483), so it gates nothing useful. The structural alternative (`whole != split`) is no more portable: a bfloat16-accumulating backend gives 256 on both sides. The `subtest(decorators=[skipXLA, skipMPS])` wrapper is gone (dead under the gate); the literals stay.
4. `0ff051d35d` (new, droppable) `_foreach_powsum.Scalar` has the same defect (bot point 2, not asked for by Jane): `dim`/`keepdim` are now read by schema position in `_dim_keepdim_from_schema`, shared by both strategies; the op identity check is gone; `test_foreach_powsum_sharded` gains a bf16 + `dtype=float32` case. The foreach expansion passes the outer op to the per-tensor strategy (`single_dim_strategy.py` `expanded_foreach_strategy`), which is why the schema lookup on `op` is right.

Validated in `venv_ptnightly` (2.15.0.dev20260902+cu130) with the branch's two source files copied over the site-packages copies (backup of the 09-13 overlay in the session scratchpad): `test_nn.py -k get_total_norm_dtype` 8/8 (cpu, cuda); `test_math_ops.py -k foreach_norm -k foreach_powsum` 8/8 on 4 GPUs; the single-process negative check (`powsum_negcheck.py`, gloo, world 1) raises `'>=' not supported between instances of 'torch.dtype' and 'int'` with the 09-13 strategy and passes with the branch's. ruff clean; ufmt (black 22.12 here) flags only the two untouched upstream lines of `clip_grad.py`. Patches: `patches_2026-09-23/`; diff: `full_2026-09-23.diff`; reply: `REPLY_2026-09-23.md` (one comment, PASTE markers). Next: the user pushes (force with lease, as on 09-14) and pastes the reply.

## 2026-09-13: rebased again and Jane's review addressed

Force-pushed 2026-09-14 with the user's approval: `git push --force-with-lease=get-total-norm-dtype:64ec4af36b origin 1ad4f1623f:refs/heads/get-total-norm-dtype` (`64ec4af36b...1ad4f1623f`). Next: post `REPLY_2026-09-13.md`, one block under each of Jane's comments.

Branch rebased onto main `b8c2b362c4` (435 commits; the test commit conflicted in `test/test_nn.py` only because upstream dropped `@skipIfMPS` from `test_clip_grad_norm`, kept dropped). Head `1ad4f1623f`: `f23d666f01` DTensor strategy (Jane's comment wording), `ddc50a05b4` dtype argument (Jane's docstring), `1ad4f1623f` tests rewritten -- `test_get_total_norm_dtype` asserts exact values (`atol=0, rtol=0`) on bf16 `[255, 32, 1]`, whole vs split, and the current bf16 split dependence at p=2 (258 vs 256); foreach skips on XLA / MPS via `subtest(decorators=[skipXLA, skipMPS])`; the DTensor dtype case folded into the existing `test_foreach_norm` as a subTest. Validated on `venv_ptnightly` (2.15.0.dev20260902 with the source changes): 8 + 6 passed. On the unpatched nightly (`venv_bfx9`, 2.15.0.dev20260906) the DTensor `test_foreach_norm` fails with the defect (`'>=' not supported between instances of 'torch.dtype' and 'int'`) and `get_total_norm(..., dtype=)` raises `TypeError: ... unexpected keyword argument 'dtype'`; `test_nn.py` itself does not import there (no `hypothesis`). `ufmt` (black 22.12 here) flags only two untouched upstream lines of `clip_grad.py`; ruff clean. Replies: `REPLY_2026-09-13.md`; patches: `patches_2026-09-13/`.

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
