# pytorch PR-194033: get_total_norm dtype + DTensor _foreach_norm strategy -- backup of the rebased branch

## 2026-09-25：review 分支 `get-total-norm-dtype-review1` 重跑验证（只验证，PR 分支和回复都没动）

**分支：**
- fork `QIU023/pytorch` 的 `get-total-norm-dtype-review1`，head `bdbf5318ba`。
- 上面是 4 个 commit：
  - `bc82bab519`：DTensor `_foreach_norm.Scalar` 的策略；
  - `2f743565e0`：`get_total_norm` 的 dtype 参数；
  - `6143bf7b97`：测试；
  - `bdbf5318ba`：powsum。
- 与 upstream main 的 merge-base 是 `69f2e45ef3`。
- 改了 4 个文件，+99/-18。117 行改动与 `full_2026-09-23.diff` 逐行一致，文件顺序也相同。
- PR 分支 `get-total-norm-dtype` 仍在 `1ad4f1623f`，本次未动。

**环境：**
- `venv_ptnightly`，torch `2.15.0.dev20260902+cu130`，`torch.version.git_version` = `3c73a854d2`。
- **兼容性：** 两个源文件在 nightly `3c73a854d2` 与 main `69f2e45e` 之间差 0 行，所以覆盖之后等于 nightly 加上本 PR 的改动。
  - 两个测试文件在两者之间有上游改动：`test/test_nn.py` 差 1788 行，`test_math_ops.py` 差 3 行。本次只跑了本 PR 的用例。
- **覆盖：**
  - 覆盖前，site-packages 里这两个文件已经和 review1 相同（md5 `06fbcd34` / `b78d164c`），是 09-23/24 覆盖后没有恢复留下的。原版 nightly 的这两个文件就是 `3c73a854d2`（= `69f2e45e`）上的版本。
  - 照样先备份到 scratchpad `pr194033_0925/venv_backup_0925/`，再用 `git show bdbf5318ba:<path>` 取出的文件覆盖，跑完恢复成备份。
- **测试文件：** 用 `git show bdbf5318ba:test/...` 取到 scratchpad 目录里运行。不能在 pytorch 源码目录下运行，否则 `import torch` 会加载源码树里的 torch。

| 项 | 原始命令（在 scratchpad 目录，`python` = `/workspace/venv_ptnightly/bin/python`） | 结果 |
|---|---|---|
| 1 | `python -m pytest test_nn.py -k get_total_norm_dtype -v -p no:cacheprovider -rs` | **8 passed**，5222 deselected。cpu 4 个、cuda 4 个：foreach False/True × norm_type 1.0/2.0 |
| 2 | `CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest test_math_ops.py -k "foreach_norm or foreach_powsum" -v -p no:cacheprovider -rs` | **8 passed**，104 deselected，2 subtests passed。`DistMathOpsTest` 与 `DistMathOpsTestWithLocalTensor` 各 4 个：`test_foreach_norm`、`_different_mesh`、`_partial`、`test_foreach_powsum_sharded` |
| 3 | 反向检查：`python powsum_negcheck.py <port>`（gloo，world 1），依次把三版 `_math_ops.py` 覆盖进 site-packages | 旧策略：PR 分支 `1ad4f1623f` 的版本（md5 `8ef9f0e5`）和 09-13 版（`venv_backup_0923`，md5 `40f893a7`，与前者只差 3 行注释）都输出 `RAISED RuntimeError '>=' not supported between instances of 'torch.dtype' and 'int'`。review1（md5 `b78d164c`）输出 `OK torch.float32 (Partial(sum),)` |
| 4a | `ruff check <四个改动文件>`（ruff 0.16.5，配置为 review1 的 `pyproject.toml`） | `All checks passed!` |
| 4b | `ufmt check <四个改动文件>`（ufmt 2.3.0，black 22.12.0） | **超出"只允许 clip_grad.py 两行"的范围**，见下 |

**4b 的原始输出：**

```
Would format torch/nn/utils/clip_grad.py
Would format test/test_nn.py
✨ 2 files would be formatted, 2 files already formatted ✨
```

**4b 的定位（只定位，不下结论）：**
- **`torch/nn/utils/clip_grad.py`：** 两处 hunk，即 `_tensor_or_tensors: TypeAlias = ...  # noqa: PYI042` 和 `] = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]`。base `69f2e45e` 上同样出现，就是上游原有的那两行。
- **`test/test_nn.py`：** base `69f2e45e` 上 ufmt 同样报这个文件，整个文件没有按 black 格式化，base 的 ufmt diff 约 1.4 万行。把 review1 与 base 的 ufmt diff 相减，只在 review1 出现的改动有下面 4 处，都在本 PR 新增的 `test_get_total_norm_dtype` 里：

```
-    @parametrize_test('foreach', (False, True))
-    @parametrize_test('norm_type', (1.0, 2.0))
+    @parametrize_test("foreach", (False, True))
+    @parametrize_test("norm_type", (1.0, 2.0))
-        exact = {1.0: 255.0 + 32.0 + 1.0, 2.0: math.sqrt(255.0**2 + 32.0**2 + 1.0**2)}[norm_type]
+        exact = {
+            1.0: 255.0 + 32.0 + 1.0,
+            2.0: math.sqrt(255.0**2 + 32.0**2 + 1.0**2),
+        }[norm_type]
-            total = get_total_norm(tensors, norm_type=norm_type, foreach=foreach, dtype=torch.float32)
+            total = get_total_norm(
+                tensors, norm_type=norm_type, foreach=foreach, dtype=torch.float32
+            )
```

- **pytorch 实际用的格式化 linter：** review1 的 `.lintrunner.toml`（第 1303 行起）里负责 Python 格式化的是 `PYFMT`（`tools/linter/adapters/pyfmt_linter.py`），不是本地的 ufmt。它的 `exclude_patterns` 包含 `test/test_nn.py`，另外三个改动文件都在它的检查范围内。本地没有跑 PYFMT。
- **与 09-23 记录的差异：** 09-23 的记录只写了 `clip_grad.py` 那两行，没有写当时是否对 `test_nn.py` 跑过 ufmt。

**本次没有做的：** 没有推 `get-total-norm-dtype`，没有贴 `REPLY_2026-09-23.md`。中间文件都在 scratchpad 的 `pr194033_0925/`：ufmt 的完整 diff、比对结果和备份。

## 2026-09-24: re-checked; rebased again onto main `a0afa8eb62` (still not pushed)

No new comment since Jane's 09-22 request. Main moved 92 commits past `54144378a7` without touching any of the four
files this PR changes; the rebase is clean: head `6f9a8ecbfc` (`775e12fd72` DTensor strategy, `3d5a0ca015` dtype +
docstring constraint, `80fc4c94d1` tests with `@onlyOn(["cpu", "cuda"])`, `6f9a8ecbfc` powsum, droppable). Re-run on
`venv_ptnightly` with the two source files overlaid (unchanged since 09-23): `test_get_total_norm_dtype` 8/8 on cpu
and cuda, DTensor `foreach_norm` / `foreach_powsum` 8/8 on 4 GPUs. A bf16-accumulating backend, simulated add by add,
gives 256 for both the whole tensor and the split, so the bot's structural `whole != split` would fail there too; CPU
gives 258 and 256. `NATIVE_DEVICES` on main is still ('cpu', 'cuda', 'xpu', 'meta', 'mps', 'mtia', privateuse1).

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
