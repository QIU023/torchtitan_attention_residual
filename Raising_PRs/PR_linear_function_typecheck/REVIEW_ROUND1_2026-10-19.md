# #5390 第一轮 review：收紧类型检查测试，rebase 到新 main（2026-10-19）

## 状态（不粘贴）

- **review：** anijain2305 10-19 APPROVED（https://github.com/pytorch/torchtitan/pull/5390#pullrequestreview-5567608968），只建议收紧测试：原测试只检查不报错、输出形状和 `lora_b` 有梯度，输出类型错成 P 或 V 也能过；不开 SP 的路径（P 经 all-reduce 变 I）没测到。建议按 SP 参数化，并在边界上断言输出类型。jinsooihm 10-14 已 APPROVED。
- **用户 10-19：** "本机上解决一下这个LoRA fix PR comment"。
- **改动：** 新提交 `c35491f4c`（rebase 前是 `89fc6f9a7`），单独一个提交，不 squash：
  - `@parametrize("enable_sp", [True, False])`，`instantiate_parametrized_tests`，写法照 `tests/unit_tests/gpu/distributed/flex_shard/test_dist_muon.py`；
  - 输入和输出都断言 `residual_activation_placement(enable_sp=enable_sp)`，导入从 `dense_sequence_parallel_placement` 换成它。
- **rebase：** PR 在 GitHub 上是 `dirty`。main 的 #5345 把 `linear.py` 挪到了 `linear/linear.py`（git 按改名自动合并），并把 `FeedForward` 改名为 `DenseFeedForward`（`test_lora.py` 的导入冲突，只解这一段：用 main 的 `DenseFeedForward`，保留本 PR 的 placement 导入）。
  - 新 base：upstream main `efdde5eda`（10-18）；分支 = `0a02d6931`（修复）+ `c35491f4c`（测试）。
  - 两个提交的增删行和 rebase 前逐行相同（patch-id 不同只是因为路径和上下文）。
  - 旧 head 备份在本地 `backup/linear_function_typecheck_pre_20261019`（= `89fc6f9a7`，含新测试；PR 线上 head 仍是 `586c1fd07`）。
- **验证（本机 5060，venv_1006，PYTHONPATH = `main_pins_1016`，和 `efdde5eda` 的 requirements 一致）：**
  - `-k TestLoRATensorParallelTypecheck`：2 passed；`linear/linear.py` 换回 main 的版本：2 failed，都是 `Partial type ... Found types: [P, R]`。
  - 只把输出那条断言的期望换成另一种 layout：2 failed，都停在输出断言那一行（有 SP 时 V 对 I，无 SP 时 I 对 V）。说明新断言确实能区分两种类型。
  - `test_lora.py` 31 + `test_linear.py` 22 = 53 passed（12 subtests），`CUDA_VISIBLE_DEVICES=""`。
  - llama3 debugmodel + main 的 LoRA recipe（rank 8，`wqkv`、`wo`），TP2 + SP，开类型检查，3 步：分支 rc 0（8.02929 / 8.05165 / 7.99786）；`linear/linear.py` 换回 main：rc 1，同一个 `SpmdTypeError`。probe 在 `scratchpad/probe_5390/`（关掉 sm120 的 FA4 选择，只为本机能跑）。
  - pyrefly 0.45.1（固定版本 hook 的版本，PYTHONPATH 指向固定的 spmd_types）：分支和 `efdde5eda` 的干净导出都是 4 个错误，集合相同。不设 PYTHONPATH 时分支会多 3 个 `No attribute rules in module spmd_types`，那是 venv 里 spmd_types 太旧，不是分支的问题。
  - flake8（venv_bfx9）、ufmt 通过；diff 没有新增注释或 docstring。
- **推送：** 还没推。`linear_function_typecheck` 是已发布 PR 的分支，rebase 后要 force push，等用户点头：
  `git push origin backup/linear_function_typecheck_pre_20261019 && git push --force-with-lease=linear_function_typecheck:586c1fd071c2f813c71b15a6f02750b57d2bc7ed origin linear_function_typecheck`
- **推后：** 回复贴在 anijain2305 的 review 下面；body 换成下面第二个粘贴区（路径、测试说明和计数变了，其余不变）。

## 回复

--- PASTE BEGIN (reply) ---

Done in c35491f4c. The test is now parametrized over `enable_sp` and asserts `residual_activation_placement(enable_sp=enable_sp)` on both the input and the output, so the output must be V with SP (reduce-scatter) and I without SP (all-reduce). Expecting the other layout on the output fails both cases, and both cases fail on main with the Partial error. I also rebased onto main, where `linear.py` moved to `linear/linear.py`; the fix itself is unchanged.

--- PASTE END (reply) ---

## body

--- PASTE BEGIN (body) ---

## Summary

LoRA under tensor parallelism fails SPMD type checking: on a row-parallel linear, `lora_b` receives a partial sum from `lora_a`, and `_LinearFunction`, which `Linear` uses by default, accepts no partial-sum input; this PR gives `_LinearFunction` the typing rule of a matmul, which is linear in its input.

- `torchtitan/models/common/linear/linear.py`: `_LinearFunction.spmd_typecheck` types the GEMM with `rules.einsum(..., linear_in=(0, 1))` and the bias add as jointly linear, in place of `register_local_autograd_function`.
- `tests/unit_tests/cpu/models/common/test_lora.py`: a two-rank test runs a LoRA feed forward with TP, with and without SP, under the type checker, forward and backward, and checks the residual type of its output.

## Design

- Typing:
  - A sharded contraction gives Partial, and a Partial operand passes through when the other operand is Replicate, as for any matmul.
  - On a row-parallel projection, `lora_a` (sharded on its input dim) gives Partial, `lora_b` (Replicate) keeps it, and `RowParallelLinear` reduces it together with the base output.
  - The bias add is jointly linear, so a Partial output only takes a Partial bias.
  - The stacked dim of a `num_linears > 1` weight is typed as never sharded.
- The computation is unchanged; only the type rule differs.

## Test plan

```
pytest tests/unit_tests/cpu/models/common/test_lora.py -k TestLoRATensorParallelTypecheck  # 2 passed; SpmdTypeError on main
pytest tests/unit_tests/cpu/models/common/test_lora.py                                     # 31 passed
pytest tests/unit_tests/cpu/models/common/test_linear.py                                   # 22 passed
```

`llama3_debugmodel` with TP2, SP, `LoRATransform` and `debug.spmd_typechecking` raises `SpmdTypeError` on main and runs here. `pyrefly check` reports the same errors as main.

--- PASTE END (body) ---
