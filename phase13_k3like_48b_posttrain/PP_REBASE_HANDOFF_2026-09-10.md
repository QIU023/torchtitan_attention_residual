# PP 4312 rebase 到 main 之后 — 给 GPU 盒子（2026-09-10）

`pp_review5` = `6042863a4`：`k3_pp_text`（`75045fed5`，与 `pp_review4` `9bd39bd40` 同树，只差最后一个提交的措辞）的 19 个提交整体 rebase 到 upstream/main `d398a8fb9`（#4572 之后）。4312 从 #4527 合入（09-09）起在 GitHub 上就是 unmergeable（`dirty`）。最后一个提交用了 pp_review4 的措辞（无 @ 提及）。

## 冲突与解法（只有两处，都对 #4527）

- `model.py` forward：main 把首段起始的空栈从 `h_TD.new_zeros(T, 0, D)` 改成 `h_TD.unsqueeze(1)[:, :0]`（带 spmd 类型），并在前面加了 `spmd.is_type_checking()` 断言。PP 分支的 `block_residual_in if block_residual_in is not None else <空栈>` 保留，空栈取 main 的写法。对老分支的 interdiff 就这两行。
- `parallelize.py`：import 块保留 main 的 `annotate_replicated_parameters`；塔的 FSDP 调用保留 main 新增的 `cpu_offload=` / `dp_mesh_dims=`，`pp_enabled=parallel_dims.pp_enabled` 不变。
- 其余 12 个文件自动合并（`pipeline_parallel.py` main 只改了一条报错文案；`utils.py` main 在 `init_distributed` 开头加了 bf16x9 调用，我们的 `device_id` 绑定在末尾）。`git diff upstream/main` 仍是 14 个文件 +1693/−34；提交信息无引用、无 trailer。
- 一个待观察点：`h_TD = tokens` 那条（非首段收到的激活）在 `typechecking=True` 下会撞 `assert_type(h_TD, {DP: S(0)})`，因为收到的张量没有 spmd 标注。PP 格不开类型检查，现在不触发；要开的话得在 stage 收包后标注。

## Windows 这边验证过的

- 无冲突标记，`compileall` 通过；`test_pipeline_neighbor_transport.py` 6 通过；`test_integration_test_definitions.py` 18 通过 1 失败（`test_hf_checkpoint_load_path_comes_from_test_config`，`/tmp` 路径断言，纯 upstream/main 同样失败，Windows 问题）。
- 三个 K3 测试文件（`test_kimi_k3_pp_fqn_injection/layout/stage.py`）在这台机器导不进：main 的 `kda.py:15` 走 attn-gym `short_conv.causal_conv1d`，要 CuTeDSL `cutlass`，只有 Linux。

## 要做的

1. `git fetch origin pp_review5`；先跑三个 K3 CPU 测试文件 + `kimi_k3_pp8_vp4` 格（8 卡）。
2. 启动陷阱：#4484 让 `init_distributed` 调 `enable_fp32_matmul_emulation_with_bf16x9()`，SM ≥ 10.0（5060 Ti 是 SM120）上 torch 不接受 `fp32_precision = "bfx9"` 就直接 `ValueError`。这是 main 的行为，不是分支的；先确认 torch 过得了这一关。
3. 数值重配对：body 的表全是 `6e2ac3dcd` 基上测的，#4484（router 反向 fp32 + bf16x9）改了 flavor 的数字。按数值表规则在 `6042863a4` 上重读 step-1 bitwise 栏（dp1 对 pp2 / pp4 / pp8×vp4，两种传输，同一个 inductor 缓存），再决定 17 行表要不要全重跑。
4. 用户批准后同步 PR 分支：`git push --force-with-lease=refs/heads/k3_pp_text:75045fed5 origin pp_review5:k3_pp_text`。`PR_BODY_PP.md` 的头部已改成这个状态。
5. DEP 分支 `k3_pp_mm` / `k3_pp_mm_v2`（`c87097ae5`，同一个老基）同样要 rebase：它们在 `k3_pp_text` 之上的额外提交直接搬到 `pp_review5` 上。
6. Elfie：优化器状态那条不用她提 PR，main 的 #4474（`dc3985ad4`）已经覆盖；两节点测试请用 `pp_review5`（基里含 #4474）。两份 Elfie 笔记已改。
