# Overnight 结果（2026-09-28 夜）

计划：`OVERNIGHT_PLAN_2026-09-28.md`。用户的目标："全部完成，不要动任何已经发布pr的分支"，任务 T2 到 T6。

## 约束的执行

- 已发布 PR 的分支一个都没动：#4656 `k3_ac_reuse_attention`、#4780、#4881、DEP #4381 `k3_pp_mm`、#4765 `k3_pp_offload`、#4764 `k3_pp_balance`。
- T4 里"推两个 draft 分支"按这条约束改成只推它们的 review 分支（`pp_offload_review1`、`pp_balance_review1`）；draft 也是已发布的 PR。
- 计划里的 T1（H100）不在目标里，H100 没动，机器还开着（按小时计费）。

## T6 pre-commit（含 pyrefly）

- **#4656**（`5d469fdf3`，`model.py`、`sharding.py`、`test_kimi_k3_attention_residual_recompute.py`）：trailing whitespace、ast、merge conflict、large files、license、flake8、µfmt、pydoclint、codespell 全部通过。pyrefly 失败，但和 main（同一环境、同一个 hook，干净的 `f35966713`）逐文件比较完全相同：22 个文件、42 个错（torch 0906 缺的接口、`torch_checkpointing` 等环境问题和 main 自己的错）。#4656 没有新增。
- **PR A**（`pp_review_optimize`）：同样只有 pyrefly 失败，但比 main 多 1 个：`stage.py` 的 `get_fwd_recv_ops` 把 `info.tensor_meta`（`_TensorMeta | None`）直接传给 `_make_tensor_from_meta`。补上 `is not None` 的判断（反向那边本来就有），amend 成 `d40bd628d`，推到 review 分支；之后 pyrefly 和 main 完全相同，PP 单测 67 passed。前向要接收的张量一定有 meta，所以这个判断不改变行为。
- hook 会在整个仓库删掉 pyrefly 认为多余的 suppression（25 个文件 40 处），这些附带改动都在临时 worktree 里撤掉了，没有进任何提交。
- 日志：`kit_overnight_2026-09-28/t6_*.log`。
