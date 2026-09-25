# 新 PR：Kimi K3 attention residual 投影的 zero init（从 #4780 拆出）

tianyu-l 在 #4780 的 review（5299962900，r4090059193）里说："fix the init in its own PR"。

## 状态（2026-09-25，不粘贴）

- **分支：** fork 上的 `k3_attnres_zero_init` = `ae3a7881b`，是 upstream main `56f04c702` 之上的 1 个 commit，只改了 `torchtitan/models/kimi_k3/__init__.py`。09-25 按用户要求（"不需要unit test"）去掉了测试文件，此前是 `c86dfc2e9`。
  - 09-24 的版本 `db483314a` 在 main `9e159aed7` 上。09-25 把它 cherry-pick 到最新 main，没有冲突，旧版备份为 `backup/k3_attnres_zero_init_pre_20260925`。
  - 按"默认不加注释"规则，`__init__.py` 里的注释合成了一行；测试文件已整个删掉。
- **开 PR：** 用户从 fork 开，链接 https://github.com/pytorch/torchtitan/compare/main...QIU023:k3_attnres_zero_init?expand=1 。标题见下。开好之后，把 PR 号填进 #4780 的回复和 body v3 里的 `#INIT_PR`。
- **09-25 验证（测试文件已不在分支里，验证时放在仓库外对新 commit 运行）：**
  - `pytest tests/unit_tests/cpu/test_kimi_k3_attention_residual_init.py -q`：3 passed。同一测试文件放在不带这个 commit 的 main 上跑，初始化那一条失败（1 failed, 2 passed），可作对照。
  - pre-commit 的 flake8、µfmt、license：Passed。
  - pyrefly hook 报"Failed"，是因为它在全仓库删多余的 suppression，改了 25 个别的文件（已回退）。这两个文件没有报错，也没有被改动。
- **还要处理的：** #4780 的 PR 分支 `k3_attnres_recompute`（`9f6bae06f`）最上面仍是这个 zero init commit。
  - 去掉它有两种做法：随 `attnres_review1`（`38fcdde4a`，不含 init）同步时一起去掉；或者单独把 PR 分支退回 `1436053ea`。
  - 用哪种由用户决定，这次没有动 #4780。

Title: `[Kimi K3] Zero-initialise the attention residual projections`

--- PASTE BEGIN ---

## Summary

Initialise Kimi K3's attention residual projections to zero, as Section 5 of the Attention Residuals technical report requires, so the depth softmax starts uniform and each residual starts as an equal-weight average of its sources.

- `torchtitan/models/kimi_k3/__init__.py`: `attention_res_proj`, `ffn_res_proj` and `output_res_proj` take `nn.init.zeros_` instead of `trunc_normal_` at std 0.02.

Split out of #4780, as suggested there.

## Test plan

- The three projections of the debug model take `nn.init.zeros_`, and at zero the aggregation returns the mean of its sources (checked locally).

--- PASTE END ---
