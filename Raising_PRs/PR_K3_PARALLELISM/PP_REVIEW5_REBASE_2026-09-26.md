# pp_review5 rebase 到 main（2026-09-26，shuhuayu 要求）

## 分支

- `pp_review5` = `ffdd169ef`，基于 upstream main `d0f3bbfd6`（比旧 base `9e159aed7` 新 32 个提交），已强推到 fork。
- 强推前的头 `7814d1f8b` 保存在 `backup/pp_review5_pre_20260926`。
- 用户说"直接同步到pr分支"后，`k3_pp_text`（PR 4312 的头）从 `7814d1f8b` 强推到 `ffdd169ef`。GitHub 上 `refs/pull/4312/head` = `ffdd169ef`，API 显示 mergeable = true（state unstable，CI 还没跑完），15 个提交、20 个文件。
- 按用户要求不重跑数值。

## 冲突

- GitHub 页面只显示 `tests/unit_tests/cpu/test_pipeline_parallel.py` 一处：main 和 4312 都在文件末尾加了测试。解法是 main 的测试在前，4312 的测试接在后面。
- 逐个提交重放时另外停了三次（import 行、测试文件末尾、`TrainingEngine` 的 import）。这三次都来自中间提交，GitHub 只合并整体 diff，所以看不到。
- 最终树与 `git merge-tree --write-tree upstream/main 7814d1f8b` 的结果对比：除了该测试文件（解法如上），只多了下面这一行 import。

## 语义冲突（GitHub 显示不出）

- 4312 最后一个提交把 `distributed/pipeline_parallel.py` 改用 `copy.copy`，同时删掉了 `import dataclasses`。main 在同一文件新加了 `@dataclasses.dataclass(frozen=True)`（第 654 行）。
- 两边不在同一行，git 不报冲突，但合并后模块一 import 就 `NameError`。
- 仓库的 `.flake8` 忽略 F821（未定义名字），lint 也发现不了；是跑 CPU 单测时暴露的。
- 修法：该提交保留这行 import，提交信息不变。

## 检查

- PR 相对 main 的 diff，与原来相对 `9e159aed7` 的 diff 按 +/- 行逐行比较：唯一差别是不再删除 `import dataclasses`。新增的注释和 docstring 仍是那 100 行。
- pyflakes、flake8、ufmt：20 个改动文件全部通过。
- 与 PR 相关的 7 个 CPU 测试文件：134 passed。
- 整个 `tests/unit_tests/cpu`：27 failed / 1230 passed / 39 skipped，另有 2 个文件因本地 torch 旧在收集阶段失败。
  - 24 个在 main `d0f3bbfd6` 上同样失败：缺 `flash_attn`；torch 没有 `pipeline_per_edge_p2p`；torchao 的 MXFP8。
  - 其余 3 个是全量运行中的 inductor `CppCompileError`，单独跑在 rebase 后的树上全部通过。
  - 没有能归到本 PR 的失败。
- main `d0f3bbfd6` 把 KDA 的门槛改成 capability ≥ 9.0，5060（SM120）和 H100（SM90）都不再需要本地放宽。

## GPU smoke（8 × 5060 Ti，`/workspace/venv_bfx9`，torch 2.15.0.dev20260906）

- 本地没有哪个 torch 带 main 新用的 `torch.distributed.config.pipeline_per_edge_p2p` 和调度器的 `unshard_lookahead` 参数。所以 smoke 带一个不提交的兼容补丁 `kit_pp_review5_rebase_2026-09-26/pr5_torch_compat_shim.patch`：前者只在存在时设置；后者按调度器构造函数的签名过滤。
- 脚本：`kit_pp_review5_rebase_2026-09-26/run_smoke.sh`；pp4 × vp4 的本地 recipe 是 `smoke_pr5.py`（临时，不进任何分支）。

- 组合格 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4`（B200 recipe 原样，8 卡），10 步，rc=0。loss 8.38703 / 7.18665 / 5.35821 / 4.86059 / 4.43614 / 4.04308 / 3.87640 / 3.59047 / 3.98431 / 3.40537。这个格子没有固定 seed，只看能否跑通，不比数值。

- pp4 × vp4（临时本地 recipe `smoke_pr5.kimi_k3_pp4_vpp4`，4 卡，Interleaved1F1B，4 个 micro-batch，无 FSDP/TP/EP）：16 段切分由 core 的 `_generate_llm_fqn_per_model_part(16, 17, 1, 1)` 推出并钉上两端模块，与 round 5 删掉的手写 recipe `kimi_k3_debugmodel_pp4_vp4` 完全相同。
  - AdamW（与组合格相同）：10 步，rc=0。loss 8.05654 / 7.18430 / 5.47913 / 5.11689 / 4.77169 / 4.13480 / 3.91005 / 3.68247 / 3.49561 / 3.67943。
  - debug model 默认的 DistMuon（旧 cell 的定义）：rc=1，建优化器时 rank 3 报 `ValueError: Optimizer param_groups pattern '...' matched no parameters`（`components/optimizer/optimizer.py:208`）。rank 3 的最后一段只有 `norm`、`lm_head`、`output_res_proj`、`output_res_norm`，没有 Muon 要的矩阵参数。这是 09-23 记下的 #4596 约束（每段都要有一个 transformer 层，K2.7 README 的规则），与 rebase 无关；round 5 删掉这个 cell 前它就是这个状态。
  - DistMuon，最后一段带上 `layers.16`（`pipeline_parallel_last_stage_less_layers = 0` 后由同一个 util 推出切分）：10 步，rc=0。16 段切分为 `[vision_encoder, tok_embeddings, layers.0]`、`[layers.1, layers.2]`、`layers.3` 到 `layers.15` 各一段、`[layers.16, norm, lm_head, output_res_proj, output_res_norm]`。loss 8.08413 / 8.00789 / 7.71354 / 7.48460 / 7.35999 / 6.83611 / 6.55681 / 6.27263 / 6.52361 / 6.04448。脚本 `run_smoke3.sh`。

## 下一步

- #4765、#4764、#4381 仍叠在 `7814d1f8b` 上，需要重新叠到 `ffdd169ef`。
- 兼容补丁仍以未提交改动留在 worktree `/tmp/wt_pp_review5`，提交时只 stage 目标文件。
