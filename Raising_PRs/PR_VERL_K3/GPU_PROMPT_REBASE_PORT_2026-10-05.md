# veRL fork：rebase 到上游 veRL main 并移植到当前 torchtitan main（给 GPU 会话的 prompt，2026-10-05）

下面 PASTE 段是交给 GPU 机器上 Claude 会话的完整任务说明。背景里的 SHA 和上游变化是 2026-10-05 在 Windows 机器上用 `git ls-remote` / `git fetch` 核对过的；GPU 会话开工时自己再核对一遍。

--- PASTE BEGIN ---

任务：把 veRL fork 的 K3 工作 rebase 到当前上游 veRL main，并把 torchtitan engine 移植到当前 torchtitan main，然后重切六个能力分支、跑分阶段测试和 GPU smoke。不提交任何上游 PR。回复我用中文。

先拉 logbook（`git pull --rebase`），按顺序读：
1. `CLAUDE.md`（规则全部适用，尤其 commit message、分支、注释、数值、草稿规则）
2. `Raising_PRs/PR_VERL_K3/PR_SPLIT_PLAN_2026-09-16.md`（拆分计划、items 1-14、各附录）
3. `Raising_PRs/PR_VERL_K3/split_2026-09-17/MANIFEST.md` 和 `work/`（`splitlib.py` / `assign.py` / `build.py` / `remap.py` / `apply_remap.py` / `verify.sh`）
4. `Raising_PRs/PR_VERL_K3/STAGE_TESTS_2026-09-19.md`、`ITEMS_11_12_2026-09-19.md`
5. `phase13_k3like_48b_posttrain/K3_INT_20260922.md` 的 veRL 段落和 `STATUS_2026-09-22.md` 的 veRL 节（09-22 的移植方式、runner、cell 列表）
6. `phase13_k3like_48b_posttrain/VLLM_PIN_AND_EXPORT_2026-09-17.md`

## 起点（10-05 核对，开工时重新核对）

veRL fork `QIU023/verl`：
- `kimi_k3_integration_rebased` = `04e55506`，base 是上游 `1a8a0f5f`（09-17），领先 79 个 commit；上游 `verl-project/verl` main = `8718ca30`（10-01），比这个 base 多 50 个 commit。
- 六个能力分支（叠放，base 上游 `3efe38c7`，09-18，落后上游 41 个）：`verl_engine_tp_packed` `ac6dd230`、`verl_engine_pp` `f0e9dff3`、`verl_engine_sync_ep_lora_qat` `8b7c95cb`、`verl_engine_cp` `f26c72d1`、`verl_kimi_k3` `d6c868d9`、`verl_metrics_logprob_diff` `18e7db00`。
- 不动：`kimi_k3_integration`、`sync_fixes_upstream`、`qat_sync_upstream_pending`。

上游 veRL 09-17 以来：
- 没有任何 commit 碰 `verl/workers/engine/torchtitan/`；`verl/workers/config/engine.py` 只改了 `McoreEngineConfig`（去掉 legacy mbridge）。
- 有 16 个 commit 碰到 fork 也改过的文件（rollout / vllm 工具、trainer、metrics），冲突在这些地方找。
- `3f9d4c0d` 把依赖升到 torch 2.13.0 / vllm 0.29.0 / transformers 5.12.1。查 vllm 0.29.0 是否已含 K3 支持（我们 rollout 现在跑的是 `6dc76a9ade` 源码构建加 `VERL_VLLM_VERSION` shim）；测出来再说，不要假设。

上游 veRL 自己的 torchtitan engine 已经落后 titan main：它在模块顶层 import `torchtitan.components.checkpoint`、`torchtitan.distributed.parallel_dims.ParallelDims`、`torchtitan.distributed.context_parallel.prepare_context_parallel_input`、`torchtitan.train.Trainer`、`torchtitan.components.dataloader`，文档钉的是 torchtitan nightly `0.1.0.dev20260701`。

torchtitan main（10-04 是 `0b7536202`，用开工时的 tip，记下 SHA）相对 09-22 移植点 `63c4e9fef` 多了 150 个 commit，至少包括：
- `ParallelDims` 改名 `ParallelismContext`（`distributed/parallelism_context.py`）；
- `prepare_context_parallel_input` 没了，`distributed/context_parallel.py` 现在是 load balancer 类加 `shard_tensors`；
- Tyro CLI 换成 Python config 加载（`config/loader.py`），训练 recipe 重组；
- `TrainingEngine`（`torchtitan/training_engine.py`）带 `finalize_gradients`；
- `TokenDispatcherTransform` / `LoRATransform` 在 `config/transform/`；
- 并行度非法时抛 `ValueError`；
- checkpointer 在 `components/checkpointer/`。
这个列表不全，以实际 import 和运行报错为准。

K3：
- main 上有 `torchtitan/models/kimi_k3/`（`flavors.py`、`pipeline_parallel/`、`vision_encoder.py`、`state_dict_adapter.py`）。
- 集成树 `k3_on_4025` = `e5e47b857`（09-22，base `63c4e9fef`）之后没再 rebase，main 没有的东西（packed MXFP4 QLoRA base、DEP、MoonEP、rl flavors 等）只在那里。本任务不 rebase 集成树。

torch：
- torch nightly ≥ dev20261003 上，FSDP × PP 会在 AttnRes stage 报 "finalize_backward requires manual backward finalization"。
- 修复是 PR-5025 的一行：AttnRes stage 的 forward override 里 `set_manual_backward_finalization(True)`。
- 开工时看 main 是否已合；没合的话，只对 FSDP × PP cell 作为未提交的本地 patch 打上，并在记录里写明。

## 步骤

1. **备份。** 任何 force-push 之前，把上面 7 个分支推成 `backup/<branch>_pre_20261005`。除这些 backup 外不建新分支。worktree 复用 `/tmp/wt_verl_new`（09-22 用的），不要再开新的。

2. **rebase。** 把 `kimi_k3_integration_rebased` rebase 到上游 veRL main tip。
   - 逐个解决冲突并记下。
   - 做完后按文件比较 branch-minus-base 的 diff，确认除冲突处理外字节不变（09-17 rebase 的做法）。

3. **移植到 torchtitan main。**
   - 先测再改：上游 veRL engine（tip）在 titan main 上能不能 import、在文档钉的 0701 nightly 上能不能 import；把结果记下来。
   - 目标是 titan main。已有的兼容探测（`_parallelism_compat_kwargs` 这类）能留就留；不新增为 0701 nightly 维持双路径的代码。
   - 是否让栈的第一个 PR 变成 "engine 跟进 torchtitan main + 文档 pin 升级"，这是我的决定，你只把测量结果和代价写出来。
   - CP 移到 main 的 CP API（load balancer、`shard_tensors`、K3 的 `preprocess_inputs`）。不要恢复 pre-4639 的 module 内 Ulysses 路径。main 的 K3 CP 不完整的话，CP 暂缺并写清缺什么。
   - 只用 titan 的 seam（`TrainingEngine`、transform、component config）。不 monkeypatch torchtitan，不在 fork 里改 titan main。titan 缺的东西记下来交给我。
   - engine 代码是要上游的：默认不加注释，只有代码表达不了的约束才写一行；提交前用 `git diff <base> | grep -E '^\+.*(#|""")'` 审一遍。

4. **重切六个分支。**
   - 用 remap 工具把 split kit 在新 head 上重新生成（原地更新 `split_2026-09-17/`，MANIFEST 里写新的 base / head SHA）。
   - `verify.sh` 每个阶段跑 ast、ruff 和累积测试，对 titan main 跑（这是 reviewer 用的树）。
   - 全绿后从 patch 重建六个分支，force-push 到 fork（已有 backup，这些分支没有挂 PR）。
   - commit message 全部新写：没有 trailer，没有 `owner/repo#N`，没有 github URL（写 PR-5025 这种形式），用 grep 检查整个区间。
   - 六个分支各自 grep 本地标记（`VERL_LOCAL`、`partial_dtensor`、`BFX9`）为零。

5. **GPU smoke**（3 个 GRPO step，`rc=0` 加 rollout-vs-actor log-prob diff）：
   - Qwen3-0.6B 在 titan main 上：fsdp2、tp2、cp2、pp2 四个 cell（reviewer 会跑的那组）。
   - K3 debug 在 titan main 上，只跑 main 具备能力的：fsdp2 × pp2、fsdp2 × pp2 × ep2、pp2 × cp2（CP 移植完成才跑）、save / resume 一对、图片 cell。tp cell 要带 ep ≥ tp。
   - 用 `/root/models/kimi-k3-debug-nt-0922b` 之前，先确认 main 的 `state_dict_adapter.to_hf` 键集合和这个导出一致（1428 个）。不一致就用 main 重新导出，并说明原因。
   - 需要 fork 独有部件的 cell（item 8 QLoRA packed base、DEP）这次不跑，列为未跑。
   - 这些是 smoke：数字不进任何 body，也不和 09-22 的数字配成表（树不同，不可比）。任何 cell 失败或读数明显离群：先定位（同 cell 同 cache 重跑），定位前不写机制、不写进文件，并明确告诉我这个数字暂扣。

6. **环境。** 不要原地升级 `venv_verl`。如果上游的 torch 2.13 / vllm 0.29 和 titan main 要的 nightly 冲突，另建 venv（如 `venv_verl_1005`），记下两边版本和冲突点。

7. **logbook。**
   - 新建 `Raising_PRs/PR_VERL_K3/REBASE_PORT_2026-10-0X.md`（中文）：起止 SHA、backup、冲突处理、移植改动表（titan API 变化 → engine 改动）、阶段测试表、cell 结果、未跑项和原因、需要我决定的问题。
   - 在 `PR_SPLIT_PLAN_2026-09-16.md` 末尾加一段短附录指向它。
   - 按 logbook 规则 commit 加 push：只 stage 自己的文件，先 fetch / rebase，不 force。永远不提交 `Raising_PRs/PR_K3_PARALLELISM/` 下这四个文件：`REPLY_4312_R3922719284_2026-09-18.md`、`REPLY_6840_WHAT_IT_OFFERS_2026-09-18.md`、`SLACK_6840_2026-09-18.md`、`TP_REWRITE_GPU_HANDOFF_2026-09-11.md`。

## 不在范围内

- 不提交 PR、不在任何上游仓库评论。
- 不动 torchtitan 的 PR 分支和 `k3_on_4025`。
- item 11（真实权重 cell）、item 12（body 级成本数字）、item 14 里是否继续叠 titan RL #4680：都留给我。

## 最后回复我（中文，简短）

- 新旧 SHA 和 backup；
- 冲突几处、怎么解决；
- 移植改了哪些 API；
- 六阶段测试结果；
- 各 cell 的 `rc` 和 diff；
- 未跑项；
- 要我决定的问题（至少包括：第一个 PR 是否改成跟进 titan main；vllm 0.29 能否替掉源码构建）。

--- PASTE END ---
