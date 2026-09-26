# PR 4381 body v3（draft PR，`k3_pp_mm`），2026-09-26 rebase 到 4312 的 `7814d1f8b` 之后

## 状态（不粘贴）

- **分支：** `k3_pp_mm` 和 review 分支 `dep_review1` 都已推到 `232834a4d`，即 4312 的 head `7814d1f8b` 之上的 4 个提交。
  - 旧 head `94e6e7116`（在 09-23 的 `3902077ca` 上）备份为 `backup/k3_pp_mm_pre_20260926`。
  - fork 上 `dep_review1` 原来是 09-22 的 `191b31bc0`，它早已备份为 `backup/k3_pp_mm_pre_20260923`。
  - 推送依据是"draft PR 直接同步"（用户 09-23、09-25）。推送前通过 GitHub API 确认 #4381 是 draft，head 是 `94e6e7116`。
  - 推送后 GitHub 显示 19 个提交、27 个文件：4312 的 15 个加上这里的 4 个。
- **rebase 里改了什么：**
  - **`pipeline_parallel/__init__.py` 冲突。** 4312 现在自己用 `get_module_fqns_per_model_part` 推导切分，再交给 `pipeline_llm`，旧的 `pipeline_with_first_last_stage_modules(..., return_split=True)` 已经没有了。解决办法：DEP 先把 vit_dep 的切分写进 `kwargs["parallelism"]`，4312 的推导看到切分已经给定，就跳过自己的推导。
  - **`_with_vit_dep_split` 改用 `copy.copy` 再赋值，不再用 `dataclasses.replace`。**
    - 原因：4312 的 `ParallelismConfig.__post_init__` 不允许切分和 `pipeline_parallel_layers_per_stage` 同时存在，而 `replace` 会重跑这个检查。
    - 本地探针（`dep_knob_probe_2026-09-26.py`，29 层、pp4、knob=2）：推导出 16 个 stage，第一个 stage 是 tower 和 embedding；copy 上保留 knob=2；调用方的配置原样不动；改之前的 `replace` 写法在新 base 上直接报这个错。
  - **`test_integration_test_definitions.py` 冲突。** 4312 已把两个 PP 格子合成 `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4`。解决办法：保留 4312 的集合，加上 DEP 的 `kimi_k3_pp4_vp2_vit_dep`，不再列 `kimi_k3_pp4_vp4`。
  - 其余自动合并。`model.py` 的合并逐段看过。DEP 的文件没有用到 4312 改成私有的函数名。
- **验证（2026-09-26）：**
  - CPU，下面粘贴区里那条命令：98 passed，1 skipped（本机缺 `fla`，与 DEP 无关；收集时需要 `--ignore=tests/unit_tests/cpu/test_torch_checkpointing.py`，因为本机缺 `torch_checkpointing`）。
  - `test_optimizer_param_groups.py` 单独跑：24 passed。更宽的 `-k`（加上 optimizer_param_groups、vit_dep、dep_bubble）：158 passed，1 skipped。
  - GPU smoke，4 × RTX 5060 Ti：B200 的 `kimi_k3_debugmodel_pp4_vp2_vit_dep` 格子跑 10 步，rc=0，loss 8.11403 → 6.17197。没固定 seed，只说明能跑通。本地放宽了 `kda.py` 的 SM120 检查，不在 diff 里。
    - 日志里 DEP 的计数与 09-22 相同：2/2 个计划内的 encode 在空闲时隙里跑了，4 个在最前面跑，2 个留在原位跑。
    - 推迟的 tower 反向：6 个在计划的时隙里跑完，没有一个留到 step 末。
- **没有重新测的：** 粘贴区测试计划第 3 条（bubble 开和关对比、tower 前向逐位一致、loss 从第 3 步分开）是 09-22 在旧 base 上测的，这次没有重跑。要保留这条，得先在新 head 上用同一份暖缓存、seed 42 重跑这一对。
- **与 v2 相比：**
  - "three commits" 改成 "four commits"；
  - Design 加了 optimizer 放宽那个提交（09-23 加入，v2 没写）；
  - 测试计划的数字改成 98，并单列 optimizer 的测试。
- **标题：** GitHub 上现在的标题是 `[DO NOT review, pending K3 text PP merging] [Kimi K3] Add K3 MoonViT DEP support to schedule ViT stages into text LLM PP bubbles`，和 v2 文件里的标题不同，以线上为准，不改。
- **没读到线上 body：** 用 GitHub API 读线上 body 的请求被权限拦下了。粘贴前请对照一下线上 body，确认 v2 之后你有没有改过。

--- PR 4381 body v3: PASTE BEGIN ---

Draft, stacked on the text-side PP PR (#4312): the diff tab shows that PR's content too, so review the four commits on top. It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3: the vision tower takes a pipeline stage of its own ahead of the text stages, so its compute leaves the critical path of the stage that owns the embedding, and its encodes are placed around the schedule's own actions: ahead of the forward that reads them (`vision_dep.prefetch`), or in the schedule's idle intervals with the tower's backwards deferred to the intervals after backward actions (`vision_dep.bubble`).

### Design

- The tower stage is the split alone, no core change. `vit_dep_split` puts `[tok_embeddings, vision_encoder]` on the first stage and spreads the layers over the remaining stages through core's own `_generate_llm_fqn_per_model_part`, so the stage count the schedule sees is unchanged and the tower stage comes out of the text stages' budget. The embedding rides with the tower because the splice needs the token ids, which only the first stage receives. The block routing accepts a stage with no layers.
- The knobs are a `vision_dep` record on the model config: `enabled`, `prefetch`, `bubble`, `bubble_cost_ratio`, `bubble_max_pending`. They change the split every rank applies, so they belong to the model rather than to the command line, and the model config tree is off the CLI.
- `encode_images` is the tower's forward on one micro-batch's images, and `forward` takes `vision_embeds`, so the schedule's forward and the ahead-of-time encode cannot drift. An encode issued before the pipeline calls the stage gathers the FSDP parameters itself; the stage's own forward reshards them as its policy says. The first step encodes inline, because FSDP2 builds its state in the root module's first forward and an encode before that makes the tower a root of its own.
- `VisionDepPipelineStage` subclasses the AttnRes stage and hands the runtime each action as it completes, which is how a placement fires after the action it is anchored to. An anchor is an action's identity rather than a slot index, because the index does not survive lowering: the runtime walks `pipeline_order_with_comms`, which inserts sends and receives and holds no idle entries.
- The plan (`dep_bubble_plan.py`, pure) is read off `pp_schedule.pipeline_order[rank]`, so every rank derives the same placements with no collective and none reaches a vision collective the others do not. The first `pp` encodes run upfront, as the report prescribes; later ones go in the last idle slot whose accumulated budget covers `bubble_cost_ratio` text-stage actions, so the features stay resident as briefly as the budget allows.
- The tower's backward is cut at the splice (`dep_backward.py`): the features are spliced in through a detached stand-in whose gradient is captured and replayed at a later slot. Whatever the slots did not take is drained at step end, because a deferred backward that never runs leaves the tower without that micro-batch's gradient and raises nothing.
- `OptimizersContainer` skips a param-group pattern that matches nothing on one model part and refuses only a pattern that no model part on the rank matches. The tower stage holds none of the matrices the per-head DistMuon recipe (#4596) assigns to Muon, and before this change such a stage raised.

### Test plan

- `pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline or cli or integration_test" -q` (98 passed). The bubble tests cover the placement invariants, a backward-anchored placement firing, the run-ahead, and the deferred backward: a cut and replayed tower backward is bitwise the gradient the inline one produces, out-of-order replays accumulate like one pass, the pending bound changes when rather than whether a gradient runs, and nothing is lost when no slot ever comes.
- `pytest tests/unit_tests/cpu/test_optimizer_param_groups.py -q` (24 passed), including a pattern that matches nothing on one stage.
- The `kimi_k3_pp4_vp2_vit_dep` cell in the B200 suite, which derives its split from `vision_dep` rather than spelling one out.
- pp4 x vp2 with the bubble on and off, same seed and batch on one warm compile cache: the tower's forward is identical either way; the losses separate from step 3 by the order the tower's gradients accumulate in. The hundred-step table on H100 goes here.

--- PASTE END ---
