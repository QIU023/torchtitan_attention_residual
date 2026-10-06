# MoonEP #4751 diff 复审（`6e3de1b8d`，2026-10-06）

用户："你再审一遍，给我你的看法，这些都是我们昨天改过的"。

审查范围：`db050eb3f..6e3de1b8d` 的 7 个 commit、12 个文件，逐行读过；同时对照 main（`e5c55525e`，比 base 多 57 个 commit）里的 DeepEP、HybridEP 和 #4541 dist_moe。

## 要改的

1. **rebase 时 `moe.py` 的冲突不能只按文本合。**
   - main 的 #4956 在共享专家输出相加前加了 `remat.recompute_needs_tensor(shared_TD)`（注释："The add reads the shared-expert output with bare ops."）。
   - 我们 side-stream 分支里的 `shared_TD` 也是被同一个 `+` 读的，rebase 时要补同一行。
   - `pyproject.toml` 是纯文本冲突：main 在同一行附近加了 `dist_moe`。

2. **三个 MoonEP op 的 remat region 名写死了，没有由 module 提供。**
   - 现状：`ops.py` 用固定的 `"moonep_dispatch"`、`"moonep_experts"`、`"moonep_combine"`，所有层同名。
   - 先例都由 module 提供 region 名：
     - DeepEP 和 HybridEP 的 dispatcher 把 `self.remat_region_name("ep_communication.dispatch")` 和 `"ep_communication.combine"` 传进 `distributed/deepep/` 的函数（`token_dispatcher.py:848`、`:870`）；
     - dist_moe 用 `self.remat_region_name("dist_moe")`，加 `recompute=False`；
     - `LocalTokenDispatcher` 的 docstring 写明 "Dispatchers are parameterless modules so they can own remat region names"。
   - 改法：`dispatch_tokens`、`routed_experts`、`combine_tokens` 各加一个 region 名参数，由 `MoonEPTokenDispatcher` 和 `MoonEPRoutedExperts` 传 `self.remat_region_name(...)`。
   - `recompute=False` 保持不变，因为重放会打乱 plan 表和共享池，dist_moe 也是这么做的。不改数值。

3. **`test_moe_shared_experts_stream` 测不到 hook 存在的理由。**
   - 这是推断，要在 GPU 上验证：删掉 `wait_for_shared_experts_backward`，这个测试大概率仍然通过。原因有三：
     - `_Delayed` 只在前向 sleep，反向不延迟；
     - autograd 在 `backward()` 结束时，会让调用方的当前 stream 等所有 leaf stream；
     - 测试读梯度前还做了 `torch.cuda.synchronize()`。
   - hook 要防的是 FSDP 在反向结束前就读参数梯度，这个场景现在没有测试覆盖。
   - 便宜的补法（单卡）：
     1. 用自定义 autograd Function 让共享专家的反向也 sleep；
     2. 在 MoE 输入的梯度 hook 里（FSDP post-backward 读梯度的位置），在主 stream 上不 synchronize 地 clone 参数梯度；
     3. 再和不开 stream 的结果逐位比。
   - 加的测试先删掉 hook 跑一次，确认会失败，才算真的守住了。

## body（昨天的三处仍未改）

1. 逐参数梯度那句没有噪声基线，也只报中位数（数值验收规则）。要么补 "standard EP 换一种规约顺序" 的逐参数基线，要么只留 routed experts 的 5.4e-3 / 5.5e-3 / 5.8e-3，并写明没有基线。
2. microbench 表 "MoE layer with shared experts" 一行加 "(stream off)"。
3. Design 里的 "forward stream" 改成 "main stream"。

rebase 之后，Test plan 里的计数（CPU 28 passed、GPU 10 passed）要重跑。

## 小问题

- `MoonEPTokenDispatcher.buffer` 只有 GPU 测试里的 `destroy()` 在用，ops 一律走 `current_buffer()`；DeepEP 的 `init_buffer` 不保留句柄。可以删掉这个属性，测试改用 `moonep.current_buffer()[0].destroy()`。
- `TokenDispatcherTransform` 换 config 时用 `assert isinstance(name, str)`：父节点是 list 时会抛一个没有信息的 AssertionError。main 的 `_replace_routed_experts`（dist_moe）处理了 list 父节点和根节点。K3 用不到，但 reviewer 可能会对照。
- `test_moonep_is_refused_outside_kimi_k3` 里的 `disable_cuda_graphs=True` 看起来没用：`validation.py` 里 MoonEP 的拒绝在 CUDA graph 检查之前，会先报错。

## 只作背景，不建议改

- **#4541 dist_moe（10-03 合入）选了不同的接缝：** 独立的 `DistMoeTransform`，experts 放在 `models/common/dist_moe/`，并且不用 dispatcher。MoonEP 的立场站得住：
  - dispatch 和 combine 可以分开，和 DeepEP、HybridEP 一样走 dispatcher 接口；
  - 复用了 `TokenDispatcherTransform` 的 buffer 尺寸推导（`num_tokens_per_microbatch_per_dp_rank / (cp·tp)` 和 `hidden_dim`），这些 `DistMoeTransform` 没有。
  - reviewer 问起时用这两点回答。
- **`validation.py` 在函数内 import `KimiK3Model`：** 这是用户 09-29 定的 "Kimi K3 only"（`ab191a771`），代价是核心 config 依赖一个具体模型；reviewer 问起再说。
- **MoonEP 要求每次 dispatch 的 token 数恰好等于 `num_max_tokens_per_rank`：** 如果 validation 用和训练不同的 batch 形状，第一次 eval 就会报错。body 写了 buffer 是静态的，但没写这个后果。

## 核对过、没问题的

- c2 的原地 `mul_` 先后顺序：权重梯度用 `.float()` 拷贝之后，才原地乘路由权重。
- plan 表用 CPU id 加模块全局字典，和 main DeepEP 的 `_handle_cache` 是同一种做法。
- FullAC、SAC、RegionAC 下不重放 dispatch：GPU 测试用 `_next_plan_id` 计数断言。
- `explicitly_destroy=True` 的理由和 DeepEP `get_buffer` 的 docstring 相同。
- 池在 `_init_self_buffers` 里分配，并且在 `init_buffer` 之后。
- buffer 的 token 数按 PP micro-batch 和 cp·tp 推导。
- slot 梯度池被覆盖之前，有 dispatch 反向的入口栅栏（`MOONEP_ITEMS_10_20` 第 14 条）。
- 新增注释都是一行约束，docstring 一到两行，diff 里没有日志路径或实测数字。

## GPU 会话核对（10-06，`6e3de1b8d`，5060，venv_1003b）

逐条对照代码和 main（`3f087cf15`，#4639 已合）核对，结论全部成立：

- 第 1 条成立：main 的 `d75ddfaac`（#4956）给 shared expert 的加法加了 `remat.recompute_needs_tensor(shared_TD)`，head 的 side-stream 分支没有。集成树新树 `k3_int_20261006a` 的 MoonEP 那一层已经按两条路径都加的方式合过（`b8d0025f7`）。
- 第 2 条成立：`ops.py:181/206/216` 写死 `moonep_dispatch` / `moonep_experts` / `moonep_combine`。DeepEP 和 HybridEP（`token_dispatcher.py:854/876`，`deepep.py:471/591` 收 `remat_region_name` 参数）、dist_moe（`routed_experts.py:175`，`recompute=False`）都由 module 的 `remat_region_name(...)` 给出带 FQN 的名字。RegionAC 用 fnmatch 匹配这个名字（`protocols/module.py:57-68`）。MoonEP 的 region 都是 `recompute=False`，所以改名不影响数值。
- 第 3 条实测成立：去掉 `x_TD.register_hook(wait_for_shared_experts_backward)` 之后，`test_moe_shared_experts_stream` 5 次全部通过；有 hook 时 3 次也通过。
  - 补测试的原型在 kit 的 `local/probe_stream_hook_test.py`（单卡，不用 FSDP）：两个 micro-batch 累加到 `.grad`，第二次在 side stream 上原地加；每个 shared expert 参数挂一个 tensor hook，在加之前 sleep；MoE 输入外面包一个 autograd Function，它的 backward 在主 stream 上 clone shared expert 的梯度（FSDP2 post-backward 读梯度的位置）。
  - 结果：有 hook 时 3 次全部和 stream off 逐位相同；去掉 hook 后 3 次都是 `w13.weight` 不同（`w2` 的累加更早，看不出来）。这个原型可以直接改成 `TestSharedExpertsStream` 的第二个测试。
- 小问题核对：
  - `MoonEPTokenDispatcher.buffer` 只在 `init_buffer` 里赋值，只有 GPU 测试 `test_moonep.py:214` 的 `destroy()` 在读。
  - transform 的 `assert isinstance(name, str)` 在 list 父节点时会抛出没有信息的 AssertionError。K3 的 `RoutedExperts.Config` 父节点是 `MoE.Config`，碰不到。
  - `validation.py` 里 MoonEP 的拒绝在第 134 行，CUDA graph 检查在第 147 行之后，所以 refusal 测试里的 `disable_cuda_graphs=True` 不起作用。
- body：GitHub 上的 live body 就是 v3，三处都还在（`forward stream`、microbench 的 MoE 行没写 stream off、逐参数梯度那句带中位数、没有基线）。改了要重新贴。
- 都没有改到 `moonep_review1` 或 PR 分支上；`k3_moonep_seam` = `moonep_review1` = `6e3de1b8d`，和 main 只在 `pyproject.toml` 冲突。

## CPU 会话复核 GPU 核对（10-06）

- 三个分支仍是 `6e3de1b8d`：GPU 会话只核对了问题，没有修，三条要改的都还在。
- 更正上面最后一句：对当前 main `3f087cf15` 做 `git merge-tree`，冲突是两个文件，`pyproject.toml` 和 `torchtitan/models/common/moe.py`。`moe.py` 的冲突正是第 1 条（#4956 改了共享专家相加那几行），不只是 `pyproject.toml`。
- 探针 `probe_stream_hook_test.py` 的设计对得上 hook 要防的场景：第二个 micro-batch 在 side stream 上原地累加，主 stream 在 MoE 输入的反向里读梯度。可以照它改成 `TestSharedExpertsStream` 的第二个测试，现有测试保留（它守的是前向 join）。

## 已改（CPU 会话，10-06），`moonep_review1` = `16ff9da7f`

- rebase 到 main `3f087cf15`。
  - `moe.py` 按第 1 条合：side stream 拿到的 `shared_TD` 也走 `remat.recompute_needs_tensor`。
  - `pyproject.toml` 两行都保留。
  - 其余 11 个文件的增删行和 rebase 前逐文件一致。
- 新增三个提交：
  - `8de0e6a97`：region 名由 module 给出（第 2 条）；
  - `6741d20a5`：hook 测试，即 GPU 探针的测试版（第 3 条）；
  - `16ff9da7f`：去掉 refusal 测试里的 `disable_cuda_graphs`。
- `buffer` 属性和 `assert isinstance(name, str)` 按讨论不改。
- 本机：`test_moonep_ops.py` 1 passed；改动的文件过了 black 22.12 和 py_compile。其余测试在 Windows 上 import 不了：main 的核心 import 链经过 KDA，需要 CuTeDSL。
- PR 分支 `k3_moonep_seam` 仍是 `6e3de1b8d`。
- body（`PR_BODY_MOONEP_v3_2026-10-05.md`）粘贴区改了三处；Test plan 的计数等下面跑完再填。

## GPU 待跑（树 `moonep_review1` = `16ff9da7f`）

5060 或任意 Linux CUDA 机器：
1. `pytest tests/unit_tests/cpu/test_transforms.py tests/unit_tests/cpu/test_moonep_ops.py -q`，记通过数，填 body 的 Test plan。
2. `pytest tests/unit_tests/gpu/test_moe_shared_experts_stream.py -q`，应为 2 passed。
   - 再临时删掉 `moe.py` 里的 `x_TD.register_hook(wait_for_shared_experts_backward)`（不提交），连跑 3 次：新用例应 3 次都失败，旧用例照常通过。然后恢复。
3. 对改动的文件跑 pre-commit 或 pyrefly。只 stage 目标文件，hook 顺带改的其他文件 `git checkout` 掉。

H100（需要 NVLink multicast）：
4. `pytest tests/unit_tests/gpu/test_moonep.py tests/unit_tests/gpu/test_moe_shared_experts_stream.py -q` on 4 H100s，预期 11 passed。
5. 端到端 8 格（`kit_moonep_perf_2026-10-03/h100_moonep_final_1005.sh`，`TREE` 换成 `16ff9da7f`），共用一份 warm cache：
   - 每组 MoonEP 和 standard EP 的第 1 步应逐位相同。
   - 如果整张表和 `6e3de1b8d` 的一致，body 的表不动。
   - 如果数字变了（main 的 #5026 把 compile region 挪进了 model config，#4956 声明了 remat region，都可能让 K3 debug 的数值整体移动），按数值规则先定位，再换表。定位之前数字一律暂扣。
   - microbench 不用重跑：改名不影响性能。

全部通过之后：用户同意 → `k3_moonep_seam` 用 `--force-with-lease` 从 `6e3de1b8d` 同步到 `16ff9da7f`，填 Test plan 计数，重新贴 body，去掉标题里的 "[DO NOT Review]"，再从 draft 转成 ready。

## GPU 会话跑 `16ff9da7f`（10-06）

5060（venv_1003b，torch 2.15.0.dev20261003）：
1. `pytest tests/unit_tests/cpu/test_transforms.py tests/unit_tests/cpu/test_moonep_ops.py -q`（GPU 隐藏，和 CI 一样）：39 passed。
2. `test_moe_shared_experts_stream.py`：2 passed。临时删掉 `moe.py:743` 的 `x_TD.register_hook(wait_for_shared_experts_backward)` 后连跑 3 次：新用例 `test_input_hook_orders_the_side_stream_gradient_accumulation` 3 次都失败（`w13.weight`），旧用例 3 次都通过。恢复后 2 passed，worktree 干净。
3. pre-commit（`--files` 为 `3f087cf15..16ff9da7f` 改动的 12 个文件，跳过 no-commit-to-branch 和 lychee）：除 pyrefly 外全部通过。pyrefly 的 18 个错误和干净 main `3f087cf15` 的错误集合完全相同，都是环境问题（缺 `torch_checkpointing` 包、attn_gym 和 torch 的 API 版本），没有一个在 MoonEP 的文件里。pyrefly 顺手删了 `hf_datasets/multimodal/utils/video.py` 里的一个 suppression，这是附带改动，已经 checkout 掉。

H100（10-06，同一台 4 × H100，torch 2.15.0a0+git68e0ae4，和 10-05 相同）：
4. `test_moonep.py` + `test_moe_shared_experts_stream.py`：11 passed（114 s）。
5. 端到端：第一次起跑时 `16ff9da7f` 的格子全部在建模型时失败：main 的 #5068（`179c8dcad`）现在总是给 `MixedPrecisionPolicy` 传 `param_dtype_override_fn`，而 torch 68e0ae4 没有这个字段；5060 的 dev20261003 也没有。这台机器的驱动只到 CUDA 12.8，用不了 cu130/cu132 nightly，cu128/cu126 又没有 10 月的包。
   - 改用 kit 的 `local/mpp_shim/sitecustomize.py`：`param_dtype_override_fn=None` 时直接去掉，非 None 时报错；每个进程都会打印提示。K3 的格子传的都是 None，所以计算和新 torch 一样，两棵树也都在同一个 torch 上。
   - 新树 standard（FSDP 4 × EP 4）第 1 步是 8.18811 / 2.1875，10-05 表里是 7.99090 / 2.4219。standard EP 不经过 MoonEP 的代码，这个变化来自 rebase 带进来的 main 提交（新 base 里有 #4881 的零初始化，旧 base `db050eb3f` 没有）。定位之前数字暂扣。
   - 复验脚本 `h100_moonep_recheck_1006.sh` 用同一份 cache 跑三组：新树和旧树 `6e3de1b8d` 各 8 格，加纯 main `3f087cf15` 的 standard 一格；对表脚本 `cmp_recheck_1006.py`。
   - 结果（`results_h100_1006/cmp.txt`）：
     - 新 head 第 1 步，MoonEP 和 standard EP 三种布局都逐位相同：FSDP 4 × EP 4 为 8.18811 / 2.1875，dp_shard 4 × EP 2 为 8.19370 / 2.2031，HSDP 为 8.17808 / 2.2031。
     - 20 步内最大相对 loss 差分别为 1.60e-3、2.03e-3、1.89e-3。
     - FullAC 和 selective 20/20 相同；standard 跑两次 20/20 相同；standard 换到另外两种布局，loss 最多变 1.1e-2。
   - 定位：数值整体移动来自 main。
     - 纯 main `3f087cf15` 的 standard 第 1 步是 8.18811 / 2.1875，和新 head 一样。
     - 旧 head `6e3de1b8d` 在同一份新 cache 上第 1 步是 7.99090 / 2.4219、7.99403 / 2.5469、7.99649 / 2.5156，和 10-05 一样。
     - 旧 head 的 20 步对比在用户 10-06 的话（"为什么每一次review都得重新跑数值"）之后停掉，没有跑。
   - body 粘贴区：端到端表换成新 head 的数，层间差那句改成 1.1e-2，Test plan 改成 39 passed 和 11 passed。

## PR 同步（10-06，用户："注意新的commit不要squash，加到当前PR分支和review分支就行"）

- `k3_moonep_seam` 从 `6e3de1b8d` 用 `--force-with-lease` 同步到 `16ff9da7f`（= `moonep_review1`），旧 head 备份在 `backup/k3_moonep_seam_pre_20261006`。
- 新提交没有 squash，在 rebase 后的 c1 到 c5 之上单独保留：`8de0e6a97` region 名、`6741d20a5` hook 测试、`16ff9da7f` refusal 测试的 flag。
- 还剩用户在 GitHub 上做：重新贴 body（`PR_BODY_MOONEP_v3_2026-10-05.md` 粘贴区）、去掉标题里的 "[DO NOT Review]"、从 draft 转成 ready。

## 更正：换布局不是规约顺序的噪声基线（10-06，用户："改 但是不补测"）

- body 以前用"standard EP 换到另外两种布局，loss 最多变 1.6e-2（新 head 上是 1.1e-2）"做 MoonEP 差距的参照，09-30 的记录也把 EP 2 当成"只换规约顺序"。这不成立：
  - 三种布局第 1 步的 loss 就不同：新 head 上是 8.18811 / 8.19370 / 8.17808，旧 head 上是 7.99090 / 7.99403 / 7.99649。
  - 用 10-05 的第 1 步梯度 dump（`kit_moonep_perf_2026-10-03/floor_1006.py`）算 standard EP 在不同布局之间的逐参数相对差：FSDP 4 × EP 4 对 dp_shard 4 × EP 2，中位数 0.53，routed 专家中位数 1.07；对 HSDP，中位数 1.34。
  - 起点权重（或数据）不同，比的是两个不同的模型，不是规约顺序。
- body 粘贴区删掉了这半句，只保留"standard EP 跑两次 20 步逐位相同，MoonEP 在 FullAC 下和 selective 下相同"。表头"对同布局 standard EP 的最大相对差"不变。
- 真正的噪声基线（两种布局加载同一个 seed checkpoint，逐参数比第 1 步梯度）按用户的话不补测。
- MoonEP 一直如此：第 1 步逐位相同，之后 20 步内差 1.5e-3 到 4.9e-3（09-30、10-01、10-05、10-06 各轮），10-05 测的第 1 步梯度相对差约 5e-3，是 bf16 舍入的量级。
