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
