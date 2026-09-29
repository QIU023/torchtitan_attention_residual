# DEP 传输修复的 GPU 验证（8 × RTX 5060 Ti，2026-09-29）

验证对象：`k3_pp_mm` = `dep_review1` = `55e4274c4`，即 main `5dc97a3e7` 上的 3 个提交。

- 用的是默认的懒加载，没有设 `CUDA_MODULE_LOADING`。
- 本机 torch 较旧，打了本地兼容补丁 `kit_dep_gpu_2026-09-29/shim_5dc97a3e7_pp.patch`，只改 PP schedule 的参数和 `pipeline_per_edge_p2p` 两处，跑前打上、跑后撤掉。
- 脚本、补丁、日志和比对输出都在 `kit_dep_gpu_2026-09-29/`，结果只进 logbook。

## 结论

- **死锁修好了。** 仓库里的 B200 格子 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep`（bubble 模式）在默认设置下跑完 10 步。
- **传输在数值上是对的。** 只要两边的初始权重相同，K2.5 和 bubble 两种模式都和 DEP 关逐位一致：前 3 步的 loss 相同，第 1、2 步每个参数的梯度都相同，包括塔。
- **要 CPU 那边修两处：**
  1. **塔副本初始化消耗了全局随机数**，所以同一个种子下，DEP 开和 DEP 关的模型初始权重不同。这一处必须修，否则 DEP 开和 DEP 关没法在同一种子下比较。
  2. **bubble 模式 GPU 测试的一处容差偏紧**（不是卡死，见下面"测试"一节）。

## 测试

- **CPU 测试**（`test_kimi_k3_dep_plan.py`、`test_kimi_k3_vision_dep.py`、K3 PP 的 stage/block_grads/layout 测试、`test_integration_test_definitions.py`）：47 passed，28 subtests passed。
- **新的 NCCL GPU 测试** `tests/unit_tests/gpu/test_kimi_k3_vision_dep.py`（4 卡）：
  - **K2.5 用例通过。**
  - **bubble 用例失败，但不是卡死**：
    - 失败点：rank 0 上，第 1 步之后塔的 `proj` 梯度的一处 `torch.testing.assert_close(grad, expected)`（`test_kimi_k3_vision_dep.py:258`），用的是默认容差。
    - 偏差：12 个元素里有 1 个超出，相对差 1.44e-6，允许值是 1.3e-6。这个 toy 模型此时梯度量级约 1.4e17，所以绝对差显示为 2e11。
    - 原因：bubble 模式下塔的反向放到了另一个 rank 上，fp32 的求和顺序变了。
    - 其余三个 rank 都通过了。rank 0 抛出断言后，测试框架在收尾时，其余 rank 在 barrier 里等它，于是整体卡到 300 秒超时，日志里看不到断言的内容。rank 0 的原始异常是用一个探针（`kit_dep_gpu_2026-09-29/test_dep_gpu_probe.py`）打印出来的，见 `logs/gpu_bubble_rank_exceptions.txt`。
- **B200 格子**（仓库里的 recipe，不设种子，10 步）：rc=0。
  - 计划日志：4 个带图像的 micro-batch，2 个在调度开始前编码，2 个放进空闲槽；反向 0 个放进空闲槽，4 个在调度结束后做。
  - 第 1 到 4 步的 loss：8.07054、7.08332、5.10681、4.74786。

## 数值

条件：同一份预热缓存，`--debug.seed 42 --debug.deterministic`，每格 3 步，导出第 1、2 步所有参数的梯度。

| 对比 | 第 1 步 loss | 梯度逐位相同（第 1 步；第 2 步） |
|---|---|---|
| DEP 关，跑两次 | 8.05527，8.05527 | 全部；全部 |
| K2.5，跑两次 | 8.14325，8.14325 | 全部；全部 |
| DEP 关对 K2.5（分支原样） | 8.05527 对 8.14325 | 0；0。初始权重就不同：stage 0 那段 249 个参数只有 67 个相同，另一段 216 个只有 58 个相同 |
| K2.5 对 bubble（分支原样，两边初始权重相同） | 相同 | 文本参数全部；塔 18/22（最大相对差 6.4e-5）。第 2 步全部不同，是塔的这点差传了下去 |
| DEP 关对 K2.5（本地探针：副本初始化包进 `fork_rng`） | 相同 | 全部；全部（初始权重也全部相同） |
| DEP 关对 bubble（同一探针） | 相同 | 全部；全部 |

打上探针后，前 3 步的 loss 三格一样：8.05527、7.00382、5.36156。

**原因**（`pipeline_parallel/__init__.py` 的 `_vision_replica`）：
- 训练引擎先 `set_determinism` 设好种子，再调 `model.pipeline(...)`。`pipeline_kimi_k3` 在里面建塔的副本，并执行 `replica.init_states()`，这一步消耗了 CUDA 的随机数。
- 之后每个 model part 才 `to_empty` 加 `init_weights`，这时随机数状态已经被推进了一截。
- 每个 pp rank 都建一份副本，所以两段的权重都受影响。
- 副本的权重在第一步开头就会被 `_sync_weights` 用 stage 0 的塔覆盖，所以这次初始化本来就用不上。

**修法**（由 CPU 那边定）：
- 已验证的做法：把 `replica.init_states()` 包进 `torch.random.fork_rng(devices=[device])`，本地探针补丁就是这样改的，见 `kit_dep_gpu_2026-09-29/rng_fork_probe.patch`。
- 另一个做法是完全跳过这次初始化，但要先确认副本里没有需要 `init_states` 才能建出来的缓冲。
- 可以补一个 CPU 测试：同一个种子下，分别建开 DEP 和不开 DEP 的模型，初始参数应当逐位相同。

## 显存（每个 rank 第 2、3 步的峰值，GiB）

| rank | DEP 关 | K2.5 | bubble |
|---:|---:|---:|---:|
| 0 | 0.365 | 0.371 | 0.371 |
| 1 | 0.366 | 0.377 | 0.377 |
| 2 | 0.349 | 0.374 | 0.347 |
| 3 | 0.349 | 0.380 | 0.354 |
| 4 | 0.344 | 0.352 | 0.352 |
| 5 | 0.335 | 0.349 | 0.349 |
| 6 | 0.347 | 0.342 | 0.342 |
| 7 | 0.339 | 0.339 | 0.339 |
| 最大 / 平均 | 0.366 / 0.349 | 0.380 / 0.360 | 0.377 / 0.354 |

- DEP 开以后，每个 rank 多出 0.01 到 0.03 GiB，来自塔的副本、fp32 累加器和收发缓冲。
- bubble 模式下带图像的 rank 2、3 比 K2.5 低，因为有两个编码挪进了空闲槽，步首要存的特征少了。

## Trace（第 10 步，每种配置一步）

| | DEP 关 | K2.5 | bubble |
|---|---:|---:|---:|
| 一步的时长，各 rank 平均（ms） | 3079.6 | 3106.7 | 3079.1 |
| 计算，各 rank 平均（ms） | 82.3 | 82.7 | 82.8 |

- 这台机器走 PCIe，debug 模型又很小：一步里计算约占 3%，其余全是通信和空闲。
- 塔的计算量在这里显不出来，所以这组 trace 回答不了"视觉计算有没有落进空闲段"，只能说明三种配置的步时看不出差别。
- §8 的第 3 项（看视觉计算的位置，比较步时）还要到 H100 或 B200 上、用真实尺寸去测。

## 还没做的

- 真 B200 上的 CI 格子，以及 body 需要的 H100 数字。
- 上面两处修好以后，我在 5060 上用同一套脚本（`t5c.sh`、`t5d.sh`）复测，不再需要本地探针补丁。
