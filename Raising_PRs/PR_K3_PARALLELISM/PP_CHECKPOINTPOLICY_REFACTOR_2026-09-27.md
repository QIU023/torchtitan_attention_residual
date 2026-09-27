# #4765 / #4764 按 CheckpointPolicy 重构（2026-09-27）

用户的要求：把已有的 #4765 和 #4764，结合之前的估算结果和 titan 已有的 `CheckpointPolicy.MUST_CPU_OFFLOAD` 重构；依赖链变了，PR 分支可能不能直接用。

## 调研结论（决定了怎么改）

- **core 的 `CheckpointPolicy`** 有 `MUST_CPU_OFFLOAD`、`PREFER_CPU_OFFLOAD`，但只在编译路径上生效：
  - eager 的 SAC（`torch/utils/checkpoint.py` 的 `_CachingTorchDispatchMode`）把这两个值当成重算处理；
  - 编译路径上，`torch/_functorch/_activation_offloading` 负责插入拷贝。
- **titan 已有的用法**在 graph_trainer：
  - `tag_all_offloadable_activations` 在 fx 图上把保存标成 `MUST_CPU_OFFLOAD`：按 CPU 预算、从大到小挑、跳过最后一层；
  - `apply_cpu_offload_pass` 插入 `ao.offload`、`ao.reload`、`ao.wait_tensor`；
  - 旋钮：`cpu_offload_budget_gb`、`cpu_offload_prefetch_n_layers`、`cpu_offload_defer_n_layers`，`memory_policy = sac_and_offload`。
- **eager 这边一直没有执行器。** titan 的 AC 跑在 torch_remat 上，它的 `saved_tensors_hooks` 本来就是给 offload 留的接口（带 `SavedTensorInfo` 和 `capture_context`）。所以 ② 的存储就定位成"eager 下执行 `CheckpointPolicy` 的那一层"，和 graph_trainer 的 pass 对应：同一个标记在两条路径上意思一样。

## 分支（都在 fork 的 review 分支上，没新建分支；PR 分支没动）

| 层 | review 分支 | head | 内容 |
|---|---|---|---|
| 4312 | `pp_review5` / `k3_pp_text` | `ffdd169ef` | main `d0f3bbfd6` 上 15 个提交 |
| PR A | `pp_review_optimize` | `7ae870508` | V4 三个提交 + PR A，从 `7814d1f8b` 重叠过来，无冲突 |
| #4765 | `pp_offload_review1` | `7c0f5cd3c` | 存储按 `CheckpointPolicy` 执行，`HostBackend`，`cpu_offload = none/all` |
| #4764 | `pp_balance_review1` | `e6241b78b` | 计划（`cpu_offload = planned`）、`balance`、`RemoteBackend` |

- 重叠：PR A 的四个提交和 09-26 的存储、计划两个提交都没有冲突；最上层 PP、K3、存储相关的 CPU 单测 157 个通过（剩下 4 个是本地 torch 缺 `pipeline_per_edge_p2p`，main 上一样失败）。
- 旧的 `/tmp/rb_ppoff` 是 9 月 4 日留下的 worktree，index 过时，没动它。
- 三个 worktree 的 KDA 本地放宽已撤掉（新 main 的门槛是 SM90 以上，不再需要）；patch 存在 scratchpad 的 `restack/`。
- 测量时三个 worktree 打了不提交的 torch 兼容补丁（`kit_pp_review5_rebase_2026-09-26/pr5_torch_compat_shim.patch`）。

## #4765 改了什么

- `ActivationStorage(device, policy, backends, route=..., prefetch_n_layers=...)`：policy 对每个被保存张量返回 `CheckpointPolicy`。
  - `{MUST,PREFER}_SAVE`：交给 autograd，不跟踪；
  - `{MUST,PREFER}_CPU_OFFLOAD`：搬到 `route(chunk)` 指定的后端，默认 host；后端满了（`put` 返回 None）就留在设备上，计入 `<后端>_full`；
  - 其他值（重算）：`ValueError`，重算由 AC 决定，存储只看到 AC 已决定保存的张量；
  - pin 住的存储（rank store 的 block）不交给 policy。
- `HostBackend(capacity_bytes)`：从 #4764 挪过来，加了容量。
- `cpu_offload_all(skip_layers)`：跳过的层之外一律 `MUST_CPU_OFFLOAD`，对应 graph_trainer 的 `tag_all_offloadable_activations`。
- 预取层数改成参数 `prefetch_n_layers`，对应 `cpu_offload_prefetch_n_layers`。
- K3：`pp_memory.cpu_offload = "none" | "all"`，外加 `cpu_offload_budget_gib`、`cpu_offload_prefetch_n_layers`、`min_tensor_mib`。跳过的是模型最后一层。
  - `BackwardPrefetch`：在本 rank 上一个计算动作开始时，把下一个 stage 反向最先要读的层取回来。
  - 需要 action-list schedule。
- 去掉了 `pp_memory.manager`（只装管理器、不搬任何东西的模式，09-26 实测与 PR A 逐字节相同，没有用户价值）。

## #4764 改了什么

- 计划的算法没变（按时间线注水到目标、窗口覆盖峰值、远程池按跨度拼）。
- 第一步 profiling 改成全部 offload（`cpu_offload_all`），这样只有 offload 才放得下的配置也能起步。
  - 每个动作的峰值加回动作期间不在设备上的最大字节数（`off_device_bytes`），还原出"什么都不搬"的时间线。
  - 偏保守：拷出期间源张量还在，会重复计一次。
- 计划的输出是 `CheckpointPolicy` 加路由：搬的 stage micro-batch 标 `MUST_CPU_OFFLOAD`，路由到 host 或 remote；模型最后一层始终 `PREFER_SAVE`。
- `RemoteBackend(protocol="tcp" | "rdma", device_names=...)`：
  - tcp 时池和 staging 在 pinned host 内存；
  - rdma 时在设备内存；
  - 原来的逻辑是按网卡探测决定池放在设备上，却固定用 tcp 初始化，两者矛盾，这次去掉了。
- 默认 `host_gbps = peer_gbps = 25`。
- `balance` 只能配 `cpu_offload = "planned"`。

## 估算（全是模型值）

见 `PP_PROD_MODEL_PRA_2026-09-27.md`（后台分支做的，`pp_prod_model_2026-09-27.py`）。

- FullAC，PR A 之后，H100：聚合融合时不 offload 也放得下（PP8 × VP4 最大 71.1，HBM 74.5）；现在的代码不融合，最大 83.7，加 offload 也只到 77.1。FullAC 的下一个杠杆是聚合的反向临时量。
- 报告配方：必须 offload；host 50 GB/s 放得下（64.6），25 GB/s 链路饱和，放不下。
- balance：叠在 offload 上没有收益；单独用只在各 rank 不均时削最大值；没有一个场景因为它从放不下变成放得下。

## 5060 实测

**条件：** 8 × RTX 5060 Ti 实测；生产切分（93 层，block 12，pp8 × vp4，dim 2048，seq 2048，M16，FullAC，seed 42）；显存取第 5 步（第 3 到 10 步相差不超过 0.01 GiB），计时取第 8 步的 trace；五个格子共用一条 cache 血缘。结果在 `kit_pp_lowerbound_2026-09-26/results/s4_*`、`s5_*`。

**显存（GiB，每个 rank 的峰值）：**

| rank | #4312 `ffdd169ef` | PR A | `cpu_offload=all` | planned | planned 加 balance |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.42 | 5.52 | 5.35 | 5.52 | 5.52 |
| 1 | 10.68 | 6.02 | 5.66 | 6.02 | 6.02 |
| 2 | 10.68 | 6.00 | 5.64 | 6.00 | 6.00 |
| 3 | 10.78 | 6.80 | 6.44 | 6.52 | 6.52 |
| 4 | 12.00 | 6.11 | 5.93 | 6.11 | 6.11 |
| 5 | 11.48 | 6.15 | 5.79 | 6.15 | 6.15 |
| 6 | 11.47 | 6.11 | 5.75 | 6.11 | 6.11 |
| 7 | 11.20 | 7.11 | 6.74 | 6.75 | 6.75 |
| 最大 | 12.00 | 7.11 | 6.74 | 6.75 | 6.75 |
| 均值 | 11.09 | 6.23 | 5.91 | 6.15 | 6.15 |

- **数值：** 四个格子 10 步的 loss 和 grad norm 都与 #4312 逐位相同。
- **新 base 比旧 base 重：** PR A 在新 base 上是 7.11 / 6.23，09-26 在旧 base（main `9e159aed7`）上是 6.82 / 5.91，每个 rank 多约 0.3 GiB，来自 main 这 32 个提交。所以 #4312 这一列是同 base 重测的。
- **每步搬了多少（第 5 步）：**
  - `all`：每个 rank 往 host 搬 0.88 到 1.5 GiB，峰值降 0.17 到 0.37 GiB；
  - planned：只有超过目标的 rank 7（0.54 GiB）和 rank 3（0.42 GiB）在搬；最大值和 `all` 一样，流量少得多；
  - 加 balance：rank 7 的 0.54 GiB 停到 rank 0；rank 3 分成 0.26 GiB 去 host、0.16 GiB 去 rank 0；
  - 5060 没有 RDMA，池在 rank 0 的 host 内存里，所以 rank 0 的设备峰值不变。在这台机器上，balance 的效果等于 host offload。
- **计划与实测：** rank 7 的计划预计降 0.54 GiB（6.31 → 5.77），实测只降 0.36。原因和 09-26 一样：profiling 用的是第 1 步，那时还没有 AdamW 状态；而且拷出比预算的带宽慢。

**计时（ms，第 8 步，8 个 rank 的均值，每格只抓一次 trace）：**

| 格子 | 一步 | 计算 | 前向 | 反向 | 通信驻留 | 重叠 | 暴露的通信 | 空闲 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| #4312 | 25169 | 7437 | 1764 | 5640 | 17005 | 2380 | 14625 | 3107 |
| PR A | 24846 | 7368 | 1733 | 5601 | 16616 | 2260 | 14356 | 3122 |
| `cpu_offload=all` | 25346 | 7366 | 2670 | 7215 | 17370 | 2462 | 14908 | 3072 |
| planned | 25107 | 7370 | 1737 | 5748 | 17015 | 2370 | 14645 | 3092 |
| planned 加 balance | 25079 | 7373 | 1773 | 5710 | 17009 | 2363 | 14646 | 3060 |

- 计算时间几乎不变（7.37 到 7.44 秒）。
- `all` 的前向和反向动作各长约 0.9 秒和 1.6 秒，是等拷出和取回；一步只多 2.0%，因为 5060 上一步大部分时间在等通信。
- planned 只搬两个 rank，一步多约 1%；rank 7 在关键路径上。
- 每格只有一次 trace，1% 上下的差别没有噪声底，不能下结论。

## balance 按所有 PP rank 之间的差来看（用户 09-27 指出）

用户："Balance不是应该看所有pp rank cache on之间的内存区别吗？pp8vp4情况下（本地自己改recipe）"。上面那格 planned 加 balance 测不出 balance：一是 5060 没有 RDMA，池在对端 host 内存里，目的 rank 的 GPU 不涨；二是 FullAC 下 rank 之间差得少，而且最重的是 rank 7（头部和最后一个反向），不是报告里 warmup 造成的那种不均。

**本地改的 recipe（不进分支）：** pp8 × vp4（93 层，block 12），cache 开，关 AC，dim 1024，seq 512，M16；本地补丁 `kit_pp_lowerbound_2026-09-26/balance_probe_local.patch`：目的 rank 在 GPU 上占住池的大小（模拟 RDMA 设备池，只看显存），以及 plan 只停到别的 rank、不去 host 的开关。8 × 5060 实测，第 5 步，GiB，四格 8 步 loss 和 grad norm 都与 PR A 逐位相同。

| rank | PR A | 只 balance | 只 host offload | 两者都开 |
|---:|---:|---:|---:|---:|
| 0 | 10.22 | 10.22 | 9.95 | 9.95 |
| 1 | 10.66 | 10.28 | 10.02 | 10.27 |
| 2 | 10.30 | 10.41 | 9.80 | 9.80 |
| 3 | 11.19 | 10.46 | 10.20 | 10.20 |
| 4 | 9.93 | 9.92 | 9.52 | 9.52 |
| 5 | 9.57 | 9.76 | 9.57 | 9.57 |
| 6 | 8.88 | 9.99 | 8.89 | 8.88 |
| 7 | 8.89 | 10.44 | 8.87 | 9.29 |
| 最大 | 11.19 | 10.46 | 10.20 | 10.27 |
| 最小 | 8.88 | 9.76 | 8.87 | 8.88 |
| 均值 | 9.96 | 10.18 | 9.60 | 9.69 |
| rank 间差 | 2.31 | 0.70 | 1.32 | 1.39 |

- **不均的来源：** 关 AC 后 warmup 的不均出现了，前面的 rank 重（rank 3 最重），后面的轻，和报告描述一致。
- **只 balance：**
  - rank 3 每步停 2.64 GiB 到 rank 7，rank 1 停 1.38 GiB 到 rank 6，rank 2 停 0.17 GiB 到 rank 5；
  - 目的 rank 相应上升（rank 7 +1.55，rank 6 +1.11）；
  - rank 间差从 2.31 降到 0.70（−70%）；总量不降，均值 +0.22，是池的冗余和 staging；
  - rank 0 高于目标但没搬，因为能接收的 rank 都已经满到目标。
- **只 host offload：** 高于均值的 rank 把超出的部分送 host（rank 3 每步 3.67 GiB），最大值压得更低（10.20），均值降到 9.60，但 rank 间差还有 1.32。
- **两者都开：** plan 几乎全选 host，只有 rank 1 的 0.42 GiB 停到 rank 7。因为这台机器上对端链路按 2 GB/s（TCP）预算、host 按 10 GB/s，多数条目的窗口只够 host。H100 上两者的比例不同（host 25 到 50 GB/s，网卡给 PP 约 25 GB/s），选择也会不同。rank 1 比只 offload 时高 0.25 GiB，是模拟 RDMA 时源 rank 的 staging 缓冲。
- **plan 用的 profile 偏高约 1.5 GiB：** 第一步全部 offload，拷出中的张量既还在设备上、又已计入 `off_device_bytes`，被算了两次。plan 因此偏保守，但 rank 的高低顺序不受影响。可以改成拷贝完成、源张量释放时才计入。
- **plan 的限制：** 一个源只有一个目的地；目的地不能同时是源；目标是各 rank profile 峰值的均值。

## 下一步

- 三个 review 分支已推到 fork（`pp_review_optimize` `7ae870508`、`pp_offload_review1` `7c0f5cd3c`、`pp_balance_review1` `e6241b78b`）。
- PR 分支 `k3_pp_offload`（`7b43e1d01`）、`k3_pp_balance`（`37967996b`）还在旧 base 上，也就是旧版本的备份，等用户说再同步；body 在 `PR_BODY_4765_2026-09-27.md`、`PR_BODY_4764_2026-09-27.md`。
- 单测：#4765 的测试计划 6 个文件 76 个通过；#4764 的 8 个文件 89 个通过，其中 mooncake TCP 远程停放那个是实际跑的。
- PR A 可以单独开 PR（要多一个分支，需要用户点头）。
- H100：RDMA 路径、真实 PCIe 带宽；FP8 后端没做。
