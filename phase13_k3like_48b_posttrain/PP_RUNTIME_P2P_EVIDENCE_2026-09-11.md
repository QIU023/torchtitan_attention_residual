# 运行时自己怎么说这件事：torch main 的两处代码（2026-09-11 查证）

为回答 Tianyu"传输那几个提交为什么不在这个 PR 里、多节点还会不会挂"而查的一手材料。
全部引自 **pytorch/pytorch main HEAD `31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6`**（2026-09-11 拉取）。
本机装的 torch 是 `2.13.0+cpu`（git `cf30153c4c13`），只有下面第 1 条，没有第 2/3 条。

## 1. STATIC 模式的 group communicator 暖机缺口 —— 仍然是 TODO

`torch/distributed/pipelining/schedules.py` L422-L432，`_warmup_p2p()` 末尾：

https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/distributed/pipelining/schedules.py#L422-L432

```
# TODO: STATIC mode group communicator warm-up gap
# The vote protocol above warms up 2-rank sub-communicators
# (used by `_batch_p2p` homogeneous fast-path).  In DYNAMIC mode,
# `_send_meta`/`_recv_meta` (called during `_prepare_forward_infra` →
# `_forward_metadata_inference`) also warm up the *group* communicator
# (used by `_batch_p2p` mixed-op path).  In STATIC mode, metadata
# inference is skipped, so the group communicator is NOT warmed up —
# it will be lazily created on the first mixed `_batch_p2p` call
# (e.g., 1F1B steady-state with both sends and recvs).
# Fix: run `_get_init_p2p_neighbors_ops` + `_batch_p2p` after the
# vote, gated by `not p2p_done`.
```

对我们的意义：新树跑 **STATIC**（路由表定死每跳载荷，不发 metadata），所以恰好落在这条 TODO
点名的未暖模式里；老树跑 DYNAMIC，它的 metadata 反而顺带把 group communicator 暖了。
也就是说在"通信器创建时机"这一条上，**新树比老树更暴露**，尽管它没有老树那股控制面流量。

单节点实测把这个风险压小了：pp8 的 `NCCL_DEBUG_SUBSYS=INIT` 日志显示默认通信器在 init 建、
子组在 DeviceMesh 构建时由 `ncclCommSplit` 建，**调度期间没有任何懒创建**，所以
`_warmup_pp_edge_communicators` 测出来是 no-op。两节点未验证。

## 2. 共享通信器的排序死锁 —— 运行时的 docstring 点名了 "skip connections"

`torch/distributed/pipelining/stage.py` L156-L199，`_build_p2p_direction_groups()`：

https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/distributed/pipelining/stage.py#L163-L169

> Pipeline P2P normally shares a single communicator for both directions, which
> serializes every send/recv in one FIFO. Coalescing makes a single mixed
> send+recv batch deadlock-free, but across *separate* batches (pipeline skew,
> **looped / V schedules, skip connections**) the shared FIFO can still form a
> dependency cycle and deadlock.

这段话把 Elfie 的诊断原封不动写在了运行时里，而且**把"skip connections"和"looped 调度"列为
会触发的形状**——正是 AttnRes + Interleaved1F1B。

## 3. 运行时已经提供了缓解手段：per-direction P2P communicators

同文件 L245-L265：

https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/distributed/pipelining/stage.py#L245-L265

```python
self.p2p_per_direction = (
    dist_config.pipeline_per_direction_p2p
    or dist.distributed_c10d._use_torchcomms_enabled()
)
if self.p2p_per_direction:
    self._downstream_group, self._upstream_group = _build_p2p_direction_groups(group)
else:
    self._downstream_group = group
    self._upstream_group = group
```

- 配置项 `torch.distributed.config.pipeline_per_direction_p2p`，环境变量
  `TORCH_DISTRIBUTED_PIPELINE_PER_DIRECTION_P2P`；用 TorchComms 时自动开启；关闭时两个组都
  别名到 `self.group`，行为逐字节不变。
- 来源：pytorch/pytorch#186173（tushar00jain），落地提交 `420415fa0`，2026-06-16。
  PR 摘要第一句："Pipeline parallelism issues all send/recv on a single communicator …
  it does not remove ordering hazards across separate batches."
- **torchtitan 没有把这个旋钮接出来**（`git grep per_direction upstream/main` 为空）。
- 本机 torch 2.13.0 **没有**这个 API（2.13 的分支早于 2026-06-16）；GPU 盒子的 2.15 nightly
  和 Elfie 的环境需要确认。

## 4. `device_id` 那条要更正

同一个 docstring L173-L175：

> Requires the default process group to be device-bound (e.g.
> ``init_process_group(..., device_id=...)``), which ``split_group`` needs for NCCL;
> torchcomms binds the device automatically.

所以 Elfie 在 `init_distributed` 里绑 `device_id` **不是多余动作**，它正是 torch 自己那条
per-direction 路径的前置条件。错的只是位置：不该由一个 PP 专用环境变量去开一个全局 comms 行为，
应该是 `CommConfig` 的字段（或干脆按 torch 的默认做法）。

另一处容易混淆的：同文件 L144 的 `_warn_if_eager_nccl()` 会在 PP 用 **eager NCCL 通信器**时告警，
建议 `backend="nccl-lazy"`，让 peer 通信器懒初始化、不同 peer 的流量可以重叠。这和 `device_id`
说的不是一回事（一个是默认 PG 是否绑定设备，一个是组内 peer 通信器何时创建），两者不矛盾。

## 5. 结论（可以直接讲给 Tianyu）

1. 老树的挂是**我们自己的控制面流量**压在共享通信器上造成的错序；4312 按构造删掉了它
   （STATIC，没有任何自发的 object P2P 或集合通信）。
2. 剩下的两条风险都是**运行时自己的、已记录在案的**：STATIC 的暖机缺口（第 1 条，至今 TODO），
   和共享通信器跨批次的 FIFO 依赖环（第 2 条，docstring 点名 skip connections）。二者与模型无关，
   任何跑 looped 1F1B 的模型都可能撞上。
3. 运行时**已经有缓解手段**（第 3 条），只是 torchtitan 没接出来。所以两节点 A/B 的第一步
   不应该是跑我们的 `k3_pp_transport`，而是**跑 torch 自己的
   `TORCH_DISTRIBUTED_PIPELINE_PER_DIRECTION_P2P=1`**。
4. 如果那个开关就够：torchtitan 侧需要的只是一个通用小 PR——`init_distributed` 传 `device_id`
   （`split_group` 的前置条件）+ 把 `pipeline_per_direction_p2p` 接进 `CommConfig`，
   惠及所有流水线模型，**和 K3 无关**。
5. 如果还不够：Elfie 的"每条边一个两 rank 组"是同一思路的更强版本（按边而不是按方向），
   届时带着两节点证据提给 `torch.distributed.pipelining`，或作为独立的 torchtitan PR。

无论哪条，都不该塞进 4312。这也把 Elfie 的工作抬高了：她的诊断和运行时 docstring 一字不差，
她加的 `device_id` 正是 torch 自己要求的前置条件。
