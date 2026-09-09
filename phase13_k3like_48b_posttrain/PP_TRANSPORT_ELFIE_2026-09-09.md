# Elfie 的 PP 多节点修复：解读、与新树的关系、要做的事（2026-09-09）

来源：pytorch/torchtitan#4281 issuecomment-5608619533（elfiegg，2026-09-09 20:55Z），分支 `elfiegg:fix/pp8-neighbor-p2p-metadata` = 老树 `k3_pr_classified_v2`（`a81722d10`）+ 两个提交：`2c81784c1 distributed: isolate PP neighbor P2P transport`（+680，含 271 行传输测试）、`344fccf17 components/optimizer: materialize missing state for DCP`（+58/-16）。

## 她说了什么

1. 想建一个沟通渠道（进度、阻塞、优先级）。
2. 用"更新过的分支"（仍是老树）在 GB200 两节点 PP=8 仍然遇到关键的 PP 相关 NCCL 挂起。
3. 次要问题：checkpoint 恢复时 Adam 缺第 0 层 residual 的优化器状态。
4. 她的本地修复把 PP 传输从"metadata 对象、模式投票链、部分张量 P2P 都走整个 PP 的 NCCL 通信器"改成：metadata 走 PP 副本内的 Gloo（CPU）组；每条边（i<->i+1）一个专用的两 rank NCCL 组；static/dynamic 决定改成一次全 PP 对齐的 all-reduce；组在 DeviceMesh 建其他 NCCL 组之前按确定顺序 eager 创建（绑定 device_id 后 `new_group` 走 ncclCommSplit）。
5. 诊断链：AttnRes 需要动态 metadata P2P → metadata 用整个 PP 的 NCCL 通信器 → PP8/1F1B 暴露通信器初始化/顺序 hazard。
6. QAT 功能上能跑，需要的是精度研究而不是功能验证。

## 她的实现（`pipeline_parallel.py` +335）

- `ParallelDims._create_pipeline_neighbor_groups`（opt-in：`TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1`）：每个 PP 副本一个 Gloo metadata 组 + 每条物理边一个两 rank NCCL 组，在 mesh 之前创建（"ProcessGroupNCCL 要求全局一致的创建顺序"）。
- `_create_pipeline_transport_groups`：把物理边组映射到逻辑 stage 边（支持 looped/interleaved：同 rank 的相邻 stage 不需要组）。
- `_NeighborP2PPipelineStage(PipelineStage)`：重写 `get_fwd/bwd_recv_ops`、`get_fwd/bwd_send_ops`，全部张量 P2P 走边组（`group_peer` 0/1）；**只允许相邻 stage** 的收发，非相邻直接 raise；`_send_meta/_recv_meta` 用 `send_object_list(..., group=Gloo, device=cpu)`。
- `_configure_neighbor_p2p_schedule`：把调度的 `_warmup_p2p` 换成一次 `all_reduce(MIN)` 投票（只对这种 stage 生效，绑定到该 schedule 实例）。

## 与 torch 运行时和新树的关系

- torch 2.15 nightly 的 `schedules.py` 自己已经有"投票协议暖两 rank 子通信器"，并留了 TODO："STATIC mode group communicator warm-up gap"——STATIC 模式跳过 metadata 推断，PP 组通信器要到 1F1B 稳态第一次混合 send/recv 才 lazy 创建；"Fix: run `_get_init_p2p_neighbors_ops` + `_batch_p2p` after the vote"。这是上游自己承认的 hazard，与她的诊断同源。
- 老树 8/27 的修复 `d1ec535d1`（`_warmup_pp_edge_communicators`，35 行，正是 TODO 开的方子）**没有** cherry-pick 到 #4312（`k3_pp_text` = `a3be242bf`，diff 里没有任何 eager/warmup/new_group）。今天已在 `pp_review4`（`fd7ff7400`）上无冲突落下，调用点在 `pipeline_llm` 建完 schedule 之后，K3 入口走 `pipeline_llm`。
- 新树的 `AttnResPipelineStage` 线上不发 metadata（grep 不到任何 send/recv/object 调用）：路由表由层布局决定，hop 载荷在第一次发送前就定了 → STATIC；DYNAMIC 那条 hazard 在新树按设计不存在，剩下的只有 TODO 那条 STATIC gap → warmup 覆盖。两节点是否还挂，只能由她在 GB200 上跑 `pp_review4` 验证。
- 她的方案更强（按构造消除共享通信器，而不是靠 warmup 的时机），但要求"只有相邻 stage 收发"——我们的设计也满足（block stack 逐 hop 相邻转发，梯度走调度自己的反向 P2P）。如果两节点仍挂，下一步是让 `AttnResPipelineStage` 继承她的边组路由；而正确的归属是运行时本身（投票协议已经在内部建两 rank 子通信器）。
- 优化器：torch 的 `_init_optim_state`（DCP `get_optimizer_state_dict` 用）是"`optim.state` 非空就整体跳过"，从未拿到梯度的参数就没有状态，恢复时缺键。老树第 0 层的 `attention_res_proj` 就是这种参数；新树第 0 层没有这个投影（`attention_res_proj=None if layer_idx == 0`），当前不触发，但缺陷是通用的（PP 下未用参数、LoRA/MTP 变体），她的逐参数补状态应作为独立小 PR 进 main（带她的测试）。

## 对 RFC 和给 Tianyu 的设计文档的实质影响

- 多消费者梯度路由（RFC 的主体）不变，而且被她的失败**加强**：任何自带控制面流量的设计在两节点 1F1B 下就是这样死的；RFC 新增第 6 条"能活过多节点 1F1B 的传输"：运行时三条通路（metadata 对象 P2P / 同质批的两 rank 子通信器 / 混合批的组通信器）、两种 hazard、四个运行时侧的修法（TODO 的暖机、metadata 走 CPU 组、投票改集体、运行时自己拥有按边的两 rank 组）、以及多消费者机制必须守的不变量（不加调度之外的 P2P、hop 载荷静态、只相邻）。
- 设计文档新增 4b 节：老树为什么挂、新设计为什么没有 metadata 在线上、剩下的 STATIC gap 与 warmup、两节点验证待做、若仍挂则边组是下一步且应由运行时拥有。

## 要做的事

1. 回复 #4281（草稿见下）并建渠道；把她从老树引到新树：base #4527，TP/SP #4499（`k3_tp_sp`），QB #4412（`k3_qb`），PP #4312（`k3_pp_text`；两节点请用 `pp_review4`），CP #4500（fegin/Shuhua 的栈）。老树 #4281 冻结，不再修。
2. 请她在 GB200 两节点跑 `pp_review4` 的 PP8（先 `--parallelism.pipeline_parallel_schedule Interleaved1F1B` 的 `kimi_k3_debugmodel_pp8_vp4` recipe），报告是否还挂；单机验证结果见本目录 `pp_warmup_verify.log`。
3. 请她把优化器状态补齐作为独立 PR 提到 main。
4. 把传输条目并进 PyTorch pipelining 的 issue/RFC。

## 回复草稿（英文，贴 #4281 或渠道首条）

--- PASTE BEGIN ---

Thanks Elfie, and yes to a channel; I'll set one up and add you and the maintainers.

The branch you tested is the old full tree (#4281), which is frozen: everything moved to a sequence of small PRs on main and I won't fix the old tree. Base: #4527 (Shuhua's multimodal spmd declarations). On it: TP/SP #4499 (`QIU023:k3_tp_sp`), Quantile Balancing #4412 (`QIU023:k3_qb`), pipeline #4312 (`QIU023:k3_pp_text`, the text decoder on `torch.distributed.pipelining`'s own runtime), CP #4500 (fegin's stack).

Your diagnosis matches what the runtime itself says: `schedules.py` carries a TODO for the STATIC-mode group-communicator gap (the mixed send/recv batch creates the PP group communicator lazily in 1F1B steady state; fix prescribed: `_get_init_p2p_neighbors_ops` + `_batch_p2p` after the vote). The new pipeline has no metadata on the wire at all (routing tables fix every hop before the first send, so the stage runs STATIC), so the DYNAMIC-mode part of the hazard does not exist there; the STATIC gap does, and `QIU023:pp_review4` (`fd7ff7400`, #4312 plus one 35-line commit) creates every edge communicator right after the schedule build. Single node: dp1 and pp2 bitwise with and without it, pp8 x vp4 runs. Could you run PP8 across your two GB200 nodes on `pp_review4` (`torchtitan_recipes.tests.features:kimi_k3_debugmodel_pp8_vp4`)? If the late edges still hang there, your per-edge two-rank groups are the next step and I'd put them into the `AttnResPipelineStage` subclass; the design already sends only between adjacent stages, so it fits your transport as is.

The optimizer-state fix is independent of the tree: torch's `_init_optim_state` skips every parameter once any state exists, so a parameter that never received a gradient has no state at save time. The new tree's layer 0 has no residual projection, so it does not hit it today, but the hazard is generic (an unused parameter under PP, LoRA, MTP); would you open it as a standalone PR against main with your test? I'll review it right away.

--- PASTE END ---
