# MoonEP 证据总档(2026-08-28,2×H100 NV18)

判据 `EVIDENCE_METHOD_2026-08-25.md`。树 `k3_on_4025`(flavor 提交 `4961dec31`,
实现:dispatcher/experts/fake 由 CPU 盒盲写、按 moonep 真实合同重构)。
库:MoonshotAI/MoonEP master @ `2bd860b`("Support MXFP4 expert weights in
remote prefetch"——与 QLoRA 线的上游交点),cutlass-dsl 4.4.2,CUDA 13 构建。
raw 日志:`raw_h100nvl_0828/`(moonep_st/smoke/probe/hot_evidence/hot_mod4)。

## 一、链条(全绿)

| 阶段 | 结果 |
|---|---|
| CPU 假世界门(本地 PCIe,无 moonep 包) | 4/4:双线程假集合、测试指定复制映射、slot-grad 归约、稠密参照前向+梯度 |
| 官方自测(真内核,2 rank) | **6/6**:planning/dispatch 12/combine 14/e2e/grad_reduce 12/prefetch 14(e2e 需 `torchrun -m tests.test_e2e`,pytest 收集器双重 init PG 是测试写法问题) |
| titan 集成冒烟 ep2×fsdp2 | **一次通过**:12.47486→9.19460,exit 0;ON-BOX 三疑点(Buffer 组绑定/dtype/cu_seqlens)零触发 |
| 逐参数梯度探针 vs standard(同 seed 3 步) | 榜首 2.1–2.6e-1 相对差,全为 1e-4 量级噪声带参数,**无 routed_experts 族**——token 守恒+反向到达成立(反例参照:未修 maep 的 0.998 专家梯度全丢) |

## 二、强制不均衡证据(临时 hack:路由 ids 置 0 / %4;方法不入库,可复刻)

| 格 | 载荷 | 末步 tps | 槽活动 |
|---|---|---|---|
| moonep 强热 | 全 token→expert0 | 345 | rank0 槽=expert16,rank1 空 |
| standard 强热 | 同 | 336 | — |
| moonep 常态 | 真路由 | 347 | — |
| moonep mod4 | 全 token→experts0-3 | 333 | **同上,纹丝不动** |
| standard mod4 | 同 | 341 | — |

**已证**:复制路径(prefetch→槽计算→reduce_grad)真内核实训激活且稳定;
复制机械零额外开销(345≈347)。

**Open 项裁决(2026-08-28 晚,planner 源码考 @2bd860b,无需硬件)**:
"planner 与实时负载无关"假设**被否**。源码事实(moonep/planning.py):

- 计划**无状态、逐步现算**:`plan=None` 的每次 dispatch 触发一枚 rank0 协作
  核(planning.py:519,610),从本步 all-gather 的 `tokens_per_expert` 现算,
  广播全 rank;无 EMA/warmup/更新周期/策略旋钮(仅 Buffer 的 B 与
  token_padding);传 `plan=` 则整段跳过(api.py:744)。
- 机制:组负载对容量 `CAP = S·K`(NvS_capacity,api.py:277)盈亏再平衡
  (planning.py:681-701)→ 盈方**最大**专家的配额挪给亏方(:766)→ 每个
  dest rank 复制"被分到 token 的远端专家 top-B"(:866-875),平手取最小索引。

两条推论:

1. **常态恒 rank0←[16] 不是 bug**:两组平均负载恰为 CAP,近均匀路由使
   group1 恒微超容,平手规则每步选中 16。恰反证集成侧每步都传入了新直方图、
   plan 未被缓存复用。
2. **强热观测与强热直方图数学不相容**:全 token→expert0 ⇒ group0 超容 S·K
   ⇒ z[0,1]=S·K ⇒ 正解必为 **rank1←expert0**;实测仍是常态纹样
   (rank0←16、rank1 空)。⇒ 强热 moonep 格里 planner 看到的是**均衡**
   直方图——反常在实验输入侧(hack 未触达 planner 输入),非 planner 行为。
   本会话复 audit 调用点:moon_ep_dispatcher.py 位置参数正对
   (hidden/route_weights/topk_experts/tokens_per_expert)、`plan=None`、
   counts 由 common/moe.py:429 从同一 ids 现算——合同无恙;hot 摘要日志
   未存 loss,无法事后核验该次 hack 在 moonep flavor 里是否真生效(盒已释放)。

## 三、遗留清单

1. ~~planner 语义源码考~~ **完成**(上节);
2. 负载驱动复制实测(需 NVLink,一格)——**带 oracle 复测**:dispatch 调用点
   加一行 `tokens_per_expert` 打印确认强制生效;预期 plan:全热 ⇒ rank1 行
   含 0;mod4 ⇒ rank1 行含 0-3;均衡 ⇒ 全 -1 或边际单复制;
3. prefetch×QLoRA-MXFP4 交点(上游 2bd860b 已支持 MXFP4 专家权重远程预取)。
