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

**Open(已良好定性,后续不需要 NVLink 盒)**:`experts_to_copy` 在两种截然
不同的载荷形状下完全相同(恒 [16,-1,...]/空)——**planner 的复制分配与实时
负载无关**:或需我们未传的显式负载输入,或其默认计划为固定模式。热点未被
分摊(强热下对 standard 仅 +2.7%,mod4 下反而 -2.3%)。下一步=读 moonep
planner 源码(公开)定 `experts_to_copy` 生成语义,再决定 titan 侧是否要传
额外的负载/计划参数。

## 三、遗留清单

1. planner 语义源码考(无需硬件);
2. 语义明确后的负载驱动复制实测(需 NVLink,一格);
3. prefetch×QLoRA-MXFP4 交点(上游 2bd860b 已支持 MXFP4 专家权重远程预取)。
