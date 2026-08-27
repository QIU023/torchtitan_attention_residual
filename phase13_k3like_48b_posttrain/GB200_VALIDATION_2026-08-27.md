# GB200/GB300 大规模验证报告(Elfie, NV,2026-08-27)与逐项分析

Elfie 在 NVIDIA GB200 上对**老树**(`kimi_k3_debugmodel_report_arch` 是老树的
flavor,新树没有它)做了目前为止最大规模的独立验证:1/4/8/16 GPU,最多 4 节点,
direct-Slurm。原文全文见文末附录。本文件是逐项分析:每一项对应哪棵树、根因、
修复计划、以及回同步到新 main(`k3_on_4025`)的路径。

## 一、通过项里最有分量的几条

* **官方 checkpoint 497,220 个 index key 全部闭合**。这是 memory 里那条
  "gates have never run the 497k-key coverage test"的外部补齐 —— 我们的 gate
  因为 worktree 布局一直没跑过这个覆盖测试,现在有了独立执行的结果。
* **PP4 文本与多模态各 10 步,2 节点**;**FSDP2xTP2xCP2xPP2 + EP2,16 GPU 4 节点**
  跑通 2 步。这是老树五轴组合第一次在多节点上被点亮。我们本地 8 卡单机的
  12 格组合矩阵(`CROSS_AXIS_EVIDENCE.md`)与它是互补关系:我们有逐位判据与
  step-2 精度,他们有真实多节点传输面。
* **TP2xCP2 / FSDP2xTP2 / FSDP2xTP2xEP2 单节点通过** —— TP 在老树是完整实现
  (新树 TP 尚在剥离状态),这些格子只有老树能给。

## 二、已知问题逐项分析

### 1. MTP:`KimiMTPLoss` 拒绝 trainer 的 `positions=` 关键字

* **树**:老树(`torchtitan/models/kimi_k3/mtp_loss.py`)。新树**没有** MTP
  文件,迁移时必须从修复后的版本移植(port-from-old-tree-only)。
* **根因**:`KimiMTPLoss.__call__(pred, labels, global_valid_tokens)` 没有
  `positions` 形参;trainer 的 loss 调用带 `positions=` 时直接 TypeError。
  同时 MTP 的 shift-by-k 目标在 packed 序列上会跨文档边界 —— 边界只能从
  `positions` 的回零点识别,所以这个参数不是接了就完,要用它把跨文档目标 mask 掉。
* **Elfie 状态**:本地草稿已验证。我们在 `gb200_fixes` 分支落正式实现。

### 2. Muon:active optimizer factory 解析不到 Muon

* **树**:老树(`muon.py`)。新树没有 Muon。
* **根因**:core 的 `_resolve_optimizer_cls` 硬编码 `{Adam, AdamW}`;
  `muon.py:273` 有一个会解析 Muon 的子类,但 **active factory 不是它** ——
  配方拿到的 container 仍是 core 的,遇到 "Muon" 就 raise。注册没有跟着
  容器的构建路径走。
* **修复**:让 Muon 的解析注册进 active factory 本身,而不是要求调用方换容器类。

### 3. Muon per-head tag 在并行化中丢失

* **树**:老树。
* **根因**:`tag_per_head_muon(model)` 把 `_muon_heads` 标在**参数对象**上;
  并行化(FSDP2/TP 的 DTensor 化)**替换参数对象**,标记随旧对象丢弃。优化器
  分组晚于并行化,拿到的是无标记的新参数 —— per-head 正交化退化或报错。
* **修复**:在 optimizer 分组之前立即重新打标(Elfie 草稿同思路)。这与新树
  8/26 的教训同构:**"属性挂在会被替换的对象上"是这棵代码库反复出现的坑**
  (cache adapter 的 marker attributes、`hasattr(vision_encoder)`)。

### 4. PP8 direct-Slurm 启动挂起(300 秒,edges 5->6 / 6->7)

* **树**:两棵都潜在暴露 —— 这不是模型数值,是 NCCL lazy P2P communicator
  的初始化顺序。rank 6/7 等在 lazy 创建上,300 秒是 NCCL 默认超时。
* **与我们 8/26 发现的关系**:同一症状族(默认 300 秒杀进程),**不同机制**。
  我们那次是冷编译(KDA/tilelang 七分钟)挡住 collective,`init-timeout` 抬高
  即可;这次是 P2P communicator 本身 lazy 创建的顺序死锁,抬 timeout 只会把
  300 秒变成更久的挂起 —— **需要 eager warmup,不是更长的 timeout**。
* **状态**:Open。**本地无法复现**(需要 2 节点 direct-Slurm;我们是单机 8 卡,
  单机 PP8 从未挂过)。修复动作:实现一个 in-process eager PP-edge communicator
  warmup(每对相邻 stage 在训练前互发一个小 tensor),等 Elfie 的诊断确认后
  在 GB200 上验证。warmup 对单机无害,可以先落。
* **新树注意**:新树 PP 走同一 pytorch pipelining 栈,多节点时**同样会踩**。
  warmup 落老树后应同步到新树的 pipeline 入口。

### 5. AttnRes 层 0 梯度断言(test-only)

* **树**:老树 `tests/test_debugmodel.py:54-57`。
* **根因**:`named_parameters()` 按注册顺序返回,`break` 让断言只查**第一个**
  `attention_res_proj.weight` —— 即层 0 的。层 0 之前没有任何已提交的块,
  聚合投影不在计算路径上,梯度可以合法地是 None/零。断言写错了对象,
  不是模型错了。
* **修复**:断言**最后一个**匹配层(它前面必有历史块),显式跳过层 0,
  并留注释说明为什么层 0 不能断言。
* **新树对照**:新树 `test_attn_res_primitive.py` 直接对 primitive 喂非空
  carrier,不经过层 0,无此问题;但迁移 debugmodel 级测试时要带上这个修正。

## 三、性能表的读法(不下超出数据的结论)

* 单卡 ~0.9 TFLOP/s 对 GB200 是极低的 MFU —— **debug 模型太小**,这些数字是
  功能验证的副产品,不是性能数据,不应外推到 1.5T/2.8T。
* CP 4 卡掉到 ~340 tokens/s、TP/PP/EP 8 卡 2 节点掉到 ~110:通信占比在小模型
  上被放大,同样不可外推。真正的性能工作在他们的 remaining work #5 之后。
* 唯一可横向读的:LoRA 峰值内存 1.02 GiB vs 全参 1.70 GiB,冻结基座的效果
  在多模态下成立。

## 四、修复分支与同步计划

老树分支 **`gb200_fixes`**(基于 `k3_tp_declarative` @ `9b93d5ca3`):

| # | 修复 | 同步到新树 |
|---|---|---|
| 1 | MTP `positions=` + 跨文档 mask | 迁移 MTP 时从修复后版本移植 |
| 2 | Muon 注册进 active factory | 迁移 Muon 时同上 |
| 3 | Muon tag 分组前刷新 | 同上 |
| 4 | AttnRes 测试断言后层 | 迁移 debugmodel 测试时带上 |
| 5 | PP-edge eager warmup | **立即同步** —— 新树同栈,多节点同样会踩 |

新树今日已推的最新发现(本报告到来之前):`e973802e6` —— DEP+cache 布局修复
(四格前向逐位相同)与 bubble deferred backward 打通(8/8 计划槽,0 兜底),
见 `MULTIMODAL_EVIDENCE_2026-08-26.md` 二之二。

## 附录:Elfie 报告原文(verbatim)

> Here is a quick summary of progress from agents, please take a look at the
> functional issue section - larger scale validation is still undergoing.

### GB200 Validated functionality

| Functionality | Registered configuration / artifact | Scale | Evidence | Status |
|---|---|---:|---|---|
| Release artifact integrity | Official Kimi-K3 checkpoint, 96 shards | CPU - File validation | All 497,220 index keys close; no unmapped or non-stacked runtime mapping mismatches | Passed |
| Basic text training | `kimi_k3_debugmodel_report_arch` | 1 GPU | Deterministic 10-step runs with identical TensorBoard scalar traces | Passed |
| KDA, MLA, routed MoE, AttnRes | `kimi_k3_debugmodel_report_arch` | 1 GPU | Finite loss and nonzero gradients in the principal KimiK3 components | Passed |
| Image + text training | `kimi_k3_debugmodel_report_arch` + MoonViT | 1 GPU | Image input changes logits; vision patch embedding receives nonzero gradient | Passed |
| LoRA fine-tuning | `kimi_k3_debugmodel_report_arch_lora` | 1 GPU | Base weights frozen, adapters update, merge error 0.0, HF export contains 604 tensors | Passed |
| QAT | `kimi_k3_debugmodel_report_arch_qat` | 1 GPU | Routed-expert MXFP4/MXFP8 fake quantization, finite STE gradients, 10-step training | Passed |
| Multi-token prediction | `kimi_k3_debugmodel_report_arch` + MTP loss | 1 GPU | Trainer-compatible `positions=` path and packed-document boundaries, 10 steps | Passed |
| Muon + Quantile Balancing | `kimi_k3_debugmodel_report_arch` | 1 GPU | Muon/AdamW resolution, QB hooks in 20 MoE layers, tag refresh, 10 recipe steps | Passed |
| Distributed parallelism | `kimi_k3_debugmodel_report_arch` | 4 GPUs / 1 node | TP2xCP2, FSDP2xTP2, and FSDP2xTP2xEP2 finite training runs | Passed |
| Direct Slurm multi-node preflight | `kimi_k3_debugmodel_report_arch` | 8 GPUs / 2 nodes | Direct rank setup, all-reduce/all-to-all, repeated FSDP2xTP2xCP2 10-step mini-run | Passed |
| PP4 text pipeline training | `kimi_k3_debugmodel_report_arch` | Up to 8 GPUs / 2 nodes | Ten pipeline training steps | Passed |
| PP4 multimodal pipeline training | `kimi_k3_debugmodel_report_arch` + MoonViT | Up to 8 GPUs / 2 nodes | Ten image+text pipeline training steps | Passed |
| Four-node initial pipeline training | `kimi_k3_debugmodel_report_arch` | 16 GPUs / 4 nodes | FSDP2xTP2xCP2xPP2 with EP2, two finite pipeline steps | Passed |

### Known functional issue

| Issue | Observed behavior | Resolution / next step | State |
|---|---|---|---|
| MTP trainer interface | `KimiMTPLoss` rejects trainer keyword `positions=` | Local draft fix accepts keyword-only `positions` and masks cross-document targets | Fixed locally; validated |
| Muon optimizer registration | Muon recipe fails before training because the active optimizer factory does not resolve Muon | Local draft fix registers/resolves Muon in the active factory | Fixed locally; validated |
| Muon tag lifetime | Parameter replacement during parallelization can discard per-head tags before optimizer grouping | Local draft fix refreshes tags immediately before groups are built | Fixed locally; validated |
| PP8 direct-Slurm startup | Text and multimodal PP8 reach all stages, then ranks 6/7 wait 300 s creating lazy NCCL P2P communicators for edges 5->6 and 6->7 | Run a minimal two-step PP8 NCCL diagnostic repro, then add an in-process eager PP-edge communicator warmup if confirmed | Open; launcher/runtime issue, not model numerics |
| AttnRes layer-0 gradient assertion | Layer 0 has no prior history, so its aggregation projection can have zero/absent gradient | Correct the test to assert a later layer rather than requiring every layer | Test-only correction |

### Remaining work (Pending GPU scheduling)

1. Run the 16-GPU quantization-aware training job and verify ten clean training
   steps, quantization activity, and complete per-rank metrics.
2. Run the 16-GPU image-and-text training job. Verify that the vision tower
   receives gradients, save a distributed checkpoint after step 5, reload it,
   and continue successfully through step 10. Then run the equivalent 16-GPU
   LoRA job and verify frozen base weights and adapter updates.
3. Reproduce the eight-stage, two-node pipeline startup hang with one small
   direct-Slurm diagnostic. If it confirms the suspected NCCL P2P initialization
   ordering issue, add an eager pipeline-neighbor communicator warmup and rerun
   the affected text and image-and-text cases.
4. Complete the distributed-layout cases that did not run before the previous
   four-hour allocation expired. Submit only those missing cases in smaller,
   bounded allocations rather than repeating completed work.
5. Full 1.5T training - A complete production-scale plan for real checkpoint
   loading, memory budgeting, optimizer state, sharding, and a larger Slurm
   allocation. Deferred until initial performance optimizations are done.

### Current measured training performance on GB200

| Workload | Scale | Stable trainer-reported throughput | FLOPs | Peak memory per GPU | Loss over 10 steps |
|---|---:|---:|---:|---:|---:|
| Text training | 1 GB200 | ~2,200 tokens/s | ~0.88-0.90 TFLOP/s | 1.64 GiB | 7.72 -> 5.14 |
| Image + text training | 1 GB200 | ~1,330-1,440 tokens/s | ~0.54-0.59 TFLOP/s | 1.70 GiB | 12.05 -> 9.84 |
| Image + text LoRA | 1 GB200 | ~1,470-1,500 tokens/s | ~0.60-0.61 TFLOP/s | 1.02 GiB | 12.03 -> 11.90 |
| Image + text QAT | 1 GB200 | ~1,230-1,270 tokens/s | ~0.50-0.51 TFLOP/s | 1.73 GiB | 12.05 -> 9.86 |
| Image + text context-parallel test | 4 GB200s / 1 node | ~335-346 tokens/s | ~0.14 TFLOP/s | 1.51 GiB | 12.07 -> 9.89 |
| Image + text TP/PP/EP test | 8 GB200s / 2 nodes | ~104-112 tokens/s | ~0.04-0.05 TFLOP/s | 1.22-1.23 GiB | 12.07 -> 9.74 |

## 五、修复落地(2026-08-27,`gb200_fixes` 分支,已推)

| commit | 修复 | 验证 |
|---|---|---|
| `b76f81269` | Muon 注册 + tag 刷新 | `kimi_k3_mini_muon` 1 卡 3 步,loss 7.71->7.53,tagged 57,无 "not added" |
| `d3f578f1d` | MTP `positions=` + 跨文档 mask | 新测试:两文档打包,mask 改变总 loss;旧签名下 TypeError |
| `e4d8118e0` | AttnRes 断言改最后一层 | 24 passed |
| `12a8fd298` | PP-edge eager warmup | 单节点 PP8(report_arch,多模态数据)2 步 rc=0,loss 12.035->11.989 |

**Muon 注册的根因比报告更具体**:core 把 `_resolve_optimizer_cls` 改名为
`_resolve_optimizer_factory`,子类覆写留在旧名字上变成死代码 —— "override 盯着
一个已被改名的钩子"。新增测试钉住解析,下次改名会红而不是静默。

**warmup 已同步新树**(`2ef00694c`,main/k3_on_4025):同一 pipelining 栈,跨节点
第一次跑就会踩同一个坑。TODO 注释在 schedules 模块里把缺口和修法都写明了
("STATIC mode group communicator warm-up gap"),我们做的是把它从 torchtitan 侧
落地 —— 训练仓改不了 torch 本体。

**双节点复现待办**:挂起本身需要 2 节点才能复现,本地只有单机 8 卡。单节点已验证
warmup 无害;是否真的解掉 300 秒挂起,要在出问题的集群上跑确认(或 Elfie 侧已修)。

MTP/Muon 不在新树上(尚未迁移);迁移时按 port-from-old-tree-only 从修复后的
版本移植。
