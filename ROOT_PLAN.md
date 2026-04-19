给定step list：进入文件夹开始phase 0

# AttnRes PR 项目 — 完整交接文档

## 一、项目背景与决策

**目标**：在 torchtitan（首选）或 Megatron-LM（备选）实现 Kimi Block Attention Residuals (AttnRes)，提交 PR，作为求职硬通货。

**职业定位**：
- 主线：VLM/CV MLE（NVIDIA BEV、Waymo、Zoox、Tesla、Wayve）
- 副线：AI infra / training framework engineer（NVIDIA Megatron team、Meta PyTorch team、Anthropic ML Networking、RadixArk、Stack AV）
- 此 PR 主要服务副线，但对主线也有显著加分（证明能驾驭大规模训练栈）
- 完美匹配差异化叙事：**datacenter networking 2 年 + ML 背景** → cross-stage caching 在 PP 下的工程实现是这个交叉点的杀手锏

**为什么选 AttnRes PR 而不是分布式 VLM 项目**：
1. 时间窗口正在关闭（论文发布 ~1 个月，主流框架未集成，先动手者通吃）
2. 外部验证 > 自我宣称（merged PR > 又一个 side project）
3. 杠杆已有 networking 背景，叙事干净
4. 对 VLM MLE 主线也加分

---

## 二、关键链接

**论文与官方实现**：
- arXiv 论文：https://arxiv.org/abs/2603.15031
- arXiv PDF：https://arxiv.org/pdf/2603.15031
- Moonshot 官方 repo（reference 实现）：https://github.com/MoonshotAI/Attention-Residuals
- Hugging Face paper page：https://huggingface.co/papers/2603.15031

**Kimi infra 工程师亲述实现细节（最重要）**：
- 知乎问题页：https://www.zhihu.com/question/2016993095078684011
- 关键回答（Megatron 实现路径）：https://www.zhihu.com/question/2016993095078684011/answer/2017381145474508331

**论文解读（背景理解）**：
- 详细论文解读：https://zhuanlan.zhihu.com/p/2016957666388387770
- 中文技术解读：https://zhuanlan.zhihu.com/p/2017270578223006105
- "纵向注意力"背景思考：https://zhuanlan.zhihu.com/p/2017100562542403612
- Kimi 官方介绍博客：https://zhuanlan.zhihu.com/p/2018056040218932797
- Ziming Liu 反思博客（No-Free-Lunch 视角）：https://kindxiaoming.github.io/blog/2026/attention-residual/

**目标框架**：
- torchtitan（首选）：需要查 `torchtitan/distributed/pipeline_parallel.py` 和 PyTorch 的 `torch.distributed.pipelining`
- Megatron-LM（备选）：https://github.com/NVIDIA/Megatron-LM
  - attention 实现位置：`megatron/core/transformer/attention.py`
  - arguments 注册位置：`megatron/training/arguments.py`

---

## 三、算法理解（核心要点）

**Standard Residual**：`h_{l+1} = h_l + f_l(h_l)`，所有前序层等权累加 → 隐藏状态幅度 O(L) 增长，深层稀释浅层贡献。

**Full AttnRes**：`h_l = Σ α_{i→l} · v_i`，softmax 注意力替代固定累加。
- 每层一个**可学习的 pseudo-query 向量** `w_l ∈ R^d`（与 input 无关，类似深度方向的位置编码）
- Key/Value 是历史层的 RMSNorm 输出
- **必须零初始化 pseudo-query**：保证训练初期等价于 uniform 平均（即标准 residual），防止训练 volatility
- 内存 O(Ld)，PP 不友好（Full AttnRes 没有好的 PP 适配方案）

**Block AttnRes**（实际使用的版本）：
- 把 L 层分成 N 个 block（N≈8 是 sweet spot）
- 块内用标准 residual 累加，块间用 attention 聚合 block summary
- 内存 O(Nd)，论文 Table 1：5.5d per layer（vs standard 3d，vs mHC m=4 是 34d）
- 通信 O(P²V)，P = 物理 stage，V = virtual stage
- N=2,4,8 性能几乎相同；N=16,32 退化
- **Block AttnRes 等同于 baseline × 1.25 算力**（25% 训练效率提升）

**关键性能数字**：
- 训练开销 < 4%（PP 下）
- 推理延迟开销 < 2%（two-phase computation + online softmax）
- GPQA-Diamond +7.5，HumanEval +3.1，Math +3.6
- 训练动态：output magnitude 周期性有界（baseline 单调递增）；gradient 跨层均匀（baseline 浅层异常大）

**两阶段推理计算（Two-phase）**：
- Phase 1: pseudo-query 是固定参数 → 一个 block 内所有 layer 的 cross-block attention 可以 batched 一次算完
- Phase 2: 块内 sequential，用 online softmax 跟 phase 1 结果合并

---

## 四、工程难点（来自 Kimi infra 工程师亲述）

**核心难点是 PP 下的 cross-stage caching adapter，不是算法本身**。

引用关键描述：
> "在这个具有良好局部性的算法下，cross-stage caching 的通信优化就比较容易想到，除了第一个 vp chunk 外，后面的所有 pipeline 通信都是常数，解决了不对称的问题。"

> "大致思路就是在 pipeline 并行的通信后增加一个适配器，将收到的 block 与适配器中缓存的 block 进行拼接。反向也是类似思路，会收到所有 block 的 grad，所有的 grad 就在适配器里面做累加，需要发到下一个 stage 的就直接把累加 buffer 发出去。整个代码逻辑还是比较对称的，不影响网络内部逻辑。"

> "在最常用的 interleave pipeline 的调度下，send/recv 的开销在 steady 阶段非常容易掩盖，只有 warmup 和 cooldown 阶段的一点点通信会暴露出来。"

**实现要点**：
1. **Adapter 模式**：在 PP send/recv 之后挂一个 buffer
2. **Forward**：recv 之后，把缓存的 block 和新收到的 block 拼接
3. **Backward**：在 buffer 里累加所有 block 的 grad，需要发到上一个 stage 的就把累加 buffer 发出去
4. **逻辑对称**，不污染网络内部
5. 必须用 **interleaved 1F1B + virtual pipeline** 才能展示 steady-state 通信掩盖

**为什么 Block 而不是 Full**：苏神最初想做 Full AttnRes，PP 适配实在没招（通信不对称问题）。Block 的局部性恰好让 cross-stage caching 通信对称化。

**Pseudo-query 零初始化**：算法上的关键，PR 里必须正确实现，否则训练前期会炸。

---

## 五、硬件决策

**最终选择：8× 5090 PCIe**（备选 8× 4090 PCIe）

**为什么不是 2× A100**：
- PP=2、VP=1 或 VP=2，只有 2-4 个 chunk
- 看不出 interleave 调度的 steady-state 通信掩盖
- Reviewer 会说"跟单卡没本质区别"

**为什么 8 卡 PCIe 是最优**：
1. **PP=8 + N_blocks=8 完美对齐**论文默认配置，每个 PP stage 对应一个 block 边界
2. **VP=2 → 16 个 virtual chunk**，足够展示 interleave 通信掩盖
3. **PCIe 互联反而是加分项**：Block AttnRes 是为低带宽设计的（O(Nd) 通信），在 PCIe 上跑通 overhead < 5% 比 NVLink 更有说服力。PR description 写："Block AttnRes makes pipeline overhead negligible even over PCIe bandwidth"
4. 5090 32GB > 4090 24GB，2B Llama3 dense + FSDP 内层 + PP 外层完全够

**预算**：Vast.ai 8x 4090 节点 ~$6-10/hr，集中跑 5-7 天主要训练，加上前期单卡试验，**总预算 $1k-1.5k**

---

## 六、Phased Plan

### Phase 0：调研 + Reference 复现 + API 摸查（5-7 天，1× 4090，~$50）

**目标**：决定走 torchtitan 还是 Megatron-LM；不烧钱地排掉风险

**任务**：
1. Clone Moonshot 官方 repo，看 reference 实现的 `block_attn_res` 函数
2. 在 200M-500M 小模型上**复现 Full AttnRes 和 Block AttnRes 的 loss 对齐**（单卡，无 PP）
3. **Clone torchtitan，深入读以下文件**：
   - `torchtitan/distributed/pipeline_parallel.py`
   - PyTorch 的 `torch.distributed.pipelining._PipelineStage` 类
   - 找 send/recv 路径的扩展点
4. **判定**：torchtitan 的 PP API 是否能干净地挂 cross-stage caching hook？
   - 干净 → 走 torchtitan
   - 太封闭 → 切 Megatron-LM（Kimi 工程师吐槽是"屎山"但确实能实现）
5. 同时在 torchtitan 和 Megatron-LM 各开一个 RFC issue 占坑

**风险止损点**：如果两个框架都没有干净扩展点，重新评估范围

### Phase 1：RFC + Maintainer 对齐（同 Phase 0 并行）

- torchtitan/Megatron 各开 RFC issue
- 描述实现方案、引用论文、列出关键设计点
- 看 maintainer 的反应速度和接受度，谁响应快做谁
- **占坑很重要**：声明你在做这个 PR，避免撞车

### Phase 2：Block AttnRes 单卡正确性（5-7 天，1× 4090）

- 把 reference 实现移植到目标框架（torchtitan 或 Megatron）的 model 定义里
- 单卡 1B 模型跑 loss curve，对齐论文小规模点的 delta（≈0.02 loss gap）
- 写好 unit test：pseudo-query 零初始化、Full vs Block 等价性边界（N=L 时）、forward/backward 数值正确性
- 加上 `--use-attn-res` / `--attn-res-num-blocks` 等 CLI flag

### Phase 3：Cross-stage Caching Adapter 实现（7-10 天，1× 4090 + fake PP）

**这是整个项目的技术核心**

- 用 PyTorch **fake process group** 在单卡 mock PP=4 的环境
- 实现 adapter 类：
  - `ForwardAdapter`：recv 之后拼接 buffered blocks
  - `BackwardAdapter`：accumulate grads，send buffered grad
- Hook 进 PP 的 send/recv 路径
- 跑 forward/backward 的数值正确性测试（fake PP 跟单卡 reference 对齐）
- 写 unit test 覆盖 warmup / steady / cooldown 三个阶段

### Phase 4：真正 8 卡 PP 端到端（5-7 天，8× 5090 PCIe，~$1k-1.5k）

**烧钱集中阶段，前面准备越充分越省**

- 配置：PP=8, VP=2, FSDP内层, N_blocks=8, Llama3 1.5B-2B dense, BF16
- 跑 ~20B token 量级
- **关键 benchmark**：
  - Loss curve：AttnRes vs baseline，对齐论文小规模点
  - Step time：steady state 阶段 overhead < 5%
  - Memory：每层 5.5d vs 3d（baseline）
  - Communication trace：interleave schedule 下 steady 阶段通信被掩盖
- 输出 wandb dashboard 或类似可视化

### Phase 5：PR + Blog（3-5 天）

- PR description：清晰列出动机、设计、benchmark 数字
- **关键卖点写法**：
  > "Block AttnRes makes pipeline overhead negligible even over PCIe bandwidth interconnect"
- Blog post（标题草拟）："Implementing Attention Residuals in torchtitan: Notes on Cross-stage Caching under Pipeline Parallelism"
- Blog 比 PR 更容易传播，单独写一份
- 可能 cross-post 到 torchtitan Slack/Discord、Twitter、知乎
- 在 PR 里 @ Kimi infra 工程师邀请 review（可选，加分）

---

## 七、风险与应对

| 风险 | 应对 |
|---|---|
| torchtitan PP API 太封闭 | Phase 0 验证；切 Megatron-LM |
| 5090 在 Vast.ai 驱动不成熟 | 退回 4090（24GB，2B 模型还是够，需缩 batch） |
| PR 被拖很久不 merge | High-quality open PR + blog 已经是简历资产；可以放自己 fork 上 |
| 同时有人在做 | RFC issue 占坑要快；从现在算 1-2 月窗口 |
| 反向传播的 grad accumulate 在 fake PP 下不好测 | 用小规模 PP=2 单机双卡（租 1 小时验证）补测 |
| 论文 scaling law 你复现不出来 | 不需要复现；PR 只需正确性 + 小规模 delta + benchmark |

---

## 八、关键设计点 Cheatsheet

```python
# Pseudo-query 必须零初始化
self.attn_res_query = nn.Parameter(torch.zeros(d_model))

# Block AttnRes 核心（来自 Moonshot reference）
def block_attn_res(blocks, partial_block, proj, norm):
    """
    blocks: N tensors of shape [B, T, D]
    partial_block: [B, T, D]
    """
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h
```

**配置默认值**：
- `num_blocks = 8`
- `block_size = num_layers / num_blocks`
- `attn_res_norm = RMSNorm(d_model)`

**PP 配置**：
- PP_size = num_blocks（= 8）
- VP_size = 2
- Schedule: interleaved 1F1B

**与 PP 集成的关键不变量**：
- 每个 PP stage 边界 = 一个 block 边界
- Cross-stage 传递的是 block summary（一个 [B, T, D] tensor），不是 N 个
- Adapter 里维护的 buffer 长度等于 stage_id

---

## 九、下一步开工动作

1. Clone Moonshot Attention-Residuals repo
2. Clone torchtitan，定位 PP 扩展点
3. 起 1× 4090 spot 实例（vast.ai）
4. 复现 reference 单卡训练
5. 同时起草 torchtitan 和 Megatron-LM 的 RFC issue
6. 决定主路径，开干 Phase 2