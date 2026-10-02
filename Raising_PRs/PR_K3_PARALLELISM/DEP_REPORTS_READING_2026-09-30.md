# DEP：K2.5 和 K3 报告原文，以及和我们实现的对照（2026-09-30）

用户 09-30："重新阅读报告段落 搜索k2.5 report dep原始描述部分 ... 不要自己瞎改 给出理解后讨论"。动态 CP（#4380）不和 DEP 混在一起，确有必要再考虑整合。

## K2.5 报告（arXiv 2602.02276，09-30 在线核对）

- 问题："due to the inherent variations of multimodal input size (e.g., image counts and resolutions), Stage-0 suffers from drastic fluctuations in both computational load and memory usage."（视觉塔和文本 embedding 都在 PP 的 stage 0。）
- Stage 1，Balanced Vision Forward："We first execute the forward pass for all visual data in the global batch. Because the vision encoder is small, we replicate it on all GPUs regardless of other parallelism strategies. During this phase, the forward computational workload is evenly distributed across all GPUs based on load metrics (e.g., image or patch counts)." 只保留最终输出，中间激活全部丢掉。
- Stage 2，Backbone Training：主干的前向和反向，沿用纯文本训练验证过的并行策略。
- Stage 3，Vision Recomputation & Backward："We re-compute the vision encoder forward pass, followed by a backward pass to compute gradients for parameters in the vision encoder."
- 效率："achieving a multimodal training efficiency of 90% relative to text-only training."

## K3 报告 §5.2.3（`phase13_k3like_48b_posttrain/official_k3/report.txt` 第 1414 到 1421 行）

"In Kimi K2.5, we introduced the Decoupled Encoder Process (DEP) [59], which splits ViT and text training into separate stages and balances vision forward and backward passes across PP stages. We observe that, under the interleaved 1F1B pipeline schedule, the text forward passes of the first PP micro-batches are all scheduled at the very beginning, while the text backward passes of the last PP micro-batches finish only at the very end. We therefore further decompose the ViT computation. The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles, and the backward passes are handled analogously. As a result, most of the ViT computation is hidden within pipeline bubbles, largely eliminating the effective overhead of the vision encoder."

## 理解

- K3 拆的是 K2.5 第一段、第三段那两整块，按 micro-batch 拆开放：
  - 前 PP 个 micro-batch 的文本前向在最开头，所以它们的 ViT 前向必须同步先做；其余的放进气泡。
  - 反向同理：最后 PP 个 micro-batch 的文本反向到最后才结束，它们的 ViT 反向只能在结尾同步做；其余的放进气泡。
- 拆的单位是 micro-batch。两份报告都没有按层拆 ViT。
- "most of the ViT computation is hidden" 的前提是 ViT 在整步里占比小：K3 的塔 0.4B、27 层，按我们的模型每张图 R_f 0.02 到 0.32。视觉占整步 10% 到 25% 时，气泡本身只有一步的 5% 到 18%，按容量就装不下大部分 ViT。

## 对照我们的实现（`k3_pp_mm` = `d27839459`）

- **和报告一致的：**
  - 每个 PP rank 上有塔的副本，前向不存中间激活，反向时重算；
  - 前 PP 个 micro-batch 的编码放在调度前，其余的进气泡；
  - 反向在梯度就绪后进气泡，放不下的放在收尾。
  - 09-30 修的两处是为了正确找出报告说的"气泡"：一是跳过了梯度就绪前开始的空闲段，二是按格数量气泡。都不是设计改动。
- **报告有、我们没有的：**
  - K2.5 的平摊单位是图或 patch，范围是所有 GPU（整个 global batch）；
  - 我们是把整个 micro-batch 交给一个 PP rank（按 patch 数做 LPT），范围是一个 DP 副本里的 PP rank，TP 组内用切分后的塔，不是每张卡一整份副本。
- **报告没写清的：** K3 把 ViT 前向和反向放进气泡时，一个 micro-batch 的 ViT 是否仍像 K2.5 那样摊到多张卡上。
- **撤回：** 09-30 我提的"按层拆 ViT"两份报告都没有，撤回。"按图拆"对应的是 K2.5 的平摊单位，要不要做由用户定。

## 2.8T K3 高清视频预训练的文本 : 视觉计算比例（估算，`dep_bubble_fix/k3_ratio.py`）

用户 09-30："这个我不确定 因为实际场景是2.8T k3高清视频预训练，即使视频有token压缩 你估算一下可能的文本：视觉计算比例？"，并且"必须吻合报告的描述 不能乱改 PP CP同时打开后再加dynamic cp 现在没有"。

- **依据：**
  - K3 表 1：激活 104.2B，93 层里 24 层 MLA，96 个头，hidden 7168；视觉塔 401M、27 层、patch 14；投影前 2 × 2 pixel-shuffle；预训练上下文 8K 到 64K，cooldown 256K 到 1M。
  - K2.5 的 MoonViT-3D：4 帧一组，塔之后做时间池化，时间上压缩 4 倍；视觉与文本 token 比例做过 10:90、20:80、50:50 的消融，比例越低越好。
- **假设：**
  - MLA 的打分和取值 head 维度按 K2 取 192 / 128；
  - 视觉塔宽度 1152（由 401M ÷ 27 层反推）；
  - 空间注意力在帧内，时间那一步和 KDA 忽略不计；
  - LLM 训练按前向的 3 倍；DEP 下视觉塔按 4 倍（前向、重算、反向）。
- **换算：** 进 LLM 的一个视频 token 对应视觉塔里 16 个 patch，一张图对应 4 个 patch。1080p 一帧约 10,549 个 patch，帧内注意力让每个 patch 的前向从 0.8 涨到 2.1 GFLOP。

| 输入 | 上下文 | 每个视觉 token 的 ViT ÷ LLM（训练） | 视觉占整步计算，视觉 token 占 10% / 20% / 50% / 90% 时 |
|---|---|---:|---|
| 448 px 图 | 8K | 0.023 | 0.2% / 0.5% / 1.1% / 2.0% |
| 3584 px 图 | 8K | 0.223 | 2.2% / 4.3% / 10.0% / 16.7% |
| 720p 视频 | 8K | 0.137 | 1.4% / 2.7% / 6.4% / 11.0% |
| 1080p 视频 | 8K | 0.210 | 2.1% / 4.0% / 9.5% / 15.9% |
| 1080p 视频 | 64K | 0.176 | 1.7% / 3.4% / 8.1% / 13.7% |

- **读法：**
  - 按 K2.5 那种低视觉比例混合（视觉 token 占 10% 到 20%），文本 : 视觉约为 96 : 4；
  - 视频为主的高清批次（视觉 token 占 50% 到 90%），约为 90 : 10 到 84 : 16；
  - 上下文越长，LLM 的注意力越重，视觉占比越小。cooldown 阶段到 256K 到 1M 时还会更小。
  - 09-30 早些时候用的 75 : 25 超出了这个范围。

**在这些比例下，气泡能藏住多少（规划器模型值，修复后）。** K3 报告没给 PP 度数，这里按 K2 的 PP16 取 PP16 × VP2：

| 布局 | 文本 : ViT | 气泡占一步 | 填充率 | ViT 藏住（前向 / 反向） | 反向进气泡，修复前 → 后 | ViT 让一步变长：DEP 关 / K2.5 / bubble |
|---|---|---:|---:|---|---:|---|
| pp16 × vp2，M32 | 96 : 4 | 18% | 15% | 83%（50% / 94%） | 27 → 30 / 32 | +54% / +3% / +2% |
| pp16 × vp2，M32 | 90 : 10 | 18% | 31% | 62%（50% / 66%） | 19 → 21 / 32 | +143% / +9% / +4% |
| pp16 × vp2，M32 | 84 : 16 | 18% | 29% | 34%（50% / 28%） | 9 → 9 / 32 | +246% / +15% / +13% |
| pp16 × vp2，M64 | 96 : 4 | 10% | 34% | 91%（75% / 97%） | 54 → 62 / 64 | +60% / +4% / +1% |
| pp16 × vp2，M64 | 90 : 10 | 10% | 43% | 43%（75% / 33%） | 21 → 21 / 64 | +159% / +10% / +6% |
| pp16 × vp2，M64 | 84 : 16 | 10% | 50% | 29%（75% / 14%） | 9 → 9 / 64 | +272% / +17% / +14% |

- **结论：**
  - 在 96 : 4 这个典型比例下，按报告的写法（前 PP 个 micro-batch 同步在前，其余进气泡，反向同理）就能藏住 83% 到 91% 的 ViT，和报告说的 "most of the ViT computation is hidden" 一致；修复让反向多进了 3 到 8 次。
  - 视频为主的 90 : 10 到 84 : 16 时，藏住 29% 到 62%。
  - 09-30 早些时候"单按容量就装不下大部分 ViT"的说法，只在 75 : 25 这种超出 K3 实际范围的比例下成立，撤回。
- **不做的：** 按层拆 ViT 不做。动态 CP 等 PP 和 CP 同时能开以后再加，现在没有。

## 10-02 补充：K3 报告 v2 给这句加了引用 [34]（Optimus）

用户 10-02 引了 "We therefore further decompose the ViT computation [34]"，问我们做对了没有。

- **版本**：本地 `phase13_k3like_48b_posttrain/official_k3/report.txt` 是 07-27 拿到的 v1，这句没有引用。arXiv 2607.24653 的 v2（08-07）改成 "...decompose the ViT computation [34]"。v2 的 [34] 是 Weiqi Feng et al., "Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation", USENIX ATC 25。DEP 的引用在 v2 里是 [60]（K2.5）。
- **为什么加**：MoonshotAI/Kimi-K3 的 Issue #3（Optimus 第一作者提的）指出 §5.2.3 缺这条引用；v2 补上了，描述 K3 自己做法的那几句没有改。
- **Optimus 的做法**（原文 §3、§4.2、§4.3）：
  - encoder 和 LLM 各用一套并行方案，每张卡都有 encoder 的状态（encoder 也可以自己切 PP，成为若干条 "encoder pipeline"）；
  - 粗粒度：encoder 前向放进 LLM 计算之前那一大段气泡（DP all-gather 加 PP warmup），反向放进之后那一大段（PP cooldown 加 DP reduce-scatter），放不下的算作没藏住；
  - 细粒度：反复挑在关键路径上的那条 encoder pipeline，把它一个 micro-batch 的计算拆到 kernel 粒度，按气泡时长塞进夹在 LLM 计算之间的小气泡（其他 PP 气泡、约 300 µs 的 TP 气泡），满足 encoder 内部的层间依赖；encoder 的通信 kernel 放在 LLM 的计算 kernel 期间；
  - 依赖按 micro-batch 检查：encoder 第 i 个前向的结束时间 ≤ LLM 用它的时刻 F_i，反向的开始时间 ≥ LLM 产出梯度的时刻 B_i；
  - 还调整了 interleaved 1F1B 的 warmup 个数，把最后几个 micro-batch 的前向依赖点往后推，腾出 warmup 到 steady 之间的气泡。
- **和我们的实现（`k3_pp_mm` = `d27839459`，`dep_review1` = `1d03bfcc6`）对照**：
  - K3 原文三句话（前 PP 个 micro-batch 的 ViT 前向同步先做，其余进气泡，反向同理）：做到了。`plan_dep` 里 `upfront = by_consume[:num_ranks]` 进 prologue（按 patch 数在 PP rank 间均衡）；其余编码按 stage 0 的消费顺序找它前面的空闲段；反向找梯度就绪之后的空闲段，后 PP 个通常没有空闲段可用，落在 epilogue。10-01 的 5060 实测在 K3 区间（代价比例 0.13 到 0.82）里，前 PP 个以外的编码全部进气泡，反向 13 个里 12 个进气泡。
  - 比 Optimus 粗的地方：一个 micro-batch 的编码或反向是整块放进一个空闲段的，不拆层也不拆 kernel，放不下就退回 prologue / epilogue；气泡只看动作网格里的 PP 空闲，不用 TP 气泡和 DP 通信气泡；encoder 不切 PP（K2.5 的 DEP 是每张卡一份塔，我们是每个 PP rank 一份、TP 组内切分）；没有调整 warmup。
  - 09-30 我写的"拆的单位是 micro-batch，两份报告都没有按层拆 ViT"是按 v1 说的。K3 原文仍然只写到 micro-batch 这一层，但 v2 引的 Optimus 拆到了 kernel。要不要往细里做（比如按层把一次编码拆到同一个 rank 的几个连续空闲段里），由用户定。
  - PR #4381 的 body 建议补一行引用 Optimus（K3 也引了它），并写明我们的粒度是整个 micro-batch、气泡只用 PP 空闲。

## 10-02 用户问：`plan.py` 的代价常数是不是 K2.5 原来的做法

用户 10-02："plan.py 为什么这里面是怎么计算cost和估算step time的？？？它是DEP（K2.5）原始的做法吗？"

- **不是。** K2.5 的 DEP 没有时间或代价模型，只按图或 patch 数在所有 GPU 上均摊前向。K3 §5.2.3 只有三句话，没说怎么判断能不能放进气泡。
- 这几个数是我们为实现 K3 那句"其余的放进气泡"加的：要判断一个 micro-batch 的整块编码，能不能在它被用到之前放进某段空闲，就得知道空闲多长、编码多长。约束和 Optimus 写的一样（编码结束 ≤ F_i，反向开始 ≥ B_i），但 Optimus 用 profile 出来的真实时长，我们用常数。
- 每个数：
  - `_ACTION_COST`：torch 的 `pipeline_order` 是锁步的格子，一格一个动作，不分前向、反向的长短。这里让每格的时长等于这一格里最长的动作：F 1，B 2（反向约为前向两倍，常用假设），I、W 各 1（拆开的反向各一半）。算出的是格子边界的时刻，单位是一次文本 stage 前向。没实测过。
  - `_BACKWARD_COST = 3`：DEP 下塔的反向 = 重算前向 1 + 反向 2，以编码为单位。H100 标注实测：1008 px 的图 3.06，224 px 的图 2.06（`DEP_K3RANGE_H100_2026-10-02.md` 第二轮）。
  - `_TRANSFER = 1`：特征或梯度换 rank 的预留时间，按一次文本 stage 前向算。拍的，没实测。
  - `bubble_cost_ratio`（配置项，默认 1.0）：一个平均 micro-batch 的编码等于几次文本 stage 前向，各 micro-batch 按 patch 数线性缩放。H100 标注实测是 0.046 到 0.170。
- 它不估计 step time，只算格子边界的时刻，用来判断整块放不放得下；放不下的退回调度前后，也就是 K2.5 的形式。
- 只在 `bubble=True` 时起作用。K2.5 形式只按 patch 数在 PP rank 间做 LPT 均衡，比例和常数对所有 micro-batch 同乘，不改变结果。B200 格子现在开的就是 K2.5 形式。
- 实测（GPU 会话 10-02，`b2a57dff7`，标注 kernel 时间）：比例 0.046 / 0.135 / 0.170 时，bubble 藏住 73% / 81% / 84% 的视觉计算，K2.5 形式是 2% 到 3%；没有一段视觉工作卡住本 rank 的下一个动作。
- 和 K2.5 原文还有一处不同（09-30 已记）：K2.5 按图或 patch 在所有 GPU 上均摊，我们按整个 micro-batch 在一个 DP 副本的 PP rank 间均摊，TP 组内用切分的塔。
