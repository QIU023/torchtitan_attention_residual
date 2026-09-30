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
