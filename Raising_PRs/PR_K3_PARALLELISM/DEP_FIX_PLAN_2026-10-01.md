# DEP 规划器修正方案（2026-10-01，只记录，等用户确认后再改代码）

用户 10-01："离谱 我觉得DEP H100结果作废了，有可能在5060上面算出与模型大小关系小一点的固定掩藏比例，然后放到H100跑？"，并引 K3 报告原文问"这是我们现在DEP分支的做法吗？不会又歪了吧"；随后"记录一下修正方案"。

报告原文：

> The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles, and the backward passes are handled analogously.

## 1. 现在的规划器和原文的对照

代码：`torchtitan/models/kimi_k3/pipeline_parallel/dep_plan.py`，`k3_pp_mm` = `dep_review1` = `d27839459`（rebase 到 main `97e673b77` 后是 `1d03bfcc6`），函数 `plan_dep`。

| 原文 | 现在的代码 | 判断 |
|---|---|---|
| 前 PP 个 micro-batch 的前向在前面同步做 | `upfront = by_consume[:num_ranks]`，进 `to_prologue` | 一致 |
| 其余前向排进气泡 | `_encode_spot` 只在规划器的空闲段里整段装得下（`end <= run.stop`）且特征能在消费它的前向之前到达（`arrival <= due`）时才放；否则记入 `unplaced`，`to_prologue(unplaced)` 退回调度之前同步做 | 偏离：装不下就退回，是我们加的，原文没有 |
| 反向同理 | 每个反向都用 `_backward_spot` 找梯度到达之后整段装得下的空闲段，装不下的进 epilogue；没有"最后 PP 个 micro-batch 的反向在后面同步做、其余进气泡"这个固定结构 | 偏离 |
| （原文没说）气泡从哪来 | `_idle_runs(pipeline_order)`：torch 动作网格里的空格，按动作时间计（前向 1、反向 2、W / I 1，一个 slot 取该时刻最长的动作） | 只看得到网格里的空格。pp4 × vpp4 上模型里的气泡占步长 5% 到 9%；trace 实测调度内空闲每个 rank 约 25%（P2P 等待、stage 层数不均），规划器看不到 |

后果（H100，09-30，`DEP_H100_2026-09-30.md`）：
- 代价比例 3.61 下，其余编码大多被退回前面（例如 13 个里 11 个）；反向只有一格放进了 1 个，其余全在后面。
- bubble 对 K2.5 的步时在 −1.8% 到 +1.4% 之间，实测填充率 0% 到 2%。
- 按整个塔的工作量算，计划里藏住的比例：seq 2048 那轮 0% 到 8.6%，原档 7.4% 到 11%，seq 6144、M16 最多 21.5%（这一格 K2.5 OOM，没有对照）。

结论：
- bubble 这部分结果作废：测的是我们加的规则，不是报告的做法。
- 仍然成立：K2.5 对 DEP 关的步时（收益大头是 DEP 关时 stage 0 给每个 micro-batch 跑塔）、相对纯文本的效率、第 1 步数值差异的定位（共享的 flex attention 被 dynamo 按动态形状重编译）。

## 2. 修正（只按原文）

- **前向：** stage 0 消费顺序里前 PP 个 micro-batch 的编码，在调度前同步做（不变）。其余每个编码一律放进它的消费前向之前的某个空闲段（任意 rank），不再要求整段装得下。比空闲段长时就溢出，推迟该 rank 后面的动作。
- **反向：** stage 0 反向顺序里最后 PP 个 micro-batch 的反向，在调度之后同步做。其余每个反向放进它的梯度到达之后的某个空闲段，同样不要求装得下。
- **退回：** 只有在允许的时间窗里根本没有空闲段时，才退回前面或后面。这是边界情况，规划日志里单独计数。
- **代价比例：** 只用来选 rank 和空闲段（负载均衡），不再决定藏不藏。
- **进气泡的比例：** 由调度固定，前向和反向都是 (M − PP) / M。pp4 时 M8 是 50%，M16 是 75%。这和模型宽度、图片尺寸无关，就是用户说的"与模型大小关系小的固定比例"。和模型大小有关的只剩收益：编码比空闲段长时溢出多少，这一项由 H100 实测。
- **原文没规定、动手前要问用户的细节：**
  - 选哪个空闲段：最早、最晚，还是按负载在各 rank 间均摊；
  - 同一空闲段里多个编码的先后；
  - 溢出时是否允许拆到下一个空闲段（原文没有，倾向于不拆）。

## 3. 验证

- **CPU 单测**（`tests/unit_tests/cpu/test_kimi_k3_dep_plan.py`、`test_kimi_k3_vision_dep.py`）：
  - 去掉"装不下就退回"的期望。
  - 新增：任意代价比例下，进气泡的前向和反向个数都正好是 M − PP。
  - 原有的保持：传输仍在同一个 slot 边界挂出，每对 rank 的顺序一致；按 schedule 回放不卡死；四个 rank 用 gloo 时，step 1 的 loss 和梯度与单卡逐位一致。
- **5060（8 卡）：**
  - debug 宽度加一个放宽的宽度，配两种图片尺寸：规划日志里进气泡的比例都是 (M − PP) / M。
  - DEP 关、K2.5、bubble 第 1 步逐位一致。比较时关掉 `torch._dynamo.config.automatic_dynamic_shapes`，见 memory "flex compile shared across tower and text"。
  - trace 里视觉计算确实落在调度窗口内。
- **H100：**
  - 同一布局（pp4 × vpp4）重跑 bubble 对 K2.5，带 `PYTORCH_ALLOC_CONF=expandable_segments:True`。
  - 用 trace 分别记前向、反向藏住了多少时间：`ana_fill.py` 再按编码、反向拆开。
  - 档位由用户定：seq 2048 那套，还是原档（seq 4096 / 6144）。

## 4. 连带要改的

- PR #4381 body v5 的 Design 段写的是"装不下的进 prologue / epilogue"、`bubble_cost_ratio` 决定能否放进气泡，修正后要改写。
- 代码改在 review 分支 `dep_review1`；PR 分支 `k3_pp_mm` 等用户的话。
