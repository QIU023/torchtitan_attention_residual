# DEP 对照 K3 报告 Figure 11 的复查和纠正（2026-10-04）

用户原话："这是K3报告里面PP DEP的排布，我怀疑我们实现歪了，可能还得稍微再调整一下，看图DEP插入气泡是非常规律的，但是我们没有做到，重新审查DEP分支，给出DEP review分支对应的纠正"（附 Figure 11）。

第二轮，用户引了我第一轮的结论"反向在这个配置下没法照搬图里的编号：一个 ViT 反向加上重算要 3 个单位，图里的方框只是示意宽度"，回复："为什么？不要随便否认图里面的正确性，并且现在的DEP body还得想办法在PR head代码里面yield和图片完全一致的stage排布"。第二轮的结果见文末"第二轮"三节，第一轮"纠正"一节里关于反向的那条结论作废。

## 报告怎么说

§5.2.3 "Encoder computation in PP bubbles"：在 interleaved 1F1B 下，靠前 micro-batch 的文本前向都排在最开头，靠后 micro-batch 的文本反向到最后才结束。"The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles, and the backward passes are handled analogously."

Figure 11 是 3 段 PP、6 个 micro-batch 的示意图：

- ViT 前向全部在每个 rank 的开头：
  - PP0 只做 mb1，开头同步做；
  - PP1 先同步做 mb2，再在自己的 warmup 气泡里做 mb4；
  - PP2 先同步做 mb3，再在气泡里做 mb5、mb6。
- ViT 反向全部在每个 rank 的结尾：
  - PP0 做 mb4，PP1 做 mb1、mb5，PP2 做 mb2、mb3、mb6；
  - 其中最后 3 个 micro-batch（4、5、6）在结尾同步做，每个 rank 一个，其余的放进 cooldown 气泡。
- 中段的稳定 1F1B 阶段没有任何 ViT 工作。

## 我们的规划器错在哪（review head `6f5312fab`）

`vision_dep/plan.py` 的 `_encode_spot` / `_backward_spot` 在**每个 rank 的每个空闲段**里找能放下的位置，再按 `(负载, 结束时间, rank)` 选 rank。PP0 没有开头的气泡，负载最小，于是总被选中，把 ViT 工作放进它中段的小空隙里：

- **3×4、6 个 mb、代价比 1**（`timelines.txt`）：mb6 的前向被放进 PP0 在 warmup 之后的空隙 [13,18)，图里它在 PP2 的开头。
- **4×4、8 个 mb、代价比 0.135**（H100 上的形状）：
  - mb8 的前向在 PP0 的 [19,19.1)；
  - mb1 的反向在 PP0 的 [67,67.4)。
- **4×4、16 个 mb**：
  - mb 8、12、16 的前向在 PP0 中段；
  - mb 1、5、9、12 的反向散落在 PP0 的 [163,170)。

开头同步做的规则（前 pp 个 micro-batch）本来就和报告一致，不用改。

## 纠正（`dep_review1` = `6b580e438`，在 `6f5312fab` 上一个提交）

- 其余的前向只放进每个 rank **开头的空闲段**（rank 第一个文本动作之前），反向只放进每个 rank **结尾的空闲段**（最后一个文本动作之后）。
- 在这些段里选**结束最早**的位置，时间相同时选低 rank。原来按负载挑 rank 的做法去掉了。
- 放不下的照旧在 schedule 之前或之后做，在各 rank 之间均衡分配。
- 结果：
  - 3×4、6 个 mb、代价比 1：前向和 Figure 11 完全一样（PP1 放 mb4，PP2 放 mb5、mb6）。
  - 4×4：PP0 中段没有任何 ViT 工作，前向在 PP1 到 PP3 的开头轮流放，反向从 PP3 开始填结尾的气泡。
  - ~~反向在代价比 1 时，一个 ViT 反向（含重算）是 3 个单位，只有 PP2 结尾放得下 1 个。图里的方框是示意宽度，反向的具体编号无法照搬。~~ 这条错了：图是按比例画的，反向编号对不上是规划器的规则造成的，见"第二轮"。
- 两份 ASCII 时间线在 `kit_dep_bands_2026-10-04/timelines.txt`：数字是 ViT 前向，字母是 ViT 反向，竖线左边是 schedule 前做的，右边是 schedule 后做的。
- 测试：
  - 新增两个规划器测试：各种形状和代价比下，放进气泡的前向都在开头段、反向都在结尾段；3×4、6 个 mb 时排布和图一致。这两个测试拿到旧规划器上会失败（7 个失败），新规划器全过。
  - 运行时的 CPU 测试和 NCCL 测试里，"反向在自己的空闲段里等梯度"的那个用例，现在落在 rank 1 的 mb6：段从 56 开始，梯度 56 就绪，加上传输，57 开始。
  - CPU：torch 0928 下 20 passed，37 个 subtest。DEP 分支基于 `db050eb3f`，用 10-03 的 nightly 时，改动前的 head 自己也有 4 个运行时测试失败，是环境问题。
  - pyrefly 和 review head 一样，是同一组 17 个。
- PR 分支 `k3_pp_mm` 不动。

## 还要做的

- body 的隐藏率表（73/81/84%）是用旧规划器测的，要在 H100 上用新规划器重测：`kit_dep_bands_2026-10-04/h100_dep_bands.sh`。数值一格是新旧逐位比对；隐藏率每档先用 K2.5 模式测出代价比，再新旧各测一次。

## 本机 5060 的检查（torch 0928，不进 body）

- NCCL 单测 `tests/unit_tests/gpu/test_kimi_k3_vision_dep.py`：`6b580e438` 上 3 passed，包括"前向和反向放在空闲槽里"以及"反向在空闲段里等梯度"两个用例。它们按新排布在真 GPU 上跑，并和单卡参考比对梯度与 loss。
- B200 suite 的 8 卡 DEP 格子（FSDP 2 × TP 2 × EP 2 × PP 2 × VPP 4）：新旧两版 10 步逐位相同。但这个 pp2 格子没有能用的气泡，两边日志都是 "encodes 4 before the schedule, 0 in idle slots; backwards 0 in idle slots, 4 after it"，所以它只证明这条路径没被破坏，测不到新排布。新排布的端到端效果要看 H100 上 pp4 × vpp4 的那组。

## 推送（10-04）

- `dep_review1` = `6b580e438`，从 `6f5312fab` 快进。
- PR 分支 `k3_pp_mm` 仍是 `6f5312fab`，没推。

## 第二轮：把图 11 逐格量出来（10-04）

方法：报告 PDF 第 19 页按 500 dpi 渲染，沿三行扫出每个方框的左右边界和填充色，再用 `pdftotext -bbox` 的数字位置给方框编号（`kit_dep_bands_2026-10-04/fig11/scan.py`、`transcribe.py`，结果 `fig11.json`）。单位取一个文本前向的宽度。

- **宽度都是按比例的：** 文本前向 1，文本反向 2；ViT 前向 0.5；ViT 反向 1，正好是 ViT 前向的两倍，和文本反向对前向的比例一样。一个格子都不是"示意宽度"。
- **文本部分就是 torch 的 interleaved 1F1B：** 每行 48 个文本动作，顺序和 `ScheduleInterleaved1F1B`（pp3 × vp4，6 个 micro-batch）的 `pipeline_order` 逐个相同（`order.py`）。
- **时间轴是"锁步 1F1B 节奏"：** warmup 阶段全是前向格；出现第一个能做的反向之后，前向格和反向格交替，直到前向做完；之后只剩反向格。每个 rank 在每一格做自己的下一个动作，前提是动作种类对、输入在更早的格里已经产出。这个模型和图里 144 个文本方框全部对上（`sim_rhythm.py`）。对照：规划器用的 torch 锁步网格在 warmup 转 1F1B 处多一格，后面整体晚 1 个单位，112 个对不上（`sim_grid.py`）；只看依赖的事件模型 53 个对不上；Megatron 那种每步阻塞交换 141 个对上（`sim_megatron*.py`）。
- **ViT 部分：** 前 3 个前向在调度前，每个 rank 一个（PP0 1、PP1 2、PP2 3）；PP1 的开头气泡做 4，PP2 的做 5、6，紧挨着调度前那一列；后 3 个反向在调度后，每个 rank 一个（PP0 4、PP1 5、PP2 6）；PP1 的结尾气泡做 1，PP2 的做 2、3，紧挨着最后那一列；稳定阶段没有 ViT。PP1 的气泡是 1 格、放 1 个，PP2 的是 2 格、放 2 个，都只用了一半。

## 第二轮：我第一轮哪里错了

- "图里的方框是示意宽度"没有根据，量出来每个格子都按比例。
- 反向编号对不上，原因在我们的规划器，不在图：
  - 选位置用"最早结束、同时取低 rank"。PP2 的结尾气泡比 PP1 的早开始两个单位，所以 mb1 的反向总落到 PP2；代价比小于 1 时，PP1 的开头气泡还放得下第二个编码，mb6 的前向也会落到 PP1。图里是按气泡长短分：PP1 一个、PP2 两个，低 rank 拿前面的 micro-batch。
  - 后 PP 个 micro-batch 的反向也被放进气泡。报告原文"the backward passes are handled analogously"对应的是"后 PP 个的反向在调度后同步做"，图里 4、5、6 正是在最后一列。
  - 结尾气泡里的反向从气泡开头往后排，图里是贴着最后一列往前排。
  - 我当时说的"3 个单位"是代价比 1 时我们的模型值（ViT 反向算 3 个前向：K2.5 Stage 3 的重算加反向；再加 1 个单位的传输预留）。图里 ViT 前向只有半个文本前向，这个代价比不是图的。

## 第二轮：纠正（`dep_review1` = `5b01a6932`，替换 `6b580e438`）

- **规则：**
  - 调度里前 PP 个 micro-batch 的编码在调度前做、后 PP 个的反向在调度后做，按 patch 数在 rank 间均衡（每个 rank 一个）。按调度位置取，不再按"带图的前 PP 个"取，和原文一致。
  - 其余的编码放进各 rank 开头的空闲段，紧接调度前的编码；其余的反向放进各 rank 结尾的空闲段，贴着段尾往前排。
  - 选段：按"已占比例"（已用时间 ÷ 段长）取最空的段，相同取低 rank。这样 rank 分到的量和它的气泡长短成正比。
  - 放不下的照旧退回调度前后。
- **复现图 11：**
  - 用图自己的比例（ViT 前向 0.5、反向是前向的两倍）跑规划器，把结果画到图的时间轴上（文本用 torch 的顺序加锁步 1F1B 节奏，ViT 用规划器给的位置），156 个方框（144 个文本、12 个 ViT）和图逐个相同（`reproduce.py`）。对照图：`fig11/fig11_vs_plan.png`，上面是 PDF 原图，下面是规划器的结果。
  - 用我们自己的代价（ViT 反向 3 个前向，含重算），代价比 0.01 到 1/3 时排布和图完全一样，ViT 反向的位置也一样（1/3 时宽度正好 1）。K3 的区间是 0.02 到 0.32，H100 实测 0.046 到 0.170，都在里面（`check_plan.py`）。
  - 超过 1/3 时，PP1 的结尾气泡（一个反向格，2 个单位，减去 1 个单位的传输预留）放不下一次 3 倍的 ViT 反向，mb1 改去 PP2、mb3 退回调度后（0.34、0.5、1.0 都是这样）。用图的两倍比例时这个界是 0.5（`check_fig_costs.py`）。
- **测试：**
  - 新增 `test_three_ranks_and_six_microbatches_lay_out_as_in_the_k3_report`：代价比 0.05、0.1、0.2、0.3 下，调度前后的分配和 6 个气泡工作的 rank 与起止时刻都和图一致。新增 `test_the_first_and_last_pipeline_degree_microbatches_run_outside_the_schedule`：三种形状下，气泡放得下时也只有前、后 PP 个在调度外。这两个测试在 `6f5312fab` 和 `6b580e438` 的规划器上各失败 7 个 subtest（`run_with_old.py`）。
  - `test_a_backward_uses_the_rest_of_an_idle_run_that_began_before_its_gradient` 改成手写顺序里再加两个 stage 0 的反向，让 mb0 不属于"后 PP 个"；代价比 2 时反向正好占满 [7, 13)（梯度 6 就绪加传输 1），2.1 时放不下、退回调度后。
  - 运行时测试里"反向在空闲段里等晚到的梯度"那个用例（CPU 和 NCCL 各一个）删掉了：后 PP 个改成调度后同步做以后，interleaved 1F1B 下气泡里的反向，梯度都在它的空闲段开始之前就绪，这条路径走不到了（4 卡 8 个 micro-batch 的形状里，梯度在 32 到 41 就绪，段从 52 到 56 开始）。规划器单测里的手写顺序还覆盖它。
  - CPU：torch 0928 下 20 passed、46 个 subtest（规划器 15 个、gloo 运行时 5 个）。ufmt、pyflakes 干净；flake8 只有一条 B905，在没改的旧测试里，`6b580e438` 上也有。pyrefly 17 个，和之前同一组，都不在 vision_dep。
  - 不开 bubble 的 K2.5 形式：随机 592 组输入，新旧规划器的所有输出逐项相同（`k25_same.py`）。B200 那个 8 卡格子开的就是这个形式，所以不用重跑。
- **本机 5060（不进 body）：** NCCL 单测 `tests/unit_tests/gpu/test_kimi_k3_vision_dep.py` 在 4 张卡上 2 passed（K2.5 形式和 bubble 形式，都和单卡参考比对梯度与 loss）。用卡前后都通知了 SATS-OPRD 会话。

## 第二轮：影响和待办

- 进气泡的个数现在是固定的：前向、反向都是 M − PP 个（带图的），对应报告的结构。以前代价比小时后 PP 个的反向也能进气泡（比如 pp4 × vpp4、M16、0.1 时 16 个里 15 个），现在固定 12 个。但调度后那 PP 个是每个 rank 各一个、并行做的，墙钟上只占一次 ViT 反向；以前最后一个 micro-batch 的反向反正也只能在调度后做，同样占一次。按模型，步长不变。
- body 里的隐藏率表（73/81/84%）和 `bubble=True` 的 100 步数值是旧放置规则下测的，要在 H100 上用 `5b01a6932` 重测：`h100_dep_bands.sh` 已改成新 SHA，probe 补丁在新树上能打上。
- `k3_pp_mm` 仍是 `6f5312fab`，同步要等用户的话。

## 第二轮：推送（10-04）

- `dep_review1` 从 `6b580e438` force-with-lease 推到 `5b01a6932`，旧 head 备份为 `backup/dep_review1_pre_20261004`。提交没有 trailer。
