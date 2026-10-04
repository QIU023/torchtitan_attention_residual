# DEP 对照 K3 报告 Figure 11 的复查和纠正（2026-10-04）

用户原话："这是K3报告里面PP DEP的排布，我怀疑我们实现歪了，可能还得稍微再调整一下，看图DEP插入气泡是非常规律的，但是我们没有做到，重新审查DEP分支，给出DEP review分支对应的纠正"（附 Figure 11）。

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
  - 反向在代价比 1 时，一个 ViT 反向（含重算）是 3 个单位，只有 PP2 结尾放得下 1 个。图里的方框是示意宽度，反向的具体编号无法照搬。
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
