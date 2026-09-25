# 理论内存下界审计：对照 AttnRes 原文（2026-09-25）

原文：Attention Residuals（Kimi Team，arXiv 2603.15031）"Pipeline communication""Cross-stage caching""Memory overhead"三段，以及 Fig. 3、Eq. 7/8。K3 报告 §5.2 的相关一句见 `K3_2P8T_PP_VP_MEMORY_2026-09-24.md` §1。

对照的白天结论：
- `K3_2P8T_PP_VP_MEMORY_2026-09-24.md` §4；
- 以及修正它的 `PP_OPTIMIZE_REPORT_2026-09-24.md` §4。

**数字说明：** 本文所有 H100、GB300 数字都是模型值；5060 探针的 13.38 → 8.04 GiB 是实测，其余探针数字也是模型值。

## 0. 结论

- **口径一致。** 白天用的"理论下界"与原文、K3 报告一致：
  - 每个 block 在每个 rank 上只存一份，对应原文 "stored exactly once across all V virtual stages"；
  - micro-batch 结束时释放，对应 K3 报告 "released as soon as the micro-batch finishes"。

  在这个口径下，白天的几条结论都成立：pp_review4 离下界有数倍差距；V4 把四项差距修掉就正好到下界；H100 上 block+hidden 的下界是 18 到 27 GiB（以 PP_OPTIMIZE_REPORT 的修正版为准）。
- **这个口径不是最紧的。** 一个 block 在某个 rank 上，按反向顺序的最后一个读者，就是把它带到这个 rank 的 stage（下称 bringer）；deposit 也在 bringer 反向时被收走。所以 bringer 反向一结束就可以释放这个 block，不必等整个 micro-batch 结束。
  - 按这个更紧的下界算，最重 rank 的 block+hidden 还能再降 11% 到 26%（模型值，见 §3）。
  - 要做到这一点，store 需要按 block 分配（V4 剩余差距里"store 整块分配"那一项），并在 bringer 反向时释放。
- **"可忽略"有前提。** 原文说 block 相对逐层激活"可忽略"，这在报告的配方下成立，因为那时逐层激活很大（a 约 15 到 17）。在 FullAC（a=1）下，即使到了下界，block 显存也和在飞的逐层激活相当；pp_review4 更是它的数倍。所以对我们在 5060 和 CI 上跑的 FullAC，block 显存不能忽略。
- **机制与原文一致。** 只做相邻传递；每一跳只传接收 rank 还没缓存的增量；反向用同样的方案（deposit）。原文没有跳发。原文 Eq. 8 是近似（每跳约 PN_p 个），我们传的是精确增量，所以总量更少。

## 1. 原文逐句对照

| 原文 | 含义 | 我们的对应 |
|---|---|---|
| "each block is stored exactly once across all V virtual stages" | 每个 rank、每个 micro-batch，每个 block 只存一份 | 这就是 K3_2P8T §4.1 "理论下界"的定义。pp_review4 不满足：每个 stage 各有一份 `torch.stack` 副本和一份 payload 副本，torch 还按 micro-batch 常驻接收缓冲。V4 用 view 加就地接收基本做到，剩下四项见 PP_OPTIMIZE §4 |
| "released as soon as the micro-batch finishes"（K3 报告） | 释放点是该 rank 上这个 micro-batch 的最后一次反向 | 与模型里的 `lower` 方案相同。更紧的释放点见 §3 |
| "the per-layer activation footprint remains identical … activation checkpointing eliminates all inter-block attention intermediates … p_l matches h_l" | block 之外的逐层激活与标准结构相同 | 前提是 AttnRes 聚合在 AC 下重算。FullAC 满足；选择性 AC 需要 PR 4780 的 remat region，它不在 4312 里 |
| "Caching reduces peak per-transition cost from O(C) to O(P) … The backward pass benefits from the same scheme" | 通信峰值是 O(P)，反向同理 | 一致：峰值 p−1，反向用 deposit |
| Eq. 8："~PN_p incremental blocks" per transition | 每跳增量的近似值 | 我们用精确增量（已累积的减去接收 rank 已有的）。每个 stage 一个 block 时，总量是 P(P−1)(2V−1)/2，比 Eq. 8 少 (V−1)P |
| Fig. 3（P=4，V=2，每个 block 跨两个 stage） | 原文说第二个 virtual stage "eliminates 6 redundant block transmissions"，即按每跳 2 个计 | 精确值为 2/1/2/1，省下 8 个。原文图里 rank 1、rank 3 的 "+[…]" 把本 rank 自己产生的 block 也算进了增量 |
| Fig. 3 的箭头 | 同一 virtual stage 内从 rank r 到 r+1，跨 virtual stage 从 rank P−1 回到 rank 0 | 原文只有相邻传递，没有跳发。本 PR 与此相同 |

## 2. 白天结论回顾（是否仍成立）

- **K3_2P8T（09-24 晚）：**
  - 结论：pp_review4 是下界的 3 到 5 倍；H100 FullAC 下最重 rank 约 112 GiB，下界约 68 GiB。
  - 已知修正：它的"本 PR"那一行少算了被钉住的 send，约 200 个单位，以 PP_OPTIMIZE_REPORT 为准。
  - 下界的定义本身与原文一致，所以下界这一行不受影响。
- **PP_OPTIMIZE_REPORT（09-24）：**
  - 实测：5060 探针上 V4 让最重 rank 从 13.38 GiB 降到 8.04 GiB，数值逐位一致。
  - 模型（H100，block+hidden）：pp_review4 是 72 到 146 GiB，V4 是 30 到 55 GiB，下界是 18 到 27 GiB。
  - V4 到下界的四项差距：
    - 反向 send 被钉到 step 结束（torch 侧）；
    - store 整块分配；
    - hidden 输出留到反向；
    - stage 中间开 block 时 `cat` 出的副本（模型侧）。
  - 本次复核：结论在原文口径下成立。
- **本次新增：** 原文口径之下还有一个更紧的下界（§3）；另外"可忽略"是有前提的（§4）。

## 3. 更紧的下界（新）

**推理：**
- block b 在 rank r 上的读者，是 r 上从 bringer（b 到达或产生的那个 stage）开始的各个 stage。
- 反向按 stage 逆序进行，所以 bringer 是最后一个处理的读者。b 的 deposit 也在 bringer 反向时被收走。
- 因此 b 在 r 上的寿命可以是 [bringer 前向, bringer 反向]，而不是 [到达, 该 micro-batch 在 r 上的最后一次反向]。
- 在 virtual stage 0 到达的 block，两种算法没有区别；到达得越晚，省得越多。

**模型结果**（`pp_memory_model_tight_2026-09-25.py`，输出见同名 `.out.txt`）。它在 v2 模型上只加了 `tight` 方案，其余假设不变：torch 实测的 send 钉住与接收缓冲行为、block+hidden、不含逐层激活。

| 切分 | V4 最高 / 均值 GiB | 下界（报告口径）最高 / 均值 | 更紧下界 最高 / 均值 | 更紧 vs 报告口径（最高） |
|---|---|---|---|---|
| 5060 探针 PP8×VP2，32 层，block 4，seq 3584 | 3.24 / 2.82 | 1.76 / 1.59 | 1.33 / 1.17 | −0.44 GiB（−25%） |
| H100 PP8×VP4，93 层，block 12，8K×7168，M=16 | 43.1 / 34.8 | 20.9 / 19.2 | 17.4 / 15.8 | −3.5 GiB（−17%） |
| H100 PP16×VP2，M=32 | 54.9 / 45.8 | 26.6 / 23.5 | 19.6 / 16.8 | −7.0 GiB（−26%） |
| GB300 PP4×VP4，M=16 | 28.2 / 26.6 | 10.4 / 9.6 | 8.6 / 7.9 | −1.8 GiB（−17%） |
| GB300 PP2×VP8，M=16 | 33.8 / 30.0 | 8.0 / 8.0 | 7.1 / 7.1 | −0.9 GiB（−11%） |

**实现条件：**
- store 按 block 分配，对应 V4 模型里的 `per_block` 那一项；
- store 在 bringer 反向时释放这个 block。读者的 AC 保存的是 store 的 view，而在那之前它们的反向都已经完成。

**标注：** 以上全是模型值。5060 上要实测，需要先实现按 block 分配的 store。

## 4. "相对逐层激活可忽略"的前提

数据来自 K3_2P8T §4.2，H100 PP8×VP4，单位是一个 micro-batch 的 T×d：

| 各 rank | 在飞逐层激活（a=1，FullAC） | 下界时的 block | pp_review4 的 block（旧模型，偏低） |
|---|---|---|---|
| 范围 | 75 到 111 | 106 到 126 | 339 到 533 |

- **a=1：** 下界时的 block 和逐层激活相当，pp_review4 是它的 3 到 5 倍。原文所说的"可忽略"不成立。
- **a=17（报告的 FP8 配方）：** 逐层激活放大 17 倍，block 只占约 6% 到 9%，原文的说法成立。

## 5. cache/stage 机制是否正确

**与原文一致，证据如下：**
- `test_kimi_k3_pp_block_grads`：4 个 gloo rank，Interleaved1F1B，cache 开和关，block 梯度与单卡逐位一致；
- layout 表：零重复投递，总量是精确的最小值（与原文近似值的差见 §1）；
- H100 body 表：关 cache 的各行 100 步逐位一致；开 cache 的行 step 1 逐位一致；
- 原文没有跳发，本 PR 也没有。

**不一致只在显存。** pp_review4 存了副本，离下界有数倍差距；V4 在 draft #4765 的栈里。

## 6. 后续（未开工）

- **到达报告口径的下界：** 修掉 V4 的四项差距。模型侧改成 block 列表以去掉 `cat` 副本，store 按 block 分配；torch 侧要让反向 send 不再被钉到 step 结束。
- **到达更紧的下界：** 在前面的基础上，store 在 bringer 反向时释放 block。
- **实测：** 在 5060 探针上实测上面两步，才能确认 §3 的模型值。
