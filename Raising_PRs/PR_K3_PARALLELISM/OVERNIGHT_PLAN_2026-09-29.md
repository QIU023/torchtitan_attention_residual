# Overnight 计划（2026-09-29 夜，等用户点头再开始）

用户："H100已经暂时冻结 现在三大目标制定overnight 给出计划
PR A和后续cache内存下界整条主线 在上游已经有一部分后 本地剩下一部分还是自己覆盖impl
DEP探索出合适的文本/视觉计算比例
MoonEP 跑不了 审计diff"

## 0. 资源和边界

- **机器：** 本机 8 × RTX 5060 Ti 16 GB，PCIe，driver 580（能跑 cu130），盘还剩 149 GB。H100 冻结，今晚不上机。
- **torch：** 现有的 `/workspace/venv_bfx9` 是 09-06 的 nightly，没有 torch#196463（接收缓冲按需分配）、per-edge P2P 和 unshard lookahead，main 要打兼容补丁才能跑。今晚第一步新建一个 venv，装 09-28 的 cu130 nightly（`2.15.0.dev20260928+cu130`，H100 上用的就是这一天的源码），和 CI 的 torch 语义一致，不再打补丁。
- **不碰：** 已发布的 PR 分支，#4656 `k3_ac_reuse_attention`、#4751 `k3_moonep_seam`、#4381 `k3_pp_mm`（DEP 今晚只做探索，不改代码）。#4765、#4764 的 draft 分支重叠后推不推，等你说（第 6 节）。
- **可以推：** logbook；review 分支 `pp_review_optimize`（PR A，还没开 PR），推之前先打备份 ref。
- **数字：** 5060 上的结果只进 logbook，body 里一律写 Pending (H100)。GPU 作业一次只跑一个。

## 1. 现状

| 线 | 分支 = head | 状态 |
|---|---|---|
| PR A | `pp_review_optimize` = `85eefa54b`（main `5dc97a3e7` 上，只含第 3 类） | 新 torch 上 2 个多卡 PP 单测失败：接收覆盖和 torch#196463 冲突 |
| PR A 第 1、2 类 | 本地 `pp_review_optimize_dev` = `1777ad806`（#4656 → PR A → 第 1、2 类） | 去掉接收覆盖后（`kit_h100_2026-09-29/pra/dev_send_only_probe.patch`），新 torch 上 71 个单测全过 |
| #4765 / #4764 | `k3_pp_offload` = `56f4cd3b0`，`k3_pp_balance` = `0a9034257`（叠在 dev 上） | 新 torch 上分别 4、7 个失败，原因同上；它们的改动不碰接收覆盖那几段 |
| DEP #4381 | `k3_pp_mm` = `dep_review1` = `refs/pull/4381/head` = `a03f74981` | H100 上正确性全过；计时显示在 cc12m-test 上视觉太轻，DEP 多花 4% |
| MoonEP #4751 | PR 分支 `k3_moonep_seam` = `f556ab4fd`（旧实现）；review `moonep_review1` = `a505f74a8` | H100 上真实 MoonEP 的 GPU 单测 2 passed；负载均衡一个数字都还没有 |
| upstream main | `a182e530a`，比 `5dc97a3e7` 多 9 个提交 | 都不碰 kimi_k3、pipeline_parallel、MoE 和 dispatcher |

## 2. 目标一：PR A 回到 cache 下界主线

**要做成什么：** 论文 §4.1 的下界，每个 block 在一个 rank 上跨 V 个 virtual stage 只存一份，并且只活到它在这个 rank 上的最后一次读取。torch 已经做了的部分（接收缓冲按需分配，torch#196463，09-22）交给 torch；torch 还没做的部分（send 一直扣到整步结束）继续在 K3 的 stage 里自己覆盖。

**T1a 新 venv（约 30 分钟）：** `/workspace/venv_0928`：torch 和 torchvision 09-28 cu130 nightly、main 的依赖、cutlass DSL 全家 4.6.2、mooncake 0.3.13.post1、pytest。验证：`torch.distributed.config.pipeline_per_edge_p2p` 在，`_recv_buffers.py` 在，BFX9 能开，KDA 在 SM120 上能跑，main `5dc97a3e7` 的 PP 单测 67 个全过。

**T1b 重建 PR A（约 1 小时）：**
- 内容：第 1 类（一个 block 一个张量，线上和 store 里都是；store 只存引用；stage 进门不再 `torch.stack` 出新 leaf；模型对外收发列表）、第 2 类（在带它进来的那个 stage 的反向时释放）、send 早等（前向 send 在本 stage 的反向时 wait，梯度 send 在能证明对端已用完的那个前向时 wait）。不再覆盖接收。
- 就是 H100 上验证过的那棵树（`1777ad806` 减去接收覆盖），压成 #4656 `5d469fdf3` 之上的一个提交。#4656 和最新 main 在 `model.py` 上有冲突，#4656 要不要 rebase 由你定，今晚 PR A 先跟着 #4656 的底。
- 规矩照旧：逐行读 diff，按"默认不加注释"清一遍新增的注释和 docstring，pyflakes，新 venv 上跑 PP 相关单测（71 个）全过才推。推之前 `pp_review_optimize` 的旧 head 打成 `backup/pp_review_optimize_pre_20260929`。本地的 dev 分支内容并入以后不再单独维护。

**T1c 下界的记账探针和模型（约 1.5 小时）：**
- 探针（本地，不提交）：在一步里的每个计算动作，记 block 相关的四样：store 引用的唯一 storage、stage 进门的拷贝、只被未 wait 的 send 扣住的张量、活着的接收缓冲；外加整卡 allocated。取每个 rank 的峰值时刻。
- 模型：按论文 §4.1 算每个 rank 的下界（每个 block 一份，按调度在最后一个读者处释放），复用 `pp_memory_model_v4b_2026-09-25.py` 和 `pp_prod_model_2026-09-27.py`，改成按实际的切分和 `pipeline_order` 算。
- 目的：结果表里每个 rank 写"block 实际占用 / 论文下界"，main 高出多少、PR A 是否贴住下界，一眼能看出来。

**T1d 5060 上验证（约 2 小时）：**
- 设置照 H100 那组，只有宽度不同：从 debug model 只放宽度，取 16 GB 放得下的 dim（先试 1024，放不下退 768），16 个 micro-batch × 2048 token，FullAC，AdamW，seed 42，c4 文本。
- 布局：4312 矩阵里 PR A 真正起作用的 interleaved 布局，pp4 × vpp2、pp4 × vpp4、pp2 × vpp2、pp2 × vpp4，外加 dp2 × pp2 × vpp2。
- 格子：#4656 对新 PR A，同一份预热缓存，20 步（100 步留给 H100）；记逐位相同的步数、每个 rank 的峰值和 T1c 的记账表。

**T1e body 和 torch issue（约 1 小时，穿插在 GPU 作业之间）：**
- PR A body v3：Summary 引论文 §4.1 那句 "each block is stored exactly once across all V virtual stages"；Design 讲逐 block、store 存引用、在带进来的 stage 的反向释放、send 早等为什么是释放生效的前提；写明接收缓冲由 torch#196463 负责。Results：Pending (H100)。
- torch issue 草稿只留 send 那一半，开头写明接收缓冲已由 #196463 解决。

**T1f #4765、#4764 重叠到新 PR A 上（约 1.5 小时）：** 只解冲突，不改功能；新 venv 上跑它们的 CPU 单测；5060 冒烟四格（`cpu_offload=all`、`planned`、只开 balance 走 tcp、planned 加 balance）。

**这条线后面还剩的（今晚只写进文档，不做）：**
- stage 的 hidden 输出一直留到反向：通用的 torch PP 问题，09-25 模型里是一个缺口。
- 聚合时每层把所有 block 叠成一条 fp32 `[T, N+1, D]` 的瞬时内存：论文 Algorithm 1 的两阶段加 online softmax 能去掉，生产尺寸下反向约多 14 GiB（推算）。这不属于 cache 下界，是单独的一项。

## 3. 目标二：DEP 的文本 / 视觉计算比例

**T2a 先定"比例"的口径（约 30 分钟）：**
- DEP 在意的是两件事：一次编码对一个文本 stage（前向加反向）的比例，以及各 micro-batch 之间图数的差别（K2.5 原文：按图像数或 patch 数把视觉前向摊到所有 GPU，消除 PP 和视觉 token 数造成的不均衡；K3 §5.2.3：把视觉的前向、反向排进气泡）。
- 用 K3 发布配置推算 K3 自己在哪个区间：塔 27 层、宽 1024；文本 93 层、宽 7168，896 个专家选 16；PP16 × VP2 左右，每个 micro-batch 0 到几张图。H100 那组（cc12m-test）的实测比例约 0.2，已经落在这个区间附近，要核对。

**T2b 数据集（约 1.5 小时）：**
- 源：先在本机确认能从 HF 下什么。要原生不小于 1024 的图、几千张以上（候选：text-to-image-2M 的 1024 子集，下载前核实大小和许可）。下不了就用真实图片加本地固定尺寸 resize。
- 打成本地 tar（`tests/assets/cc12m_test/pack_test_dataset.py` 的格式），几千个样本，每个样本 0 到 k 张图，分辨率 448、768、1024 三档。
- 本地 recipe：`max_patches` 放到至少 5184、每边至少 72；换成固定尺寸 resize（K3 的 resize 只缩不放）；短 seq（1024 到 2048）。
- 上机前抽几个 micro-batch 核对 `num_valid_tokens`、图数和 patch 数。cc12m-test 那次就是没核对，才没发现一个 micro-batch 只有 162 个有效 token、一张 192 patch 的图。

**T2c 5060 上量比例曲线（约 2 小时）：**
- 单卡：塔的前向加反向，对 pp2 × vpp4 切分下一个文本 stage 的前向加反向，dim 1024 和 2048（16 GB 放得下的宽度），分辨率三档，seq 1024 和 2048。塔和文本同宽一起放大，这个比例基本不随宽度变，可以外推到 6144；同时标出 K3 真实塔（窄得多）的推算值。
- 8 卡小宽度冒烟：DEP 关、K2.5、bubble 在新数据上各跑 10 步，看 rc 和计划日志（多少编码、反向进了空闲槽）。5060 走 PCIe，计时只作参考。

**T2d 产出：** 比例表（实测加 K3 推算），选三档给 H100（K3 所在的区间、1 左右、视觉很重），写成下次 H100 的格子清单、数据集说明和预计时间。

## 4. 目标三：MoonEP 审计

**T3a 逐行审 `moonep_review1` = `a505f74a8` 的 diff（约 1.5 小时）：**
- 对照你的原则：MoonEP 库的调用之外尽量少改；默认不加注释；#4577 的 docstring 标准；flavor 规则。
- 对照 MoonEP 公开版 `33327eb`：本地 clone 读源码和 README，逐个调用核对参数、对齐、池的大小、barrier 语义。
- 对照 main 现在的 seam（`RoutedExperts`、token dispatcher、`config_utils`），看有没有能复用却没用的。
- upstream main 自 `5dc97a3e7` 进了 9 个提交，都不碰 MoE 和 dispatcher；试一下能否干净 rebase 到 `a182e530a`。

**T3b 负载均衡的证据方案（约 1.5 小时）：**
- 探针：每层每个 rank 路由到的 token 数。标准 EP 下的 max / mean，MoonEP 下是否正好 S × K。数据取 MoE 里现成的 `tokens_per_expert_E`。
- 本地 recipe：放大专家（128 或 256 个，top-8 或 top-16；K3 是 896 选 16），宽度保持 debug 那么小，跑得快。
- 失衡的来源：router bias 按 Zipf 分布人为偏置，造出热门专家；再配一格自然路由作对照。
- 用本地的假 MoonEP（`kit_moonep_rewrite_2026-09-29/local/fake_moonep/`）在 5060 上把探针和 recipe 跑通。假包不做真的预取，只验证记账和流程。

**T3c 产出：** `DIFF_AUDIT_MOONEP_2026-09-28.md` 加一节；body v2 的 Test plan 和 Results 口径改成负载统计加步时；下次 H100 的格子清单（负载表、步时表、各自的时间）。

## 5. 顺序和时间

- GPU 作业串行：T1a 验证 → T1d → T1f 冒烟 → T2c → T3b 跑通。
- CPU 的活穿插在 GPU 作业之间：T1b、T1c、T1e、T2a、T2b、T3a。
- 每做完一项就推 logbook；结果汇总进 `OVERNIGHT_RESULTS_2026-09-29.md`。
- 预计 10 到 12 小时。

## 6. 需要你定的

1. PR A 的新范围：第 1、2 类加 send 早等，不再覆盖接收；`pp_review_optimize` 可以重写（先打备份 ref）。
2. #4765、#4764 重叠到新 PR A 以后，推不推它们的 draft 分支。
3. DEP 的数据集：能不能从 HF 下（大小、许可），还是只用真实图片加本地固定尺寸 resize。
4. `moonep_review1` rebase 到最新 main 以后能不能推（review 分支，照规矩可以推，先问一下）。
