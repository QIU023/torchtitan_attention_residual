# 2.8T Kimi K3 预训练：5D 切分估算，以及 vp>1 时各 PP rank 的显存时间线（2026-09-24）

> **更正（同日晚）**：本文的切分推荐和“本 PR”一行的数字已被 `PP_OPTIMIZE_REPORT_2026-09-24.md` 取代。
> - PP8 × VP4 和 PP4 × VP8 只是可行域里的一个点，报告并没有给出并行度。
> - 本文漏了 MTP 层、MoonEP 的缓冲，也没算 torch 把被发送张量钉到 step 末的开销，“本 PR”一行因此低估了约 200 个单位。
> - 用 8 卡实测校准过的模型见 `pp_memory_model_v2_2026-09-24.py`。

脚本：`pp_memory_timeline_2026-09-24.py`（本目录；需在 pp_review4 的 checkout 里运行，调用 PR 自己的 layout 表和 torch 的 `ScheduleInterleaved1F1B` 排程）。原始输出：`pp_memory_timeline_2026-09-24.out.txt`。

## 0. 结论

- **推荐切分（估算）**
  - H100 80 GB（8 卡/节点 NVLink，IB 400 Gb/s/卡）：PP8 × VP4，EP64（跨 8 节点走 IB），TP1，CP 在 8K 阶段为 1、64K 阶段为 8，1024 路数据并行（N=8192），专家参数的 ZeRO-1 分片 16 路。每步 16 个 8K micro-batch，约 1.34 亿 token/步，气泡约 11%。
  - GB300 288 GB（NVL72，机柜外 IB 400 Gb/s/卡）：PP4 × VP8，EP64（机柜内 NVLink），其余同上（N=4096）。每步 8 个 micro-batch，约 6700 万 token/步，气泡约 4.7%。
- **显存结论（vp>1，8K token/micro-batch）**
  - 本 PR 当前实现的 AttnRes block 显存是理论下界的 3 到 5 倍。
    - 这不是拓扑造成的，来自三件事：每个 stage 前向都用 `torch.stack` 复制一份整条 stack 并留到反向；发出的 payload 也是一份副本，留到反向；torch 的 `PipelineStage` 为每个 micro-batch 常驻分配接收缓冲，delta 和 payload 梯度的缓冲都在其中。
    - H100 上即使每层完整重算，最重的 rank 也要约 112 GiB，放不下；按下界实现约 68 GiB。
  - jinsooihm 的混合排布：直发缓冲按需分配时，H100 最重的 rank 能降约 10 GiB，GB300 基本不变；直发也按 micro-batch 常驻分配时，最重的 rank 反而更高。只比拓扑、两边缓冲都按需分配时，本 PR 更低。
  - offload 草案（#4765）在这两种切分下省不下任何显存。它停放的 block 要么是 torch 常驻接收缓冲的别名，要么被同一 stage 下一层的 AC 引用、留到反向。
  - balance 草案（#4764）能把各 PP rank 的峰值拉平。但它只能搬 autograd 保存的层激活，搬不动 torch 持有的缓冲和 `fwd_cache` 里的 stack 副本。

## 1. 锚点与假设

**K3 技术报告 §5.2（`/tmp/k3report.txt` L1264-1400）**
- 并行维度：PP 加虚拟 stage（VP）、EP、ZeRO-1 DP、Pipeline ZeRO-2 梯度分片、CP。报告没列 TP。
- 激活：“most activations use block-wise FP8 quantization combined with offload/remote-offload, and element-wise operators are configured with recomputation”。
- AttnRes：checkpoint 包裹，PP 采用“cache-based pipeline communication … only newly generated blocks are incrementally transferred between stages and released as soon as the micro-batch finishes, reaching the theoretical lower bound on memory footprint”。
- PP rank 平衡：“Under interleaved 1F1B … the number of resident activations decreases as the PP rank increases … we remotely offload activations to the memory of other PP ranks using the Mooncake Transfer Engine”。
- 梯度：分片后放 CPU，GPU 上保留双梯度缓冲。
- 报告没有给出具体并行度数值。

**模型（in-tree 的 `Kimi-K3` flavor，在 meta device 上实测）**
- 总参数 2.780T：专家 2722.7B，即 92 个 MoE 层，每层 29.6B；非专家 54.4B（KDA 层每层 0.634B，MLA 层每层 0.422B，第 0 层为 dense，1.17B）；embedding 和 lm_head 各 1.17B；视觉编码器 0.45B。
- 每 token 激活约 104B。
- 结构：93 层，dim 7168，AttnRes block 大小 12，共 8 个 block；896 个专家，top-16，专家在 latent 维度 3584 上计算。

**硬件与单位**
- H100 80 GB 即 74.5 GiB；GB300 288 GB 即 268 GiB。IB 400 Gb/s 按每卡每方向 50 GB/s 计。如果 400 GBps 指的是字面上的 400 GB/s，下面的通信都不成问题，显存结论不变。
- 1 个单位 = 一个 micro-batch 的一个 block，即 8192 × 7168 × 2 字节，约 0.109 GiB。hidden state 也是这个大小。
- a = 每层每个 micro-batch 为反向保存的激活，取两档：a=1，即每层完整重算、只存层输入，是下限；a=17，是 FP8 存储加逐元素重算、不 offload 的粗估（bf16 下 KDA+MoE 层约 35 个单位）。
- 静态显存：bf16 参数；ZeRO-1 下 fp32 master 加 Muon 动量共 8 B/参数（专家按 16 路分片，非专家按 1024 路）；梯度放 CPU，GPU 上 2 GiB 双缓冲。

## 2. 5D 取值

| 维度 | H100 | GB300 | 依据 |
|---|---|---|---|
| EP | 64，跨 8 节点 IB | 64，NVL72 机柜内 | 896 个专家时 EP 取 64，每卡 14 个专家。H100 若 EP=8 留在节点内，每层每卡专家 3.7B 参数，放不下。每层每个 micro-batch 的 all-to-all 量：FP8 latent 分发约 470 MB，bf16 合并约 940 MB。在 IB 上约 25 ms，与每层约 24 ms 的 FP8 计算相当，所以 H100 必须靠重叠（报告中的 MoonEP 和 overlap）；在 NVLink 上约 1.6 ms。 |
| PP | 8 | 4 | 每卡参数 = 专家/(PP×EP) + 非专家/PP。H100 PP8 时约 24 GiB bf16，静态共约 30 GiB；PP4 时约 57 GiB，放不下。GB300 PP4 时静态约 57 GiB，288 GB 绰绰有余。 |
| VP | 4 | 8 | 让 PP×VP=32 段，每段约 3 层；气泡为 (PP−1)/(VP×M)。 |
| micro-batch M | 16 | 8 | torch 的 Interleaved1F1B 要求 M 能被轮数整除；1024 路数据并行 × M × 8K 约为 1.34 亿（H100）和 0.67 亿（GB300）token/步。 |
| CP | 8K 时为 1，64K 时为 8 | 同左 | 保持每卡每 micro-batch 8K token。冷却期 256K 到 1M 时 CP 取 32 到 128，M 相应减小。 |
| TP | 1 | 1 | 报告未列；attention 每层 0.4 到 0.6B，TP 只会增加每层的 all-reduce。 |
| DP | 1024 路数据并行；专家 ZeRO-1 16 路 | 同左（N=4096） | N/(PP×CP) 路数据并行，EP rank 也各自处理自己的数据。 |

PP 的 P2P 走 IB：每跳 hidden 117 MB 约 2.4 ms；delta 峰值在 H100 切分下为 2 个 block、在 GB300 切分下为 1 个 block，另加 2.4 到 4.7 ms。每段 3 层的前向在 H100 上约 70 ms，在 GB300 上约 30 ms，都能重叠掉。

## 3. PP 少、VP 大的利弊

**好处**
- 气泡 (PP−1)/(VP×M) 更小，M 可以更小。
- AttnRes 缓存最有效：每个 micro-batch 的 block 总传输量不超过 8×(PP−1)，与 VP 无关；峰值约为 (PP−1)×8/(PP×VP) 取整，PP4×VP8 时为 1。
- 与 PP 大、VP 小相比，每个 rank 在飞的激活总量差不多，都在“全模型层数 × 1 个 micro-batch”左右。

**代价**
- 每卡参数量按 1/PP 增长：PP4 的静态显存是 PP8 的两倍。
- stage 越薄，每段计算越少，P2P 延迟和 kernel 启动开销占比越大；首段的 embedding 加视觉编码器、末段的 lm_head 加 loss 越显得不均衡。
- torch 的 PipelineStage 为每个 stage、每个 micro-batch 常驻分配接收缓冲，stage 越多缓冲越多：hidden 缓冲每 rank 约 2×M×VP 个单位，AttnRes 的 delta 缓冲再按传输量叠加。
- interleaved 1F1B 的 warmup 让 rank 0 在飞的 micro-batch 最多，各 PP rank 不均衡，需要 balance。
- 我们当前实现每个 stage 都复制一份 stack，VP 越大副本越多。

## 4. vp>1 时各 PP rank 的显存时间线

### 4.1 建模了什么

排程直接用 torch 的 `ScheduleInterleaved1F1B._calculate_single_rank_operations`，各 rank 按时间对齐，每个前向或反向占一个槽。block 显存按 pp_review4 代码的实际生命周期计算：

| 项 | 本 PR（按当前实现） | 依据 |
|---|---|---|
| stage 输入 stack | 每个 stage 前向时用 `torch.stack` 复制 N 个 block，作为输入留在 `fwd_cache` 里直到本 stage 反向 | `stage.py` `_assemble_stack` 和 `forward_one_chunk` |
| 产生 block 的 stage 的输出 stack | N+c 个 block。同一 stage 里若还有下一层，由 AC 保存到反向；否则只被 store 引用到释放 | 模型在 block 边界 `torch.cat`；store 保存的是它的 view |
| 发出的 payload | 新的 stack 副本，k 个 block，作为输出留在 `fwd_cache` 直到反向 | `_pack_outgoing_delta` |
| 接收缓冲 | torch 为每个 micro-batch 常驻分配：delta 接收 k_in 个、payload 梯度接收 k_out 个 | torch `PipelineStage._setup_forward_recv_info`、`_prepare_backward_infra` |
| deposit | 反向时每个 (micro-batch, block) 一个，从第一个读它的 stage 反向起，到带来它的 stage 反向止 | `PPRankLocalCache.deposit` 和 `collect` |

对照方案：
- **理论下界**：每个 rank 用到的 block 只存一份，从到达或产生起，到该 rank 上这个 micro-batch 最后一次反向止；另加 deposit；没有常驻接收缓冲，接收直接进 store。
- **混合方案（jinsooihm 的规则）**：相邻跳走 torch 常驻缓冲；直发分两种，一种同样按 micro-batch 常驻分配，另一种按需分配，即到达后分配、释放时回收。
- **本 PR、缓冲按需分配**：只用来单独比较拓扑，这需要改变 torch 的接收缓冲机制。
- **offload（#4765）**：store 里的 block 停到 host，只有当它是 GPU 上唯一持有者时才真正省显存。
- **balance（#4764）**：把源 rank 的一部分层激活挪到最轻的 rank，按总显存二分出均衡点，这是理想化模型。

总显存 = 静态 + torch 自带的 hidden 接收缓冲（任何模型都有）+ AttnRes block + a × 在飞的层数。

### 4.2 H100，PP8 × VP4，16 个 micro-batch

各 rank 基本情况：

| rank | 层数 | 静态 GiB | 在飞层激活峰值（层 × micro-batch） | torch hidden 接收缓冲（单位） |
|---:|---:|---:|---:|---:|
| 0 | 11 | 30.6 | 102 | 112 |
| 1 | 12 | 29.4 | 111 | 128 |
| 2 | 12 | 29.4 | 105 | 128 |
| 3 | 12 | 31.2 | 99 | 128 |
| 4 | 12 | 29.4 | 93 | 128 |
| 5 | 12 | 29.4 | 87 | 128 |
| 6 | 12 | 29.4 | 81 | 128 |
| 7 | 10 | 28.2 | 75 | 112 |

AttnRes block 显存峰值，常驻 + 动态，单位为 block：

| 方案 | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 本 PR | 160+301 | 224+199 | 224+193 | 176+163 | 192+341 | 256+212 | 256+204 | 176+171 |
| 本 PR + offload | 同上 | 同上 | 同上 | 同上 | 同上 | 同上 | 同上 | 同上 |
| 本 PR，缓冲按需 | 0+302 | 0+213 | 0+199 | 0+167 | 0+343 | 0+224 | 0+212 | 0+174 |
| 混合，直发常驻 | 304+277 | 224+199 | 112+134 | 176+163 | 288+310 | 256+212 | 128+150 | 176+171 |
| 混合，直发按需 | 112+284 | 176+203 | 112+134 | 64+176 | 128+313 | 192+222 | 128+150 | 48+183 |
| 理论下界 | 0+118 | 0+114 | 0+110 | 0+106 | 0+126 | 0+122 | 0+118 | 0+114 |
| 理论下界 + offload | 0+40 | 0+40 | 0+40 | 0+40 | 0+48 | 0+48 | 0+48 | 0+48 |

总显存峰值（GiB），a=1 即每层完整重算，最后一列为 balance 之后：

| 方案 | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | balance 后 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 本 PR | 104.4 | 101.9 | 100.5 | 93.1 | 111.9 | 104.2 | 102.6 | 86.6 | 最高 101.9 |
| 本 PR，缓冲按需 | 87.1 | 78.9 | 76.7 | 74.2 | 91.1 | 77.5 | 75.5 | 67.7 | 最高 81.9 |
| 混合，直发常驻 | 117.6 | 101.9 | 81.8 | 93.1 | 119.0 | 104.2 | 82.7 | 86.6 | 最高 108.9 |
| 混合，直发按需 | 97.3 | 97.0 | 81.8 | 82.2 | 101.9 | 98.2 | 82.7 | 73.9 | 最高 93.7 |
| 理论下界 | 66.8 | 68.1 | 67.0 | 67.6 | 67.4 | 66.3 | 65.2 | 61.1 | 最高 66.3 |
| 理论下界 + offload | 58.3 | 60.0 | 59.3 | 60.4 | 58.9 | 58.2 | 57.6 | 53.9 | 最高 58.5 |

a=17（FP8、不 offload）时，所有方案都在 185 到 296 GiB，远超 74.5 GiB。H100 必须走报告里的 CPU offload 加远程 offload。

### 4.3 GB300，PP4 × VP8，8 个 micro-batch

| rank | 层数 | 静态 GiB | 在飞层激活峰值 | torch hidden 接收缓冲 |
|---:|---:|---:|---:|---:|
| 0 | 23 | 58.1 | 98 | 120 |
| 1 | 24 | 56.9 | 99 | 128 |
| 2 | 24 | 56.9 | 93 | 128 |
| 3 | 22 | 57.4 | 87 | 120 |

| 方案 | block 峰值 r0 / r1 / r2 / r3（单位） | 总显存 a=1（GiB） | 总显存 a=17（GiB） | a=17 加 balance |
|---|---|---|---|---|
| 本 PR | 361 / 307 / 302 / 205 | 121.4 / 115.3 / 114.1 / 102.4 | 292.9 / 288.6 / 276.8 / 254.7 | 最高 278.7，超出 268 |
| 本 PR + offload | 同上 | 同上 | 同上 | 同上 |
| 本 PR，缓冲按需 | 298 / 185 / 178 / 142 | 114.5 / 102.0 / 100.5 / 95.5 | 286.0 / 275.2 / 263.3 / 247.8 | 最高 269.6 |
| 混合，直发按需 | 364 / 307 / 207 / 145 | 121.7 / 115.3 / 103.7 / 95.9 | 293.2 / 288.6 / 266.5 / 248.1 | 最高 276.6 |
| 理论下界 | 67 / 65 / 63 / 61 | 89.1 / 88.8 / 88.0 / 86.7 | 259.9 / 262.1 / 250.7 / 238.9 | 最高 253.6，能放下 |

### 4.4 时间线（H100，a=1，总显存，三个方案共用纵轴，满格约 112 GiB，横轴是一步里的时间槽）

```
本 PR
  rank 0: ▄▄▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅▄▄▄  104.4
  rank 1: ▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅  101.9
  rank 4: ▅▅▅▅▅▅▆▆▆▇▇▇██████████▇▇▇▇▇▇▇▇▇▇▇▇████████▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅▅  111.9
  rank 7: ▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄  86.6
混合，直发按需
  rank 0: ▄▄▄▄▅▅▅▅▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄  97.3
  rank 1: ▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅▅▅▄  97.0
  rank 4: ▄▄▄▄▅▅▅▅▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▅▅▅▄▄▄▄▄▄  101.9
  rank 7: ▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄▄▄▄▄▃▃▃▃▃▃▃  73.9
理论下界
  rank 0: ▃▃▃▃▃▄▄▄▄▄▄▄▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▃▃▃▃  66.8
  rank 1: ▃▃▃▃▄▄▄▄▄▄▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▃▃▃▃  68.1
  rank 4: ▃▃▃▃▃▄▄▄▄▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▅▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▃▃▃▃▃  67.4
  rank 7: ▃▃▃▃▃▃▃▃▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▃▃▃▃▃▃▃▃▃  61.1
```

全部 rank 以及 balance 之后的时间线见 `pp_memory_timeline_2026-09-24.out.txt`。

### 4.5 怎么读

- **形状**：各 rank 都是 warmup 爬升、稳态平台、cooldown 下降。rank 0 和 rank 4 最高，因为在 PP8×VP4、32 段、8 个 block 的切分下，所有 block 都产生在 stage 0、4、8 … 上，而这些 stage 只落在 rank 0 和 rank 4。产生 block 的 stage 既有输入 stack 又有输出 stack，deposit 也集中在这里。
- **差距的构成**：本 PR 与理论下界的差距约 35 到 45 GiB（H100）和 25 到 35 GiB（GB300），分成两部分。
  - 常驻接收缓冲：H100 上每 rank 160 到 256 个单位，约 17 到 28 GiB，大小正比于 M × 传输量。
  - stack 与 payload 副本：H100 上每 rank 动态 163 到 341 个单位，下界只有 106 到 126。
- **拓扑的作用很小**：混合方案只是把一部分流量挪出 torch 的常驻缓冲。两边缓冲都按需分配时，本 PR 反而更低：H100 最重的 rank 91.1 对 101.9 GiB。
- **offload（#4765）当前省不下显存**：收到的 delta 是 torch 常驻缓冲的别名；在这两种切分下，产生 block 的层后面同一 stage 里总有下一层，输出 stack 由 AC 留到反向。要让 offload 生效，store 必须自己拥有 block 的存储，不能是 torch 缓冲或 autograd 张量的别名。
- **balance（#4764）**：只能搬 autograd 保存的层激活。a 较小时可搬的量有限，H100 a=1 时最重的 rank 只从 111.9 降到 101.9 GiB；a 较大时效果明显，a=17 时从 296 降到 271 GiB。它搬不动 torch 的接收缓冲，也搬不动 `fwd_cache` 里的 stack 和 payload 副本。

## 5. 对 PR 和回复的含义

- **本 PR 离报告所说的“理论下界”还有距离。** 要达到下界，需要：stage 输入以 block 列表而不是新拼的 stack 形式传入；payload 发送 view 而不是副本；接收直接进入 store，而不是经过 torch 按 micro-batch 常驻的缓冲。最后一项还依赖 torch pipelining 能复用或按需分配接收缓冲。这些都是后续工作，不在 4312 本身。
- **#4765 需要先改设计再谈收益。** 在当前切分下它不省显存，给 jinsooihm 的回复里暂不引用它。
- **给 jinsooihm 的回复**：显存那一段只引用 #4764。
- **说明**：这些数字依赖 a 的取值和静态显存的粗估，只用于比较各方案之间的相对差距；a 值需要在 H100 上实测一层的保存量才能定。
