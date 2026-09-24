# PP rank cache 显存优化：8 卡实测，以及 2.8T 切分估算的重审（2026-09-24）

分支：fork 上的 `pp_review_optimize`，基于 pp_review4 `901ef34de`，两个提交 `14cba2237`、`202b6a974`。4312 的 PR 分支和 pp_review4 都没有动。

测量工具在 `kit_pp_optimize_2026-09-24/`：`probe_ppmem.py`（探针 flavor，只在本地用）、`run_ppmem.sh`、`tab_ppmem.py`、`stash_probe.py`、`base_own_sends_probe.patch`，每次运行的逐 rank 记录在 `results/`。显存模型是 `pp_memory_model_v2_2026-09-24.py`（输出 `.out.txt`），2.8T 可行域计算是 `k3_2p8t_sizing_2026-09-24.py`（输出 `.out.txt`）。

## 0. 结论

- **实测**（8 × RTX 5060 16 GB，探针模型见 §2.1，pp8 × vp2，16 个 micro-batch）
  - 在 seq 3584 下对比（基线能跑的最长 seq，4096 时 rank 6 OOM），看全部 8 个 rank：各 rank 峰值 allocated 降 2.65 到 5.59 GiB，均值从 11.36 降到 6.98 GiB（−39%），最重的 rank 从 13.38 降到 8.04 GiB（−40%），rank 间极差从 4.95 缩到 2.25 GiB。
  - 这个探针用 FullAC，每个 rank 能被 balance 草案（#4764）搬走的只有 0.12 到 0.75 GiB，叠加 balance 后两棵树的峰值都不变（§8）；所以这里的收益就看全部 rank 的逐 rank 对比和均值。
  - 步与步之间常驻的显存从 3.40–6.02 降到 0.99–1.76 GiB，torch 的常驻接收缓冲从 2.41–4.16 GiB 降到 0。
  - 100 步的 loss 和 grad norm 与基线全部逐位一致，tps 中位数相同（都是 570），见 §2.4。
  - 显存相同时，优化后每个 micro-batch 能放 6144 个 token，基线只能放 3584（1.71×）：优化后 seq 6144 峰值 13.32 GiB，基线 seq 3584 峰值 13.38 GiB。
- **性质**：传输机制没变。路由表、每跳发哪些 block、deposit 和梯度都没变。改的是 stage 适配层里缓冲归谁、活多久，外加绕开 torch runtime 的两个行为。节省按来源分三块（§3）：
  - torch 的常驻接收缓冲改为收进 store 或按需分配，这块最大；
  - 深拷贝改成 view；
  - torch 在 step 末尾才 wait send，被发出去的张量因此一直占着显存到 step 末尾。
- **离报告说的下界还差两处**，都不在 stage 层：
  - 反向 send 同样被钉到 step 末。前向 send 可以在该 micro-batch 反向开始时安全 wait，反向 send 在本 rank 上没有这样的安全点，需要改 torch 的 runtime。
  - 模型在 stage 中间开新 block 时，`torch.cat` 会拼出一条新 stack，被本 stage 后面的层留到反向。这要把模型里的 block 载体从 stack 张量改成 block 列表。
  - 模型已用实测校准。在 2.8T 的 H100 切分上，这两处合计约 20 到 28 GiB/rank（§4）。
- **2.8T 估算**：之前推荐的 PP8 × VP4 不准，给得过于确定（§5）。
  - 报告没有给出集群规模、global batch，也没有给任何并行度。
  - 之前漏了 MTP 层、MoonEP 的缓冲、torch 钉住 send 张量的开销，以及随 M × VP 增长的 hidden 接收缓冲；激活只取了 a=1 和 a=17 两档，是区间不是估计。
  - 按报告的激活配方（每层每个 8K micro-batch 约 1.63 GiB），H100 上 TP1 的所有切分都要把 90% 以上的激活移出 HBM。所以 H100 上 PP 由静态显存（要求 PP × EP ≥ 256）、EP 流量和气泡决定。可行的有 PP8 × VP3–4 配 EP64、PP16 × VP2 配 EP32–64、PP32 配 EP16–64。K2 用的是 PP16 × EP16，这是 K2 一致的方向，但公开信息定不下具体取值。
  - GB300 上 PP4 × VP3–8 配 EP32–64 基本不用 offload，PP2 需要 offload 约 20%。
  - 用校准过的模型外推：pp_review4 现在的实现在 H100 候选切分上，仅块残差加 hidden 一项就要 72 到 146 GiB/rank，放不下；优化分支为 30 到 55 GiB；下界为 18 到 27 GiB。

## 1. 改了什么

| 项 | pp_review4 | pp_review_optimize | 性质 |
|---|---|---|---|
| delta 接收 | torch 为每个 micro-batch 常驻一块接收缓冲，跨步保留 | 直接收进 rank store 中该 micro-batch 那块 [N, T, D] 缓冲的对应行 | 局部：重写 torch 的 `_setup_forward_recv_info` 和 `get_fwd_recv_ops` |
| stage 输入 stack | `torch.stack` 拼一份新副本，留到反向 | store 行 [0, N) 的转置 view | 局部：深拷贝改为 view |
| payload | `torch.stack` 新副本 | store 行的 view；梯度经一个 autograd Function 回到模型输出，所以模型输出的 stack 在前向结束后就能释放 | 局部：深拷贝改为 view |
| 其余接收缓冲（hidden，以及两个梯度） | torch 每个 micro-batch 常驻 | post 接收时分配，读完即丢 | 局部，对任何 PP 模型都适用 |
| 前向 send | torch runtime 在 step 末才 wait，NCCL Work 在 wait 前一直钉住被发送的张量（view 会钉住整块底层存储） | action-list runtime 下由 stage 自己发，在该 micro-batch 反向开始时 wait，此时接收方一定已经用过；单 stage schedule 会把 send 和 recv 合成一批，这时仍交给 torch 发，payload 改发只含这几个 block 的副本 | 绕开 torch runtime 的行为 |
| 不变 | layout 表、每跳发哪些 block、deposit 与 collect、模型代码 | 同左 | |

- torch Work 钉住张量这一点是实测的（`stash_probe.py`，2 卡）：发出 256 MiB 后，对端收完 3 秒，发送方仍占 256 MiB，wait 之后才归零。
- 验证方式：CPU 单测 40 项全过。其中 `test_kimi_k3_pp_block_grads` 在 4 进程 gloo 上跑真实的 Interleaved1F1B，cache 开和关两种情况下，每个 block 的梯度都与单卡逐位一致，eval 路径跑完后 store 为空。

## 2. 8 卡实测

### 2.1 设置

- 探针模型：debug flavor 放大到 dim 2048，32 层，block 4（共 8 个 block，与发布模型的 block 数相同），MLA 放在每组 4 层的第 4 层。MoE 很小（8 个专家、top-2），这样显存主要由 block 占。
- 训练设置：FullAC，AdamW，744M 参数。pp8 × vp2，Interleaved1F1B，16 个 micro-batch，每个 micro-batch 3584 个 token。seed 42，deterministic。
- 缓存：每个 cell 用同一份暖缓存（`cache_W`）的独立副本。
- 读数方法：titan 每步 log 后会 reset 峰值统计，所以探针在每次 reset 之前记录一次。下表取第 3 步。之前那次 atexit 读到的 3.4–6.0 GiB 是第 3 步之后的残余，不是训练峰值，已作废。

### 2.2 同一 seq 3584，第 3 步每 rank 峰值 allocated（GiB）

| rank | 基线 | 基线 + 自持 send | V1 | V2 | V3 | V4（提交） | 基线 − V4 | 模型预测 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.44 | 7.92 | 6.99 | 7.21 | 6.32 | 5.79 | 2.65 | 2.72 |
| 1 | 10.04 | 9.17 | 8.97 | 8.36 | 6.93 | 6.37 | 3.67 | 3.69 |
| 2 | 10.84 | 10.07 | 8.32 | 8.76 | 7.35 | 6.62 | 4.21 | 4.01 |
| 3 | 11.36 | 10.37 | 9.71 | 9.20 | 7.43 | 6.81 | 4.54 | 4.47 |
| 4 | 12.11 | 11.18 | 9.02 | 9.57 | 7.94 | 7.20 | 4.90 | 4.87 |
| 5 | 12.64 | 11.34 | 10.63 | 10.03 | 7.96 | 7.22 | 5.42 | 5.47 |
| 6 | 13.38 | 12.12 | 9.72 | 10.37 | 8.53 | 7.79 | 5.59 | 5.52 |
| 7 | 12.05 | 11.50 | 9.85 | 9.31 | 8.32 | 8.04 | 4.01 | 3.54 |
| 最大 | 13.38 | 12.12 | 10.63 | 10.37 | 8.53 | 8.04 | | |
| 均值（8 个 rank） | 11.36 | 10.46 | 9.15 | 9.10 | 7.60 | 6.98 | 4.38 | |
| 极差（最大 − 最小） | 4.95 | 4.20 | 3.64 | 3.16 | 2.20 | 2.25 | | |
| 最大 reserved | 14.26 | 12.98 | 11.55 | 11.17 | 9.37 | 8.80 | | |

各列的含义：
- 基线 + 自持 send：pp_review4 只加上“前向 send 自己发、在反向时 wait”（`base_own_sends_probe.patch`）。
- V1（`14cba2237`）：store、stack view、payload view（view 的是模型输出的 stack）。
- V2：payload 改为 view store，send 时发副本。
- V3：payload view store，send 在反向时 wait。
- V4（`202b6a974`）：V3 再加上所有接收缓冲按需分配。

几点观察：
- 所有列 3 步的 loss（8.18973 / 6.70710 / 4.82671）和 grad norm（19.7500 / 15.8750 / 17.3750）都逐位一致，tps 在 563 到 570 之间。
- V1 在偶数 rank 上比 V2 好，奇数 rank 上反过来。原因是 send 把 view 的底层存储钉到 step 末：偶数 rank 的 stage 不产生 block，它的 payload view 钉住的是 store 缓冲，本来也要留很久；奇数 rank 的 stage 在最后一层产生 block，payload view 钉住的是模型新拼出的整条 stack。这说明“payload 发 view”只有在 send 能及时 wait 时才划算，V3 和 V4 就是这样做的。

### 2.3 容量

| 版本 | seq/micro-batch | 最大 allocated | 最大 reserved | 结果 |
|---|---:|---:|---:|---|
| 基线 | 3584 | 13.38 | 14.26 | 通过 |
| 基线 | 4096 | 13.92（OOM 时） | 14.08 | rank 6 OOM，申请 288 MiB（aggregation 的 FP32 副本）失败 |
| V4 | 4096 | 9.09 | 9.87 | 通过 |
| V4 | 5120 | 11.21 | 12.17 | 通过 |
| V4 | 6144 | 13.32 | 14.29 | 通过，峰值与基线在 3584 时相同 |

V4 之后最重的 rank 从 rank 6 变成了 rank 7（它放着 lm_head 和 loss）。

### 2.4 100 步逐位对比

基线与 V4，seq 3584，用同一暖缓存谱系的两份副本。

| 步 | 基线 loss | V4 loss | 基线 grad norm | V4 grad norm |
|---:|---:|---:|---:|---:|
| 1 | 8.18973 | 8.18973 | 19.7500 | 19.7500 |
| 10 | 3.60542 | 3.60542 | 7.1250 | 7.1250 |
| 20 | 1.37265 | 1.37265 | 2.7500 | 2.7500 |

- 100 步的 loss 和 grad norm 全部逐位一致，tps 中位数都是 570。
- 各 rank 在 100 步里的最大峰值就是 §2.2 第 3 步的读数（基线 13.38 GiB，V4 8.04 GiB）。
- 这是 debug 数据集，会被模型记住：参考轨迹从第 25 步起在 0.61 到 0.94 之间来回摆动，第 100 步降到 0.013。所以表只列到第 20 步。
- 记录在 `kit_pp_optimize_2026-09-24/results/base_s3584_100`、`v4_s3584_100`。

## 3. 这是机制调整，还是拷贝和复用上的优化

判断：传输机制本身没变，改的是缓冲的所有权和生命周期，外加绕开 torch runtime。下面按实测分解最重的 rank（13.38 → 8.04 GiB，共省 5.35 GiB）：

| 步骤 | 省下（最大 rank） | 各 rank | 属于 |
|---|---:|---|---|
| 基线 → 基线 + 自持 send | 1.26 | 0.52–1.30 | torch runtime：send 在 step 末才 wait，Work 钉住张量；对任何 PP 模型都成立 |
| 基线 + 自持 send → V3 | 3.60 | 1.60–3.60 | stage 局部：delta 收进 store，stack 和 payload 都是 view，梯度接收按需分配 |
| V3 → V4 | 0.49 | 0.29–0.74 | stage 局部：hidden 的接收缓冲也按需分配；对任何 PP 模型都成立 |

代码量：`stage.py` 净增约 100 行，`cache.py` 重写（约 70 行），模型和 layout 都没改。

## 4. 离报告的下界还差什么

模型把 torch 实测到的两种行为（send 张量钉到 step 末，接收缓冲按 micro-batch 常驻）都放进去了。它对探针的预测与实测对照见 §2.2 最后两列：6 个 rank 误差不超过 0.07 GiB，rank 2 差 0.21 GiB，rank 7 差 0.47 GiB。常驻接收缓冲那一项与实测逐位相同（176–304 个单位，即 2.406–4.156 GiB）。

下面是 V4 到下界之间的各项差距（最重 rank，GiB）。每项是单独修掉它时的降幅，峰值会移动，所以各项不能相加；四项一起修掉恰好等于下界：

| 切分 | V4 | 反向 send 钉到 step 末 | store 按 rank 整块分配 | hidden 输出留到反向 | stage 内开 block 的 cat 副本 | 下界 |
|---|---:|---:|---:|---:|---:|---:|
| 探针 pp8 × vp2 | 3.2 | 0.9 | 0.0 | 0.1 | 0.0 | 1.8 |
| 2.8T H100 PP8 × VP4，M=16 | 43.1 | 4.8 | 2.0 | 3.4 | 9.2 | 20.9 |
| 2.8T H100 PP16 × VP2，M=32 | 54.9 | 9.0 | 0.0 | 2.5 | 3.6 | 26.6 |
| 2.8T GB300 PP4 × VP4，M=16 | 28.2 | 9.2 | 0.5 | 1.4 | 0.5 | 10.4 |
| 2.8T GB300 PP2 × VP8，M=16 | 33.8 | 18.3 | 0.0 | 0.9 | 0.0 | 8.0 |

- **反向 send**：属于 torch runtime。前向 send 有安全点：本 stage 对这个 micro-batch 的反向开始时，接收方一定已经用过数据。反向 send 在本 rank 上没有这样的点。要由 schedule 根据全局动作顺序，找出第一个能证明对端已收到的动作，在那里 wait。可以向 torch 提 issue，附上本探针的数据。
- **cat 副本和按 block 释放**：属于模型。现在模型在每个 block 的第一层用 `torch.cat` 追加新 block，每次都复制一遍整条 stack；store 缓冲也必须连续，只能按本 rank 最大的 stage 整块分配。把 block 载体改成 block 列表后：
  - 追加不再拷贝；
  - store 可以按 block 分配和释放；
  - aggregation 里的 `torch.stack` 与现在的 `cat` 结果相同，数值不变。
  - 代价是改模型的 forward，以及 TP、SP、CP 对这个载体的切分声明。
- **hidden 输出留到反向**：这是 torch PP 保存 stage 输出的常规做法。要更早释放，需要 Megatron 式的“发送后清空 data 并自定义 backward”。

## 5. 2.8T 切分估算的重审

### 5.1 之前哪里不准

- **报告没有给出并行度**。§5.2 只列了 PP 加 VP、EP、ZeRO-1 DP、Pipeline ZeRO-2（梯度放 CPU，GPU 上保留双缓冲）、CP，没有 TP，也没有集群规模和 global batch。之前的 N=8192、M=16 都是我的假设。
- **漏项**：
  - MTP 层（报告 3.3 节：预训练带 1 层 MTP），这是一个完整的 MoE 层，在最后一个 rank 上，约 30.2B 参数。
  - MoonEP：每个 rank 预留 E/EP 个冗余专家槽（权重加梯度暂存），加上固定的 S × K 分发缓冲。
  - torch 把被发送的张量钉到 step 末（本次实测发现）。
  - hidden 接收缓冲随 M × VP 常驻。
- **激活**：之前只给了 a=1 和 a=17 两档。按报告的配方（激活 FP8 存储、逐元素算子重算、dispatch 重算、AttnRes 用 checkpoint）逐项估算，每层每个 8K micro-batch 约 1.63 GiB（约 15 个单位）。
- **EP 通信**：之前按 FP8 计算估每层前向 24 ms。按 BF16 GEMM、450 TFLOPS 有效算力估约 42 ms；EP64 跨 8 个节点时，前向 all-to-all 走 IB 约 25 ms，可以重叠。

### 5.2 rank 怎么分

- N = PP × TP × CP × DP。
- EP 不单独占 rank，它从 DP × CP × TP 里借（titan 的 `ParallelDims`：efsdp = dp_shard · cp · tp / ep）。
- 所以“其他维度占 rank”只对 TP 和 CP 成立。N 固定时，它们压缩的是 DP；global batch 也固定时，每条流水线的 micro-batch 数 M 随之变大，气泡变小。它们不会把 PP 压小。
- PP 由每卡显存决定：专家参数按 PP × EP 分片，非专家参数按 PP × TP 分片。

### 5.3 TP

- MLA 和 KDA 都是 96 个 head，所以 TP 必须整除 96。
- H100 上 TP 要在一个 8 卡 NVLink 节点内，只能取 1、2、3、4、6、8。
- GB300 NVL72 上，只看 head 数，TP 最高可到 48。但 EP 也要占用同一个 NVLink 域；titan 要求 MoE 模型 EP ≥ TP（#4794），而且 EP 会借用 TP 的 rank。
- 报告没有用 TP。H100 上开 TP8 能把激活降到 1/8，PP4–8 配 EP64 时不 offload 也放得下；代价是每层前向 4 次 SP 集合通信。报告选的是 offload，没有选 TP。

### 5.4 可行域（TP1，即报告的配方；N=4096 只影响优化器分片）

**H100 80 GB**
- 静态显存：
  - PP8 配 EP64：39–40 GiB
  - PP16 配 EP32：35 GiB；配 EP64：27 GiB
  - PP32 配 EP16：37 GiB
  - PP4 配 EP64：62–64 GiB
- 在飞的“层 × micro-batch”数由 schedule 决定，是 99 到 183，几乎与 PP 无关。对应激活 160–300 GiB，所有切分都要移出 90% 到 100%。这与报告同时用 CPU offload 和远程 offload 相符，也暗示训练用的是 80 GB 级别的卡（这是推断，报告没写）。
- 所以 H100 上 PP 不由激活决定，而由三件事决定：
  - 静态显存，要求 PP × EP ≥ 256；
  - EP 走 IB 的流量：前向每层 EP16 约 14 ms，EP64 约 25 ms，计算约 42 ms；
  - 气泡和 AttnRes 加 hidden 的显存（§5.5）。
- 候选：PP8 × VP3–4 配 EP64；PP16 × VP2 配 EP32–64；PP32 × VP1 配 EP16–64。
- K2 技术报告（arXiv 2507.20534，已核对原文）的配置是：PP16（带虚拟 stage）× EP16 × ZeRO-1，H800，节点数为 32 的倍数，其余激活全部 offload 到 CPU，global batch 67M token。K3 的参数量是它的 2.7 倍，同样 PP16 时，EP 大概率要到 32 或更高。

**GB300 288 GB**
- PP4 × VP3–8 配 EP32–64（在 NVL72 内）：offload 0 到 10%。
- PP2 × VP4–8 配 EP64：offload 17% 到 20%，Grace C2C 带宽足够。
- 显存上不需要 PP8。

完整表格见 `k3_2p8t_sizing_2026-09-24.out.txt`。其中的假设：
- 双梯度缓冲按桶级 2 GiB 算（报告没给大小）；
- CUDA、NCCL 等预留 4 GiB；
- 有效算力：H100 450 TFLOPS，GB300 1100 TFLOPS。

### 5.5 这些切分上块残差加 hidden 的显存（校准后的模型，全部 rank，GiB）

| 切分 | pp_review4：最小 / 均值 / 最大 | V4 | 下界 |
|---|---|---|---|
| H100 PP8 × VP4，M=16 | 71.8 / 84.0 / 94.6 | 30.4 / 34.8 / 43.1 | 17.9 / 19.2 / 20.9 |
| H100 PP16 × VP2，M=32 | 90.8 / 117.9 / 146.1 | 37.2 / 45.8 / 54.9 | 20.5 / 23.5 / 26.6 |
| GB300 PP4 × VP4，M=16 | 67.8 / 72.6 / 82.4 | 25.2 / 26.6 / 28.2 | 9.0 / 9.6 / 10.4 |
| GB300 PP2 × VP8，M=16 | 86.0 / 89.9 / 93.8 | 26.2 / 30.0 / 33.8 | 8.0 / 8.0 / 8.0 |

- **balance 碰不到这一项**：块残差加 hidden 全是 balance 搬不动的部分（接收缓冲、副本、被钉住的 send、store）。叠加 balance 后，每个 rank 的峰值仍不低于它自己的这一项加静态显存。V4 把每个 rank 的这一项都降了，不只是最重的那个。
- **包含通用开销**：这些数字含 torch 的 hidden 缓冲和被钉住的 hidden send，任何 PP 模型都有这部分，不全是 AttnRes 的。每个 rank 的明细见 `pp_memory_model_v2_2026-09-24.out.txt`。
- **修正旧文档**：这里修正了 `K3_2P8T_PP_VP_MEMORY_2026-09-24.md` §4 里“本 PR”那一行。旧模型没有算被钉住的 send，低估了约 200 个单位。
- **结论**：pp_review4 现在的实现在 torch runtime 下跑不了 H100 上的 2.8T，最轻的 rank 也要 72 GiB；V4 为 30 到 55 GiB。要达到下界，还需要 §4 里 torch 和模型两侧的改动。

### 5.6 可靠程度

- **最可靠**：块残差和 hidden 这一项。它只依赖 layout 表和 torch 的行为，两者都在探针上实测校准过，每 rank 误差不超过 0.5 GiB。
- **量级可信，数值会变**：静态显存和激活。它们依赖梯度缓冲大小、MoonEP 缓冲、激活配方、N 和 M。
- **公开信息定不下**：具体的 PP、VP 和 EP 取值。

## 6. 下一步（待定）

- 4 × H100 上按 §9 测量：全部 rank，先不叠加、再叠加 balance。
- 模型改为 block 列表：在 pp_review_optimize 上做，并实测 stage 中间开 block 那份 `cat` 副本。
- torch issue：send 在 step 末才 wait，接收缓冲按 micro-batch 常驻。附 `stash_probe.py` 和 §2.2 的数据。
- 按规则，这里的数都是 5060 上的冒烟级数据；要写进 PR 的数，需要在 H100 上重跑。

## 7. 限制

1. **send 只在 action-list runtime 下由 stage 持有。** 包括 Interleaved1F1B、LoopedBFS、InterleavedZeroBubble。单 stage schedule（1F1B、GPipe）会把 send 和 recv 合成一批来避免死锁，这时仍交给 torch 发，payload 改发只含这几个 block 的副本，副本同样被钉到 step 末，所以 vp=1 时收益小。
2. **反向 send（发给上一段的梯度）仍被 torch 钉到 step 末。** 发送方在本步里没有能证明对端已收到的动作，要改 torch 的 schedule。按 §4 的模型：探针上是 0.9 GiB，2.8T 上 4.8 到 18.3 GiB。
3. **stage 中间开 block 时模型 `cat` 出的整条 stack 还在。** 本 stage 后面的层用 checkpoint 保存它，一直留到反向。发布模型的切分里 8 次开 block 全在 stage 中间（pp8 × vp4，93 层，block 12），这一项在 2.8T H100 PP8 × VP4 上约 9.2 GiB/rank。要改成 block 列表载体，涉及模型 forward 以及 TP/SP/CP 对它的切分声明。
4. **store 是每个 micro-batch 一块连续缓冲。** 按本 rank 最大的 stage 分配，从该 rank 第一次前向留到最后一次反向，不能按 block 单独释放。这也使 offload 没法逐 block 停放（§8）。
5. **多用了一批 torch 私有接口。** `_setup_forward_recv_info`、`_setup_backward_recv_info`、`_retrieve_recv_grads`、`_make_tensor_from_meta`、`_batch_p2p`、`_PipelineScheduleRuntime`。torch 升级时可能要跟着改；上游 review 也可能要求把“按需接收缓冲”和“提前 wait send”放到 torch 里做。
6. **store 的行通过 `.data` 读写，绕过了 autograd 的 version counter。** 这样写入别的行时，不会让已保存的 stack view 报原地写错误。代价是 autograd 不再替我们检查。正确性依赖 layout 的不变式：每个 micro-batch 每行只写一次，而且写在所有覆盖它的 stack view 创建之前。这由 4 进程 gloo 的逐位梯度测试覆盖。
7. **stage 输入是转置后的非连续 view。** 模型只把它送进 `torch.cat`，所以数值逐位不变。如果以后模型直接对 stack 做 reduction，或者打开 compile，布局可能改变 kernel 的选择。探针跑的是 eager。
8. **覆盖面有限。** 只在 PP-only（dp1、tp1）、FullAC、5060 上测过。TP/SP、EP、FSDP>1、CP 的组合，以及 4312 的 CI 格子（pp2 × vp2 的 17 层模型、B200 上的 fsdp2 × tp2 × ep2 × pp2 × vpp4）都还没跑。
9. **其余两点。** eval（只有前向）不持有 send，仍在 step 末 wait；stage 放置仍只支持 loop-style（与 pp_review4 相同）。

## 8. 与 pp balance（#4764）、pp offload（#4765）的关系

两个草案都叠在 `3902077ca` 上，代码已核对。

**balance（`k3_pp_balance`）**
- 机制：在每个 stage 的前向外面套 `saved_tensors_hooks`，把 autograd 保存的、CUDA 上连续的、至少 1 MiB 的张量写进一个目标 rank 预先分配的池（Mooncake）。池整步常驻在目标 rank 上，各源 rank 平分；每个源 rank 另有 256 MiB 的 staging 缓冲。
- 它能搬的，只有“autograd 保存、且别处不再引用”的张量：FullAC 下是 stage 内非首层的层输入；在 V4 里还包括 stage 中间开 block 留下的那条 stack。
- 它搬不动 torch 的接收缓冲、fwd_cache 里的副本、被钉住的 send 张量和 store，而这些正是 V4 去掉的。所以两者是**互补可叠加**的：V4 降低每个 rank 搬不动的底，balance 再把能搬的部分在 rank 间摊平。
- V4 的 stage 输入 stack 是 store 的非连续 view，草案会自动跳过它，不会白传。基线的 stack 副本虽然连续，但同时被 fwd_cache 引用，搬走也释放不了显存，只会白占带宽。
- V4 之后最重的 rank 变了（5060 上 r6 → r7，H100 探针上 r2 → r3），balance 的目标 rank、源 rank 和池大小都要重新定，池也会小很多。
- 代码上：草案在 `forward_one_chunk` 外面加了一个 forward context，和 V4 对这个函数的改写能直接合并；需要 rebase 到 pp_review_optimize 上。

叠加 balance 后的估算（全部 rank，GiB；`kit_pp_optimize_2026-09-24/balance_overlay.py`，输出见 `balance_overlay.out.txt`）：

| 配置 | 树 | 各 rank 不叠加 | 最大 / 均值 | 叠加 balance 后各 rank | 最大 |
|---|---|---|---:|---|---:|
| 5060 探针，FullAC | 基线（实测） | 8.44 … 13.38（8 个 rank） | 13.38 / 11.36 | 能搬的只有 0.12–0.75，不变 | 13.38 |
| 同上 | V4（实测） | 5.79 … 8.04 | 8.04 / 6.98 | 不变 | 8.04 |
| H100 探针（§9），FullAC，dim 4096 | 基线 | 62.5, 60.3, 70.5, 66.9 | 70.5 / 65.1 | 66.5, 60.3, 66.7, 66.9 | 66.9 |
| 同上 | V4 | 36.7, 35.1, 39.6, 43.9 | 43.9 / 38.8 | 38.2, 35.1, 39.6, 42.7 | 42.7 |
| 同上，每层多存到 4 个单位（类似 selective AC） | 基线 | 69.6, 66.6, 76.1, 71.6 | 76.1 / 71.0 | 69.6, 71.6, 71.4, 71.6 | 71.6 |
| 同上 | V4 | 43.8, 41.4, 45.2, 48.7 | 48.7 / 44.8 | 43.8, 45.2, 45.2, 45.2 | 45.2 |

- 能搬的越多，balance 越能把各 rank 压到接近均值，这时 V4 的收益接近各 rank 的平均节省，而不是最重那个 rank 的节省。
- 表中“每层 4 个单位”是假设值，要以 H100 上的实测为准。

**offload（`k3_pp_offload`）**
- 机制：在旧 store（按 block 存的 dict）的 `put` 里把提交的 block 拷到 pinned host，`blocks()` 读的时候再拷回 GPU。
- 在 pp_review4 上省不下显存：它停放的 block 是 torch 接收缓冲或 checkpoint 所持 stack 的别名。
- 在 V4 上它和重写后的 store 直接冲突，而且照搬也省不下：store 是一块连续缓冲，不能逐行释放；stage 前面那些行又被该 stage 的 checkpoint 当作输入保存，要到反向才释放。
- 要让 offload 在 V4 之上真正省显存，需要三件事：
  - block 列表载体，让 store 能逐 block 持有；
  - checkpoint 重算时从 store 重新取 stack，而不是一直引用它，这样 store 才能在两次使用之间把 block 停到 host；
  - 在每次使用前异步预取回来。
- 做到之后，显存可以低于 §4 的下界，因为下界假设 block 从到达一直驻留到最后一次反向。

## 9. 4 × H100 的度量方案与探针尺寸（看全部 rank，叠加 balance）

### 9.1 探针尺寸（`h100_probe_sizing.py`，输出见 `h100_probe_sizing.out.txt` 和 `balance_sizing.out.txt`）

- **结构**：pp4 × vp4，用 `--parallelism.pipeline-parallel-layers-per-stage 3` 切出 16 段；46 层，block 6，共 8 个 block。
  - 这样每次开 block 都在 stage 中间，与发布模型的切分一致；32 层、block 4 的版本里提交几乎都落在 stage 的最后一层。
  - 16 个注意力 head（KDA 与 MLA 的内部维度 2048），seq 8192（K3 预训练的 micro-batch），M=16，FullAC，AdamW。
- **标定**：静态显存按每个参数 16 字节算，参数在 meta 上实数。反向重算的瞬时项用 5060 的 V4 实测拟合（(−201 + 48 × (N + 1)) 个单位，最后一个 rank 加 62），拿来反推 5060 基线，每个 rank 误差不超过 0.5 GiB。
- **两种尺寸**：
  - **不叠加 balance：dim 4096**（20.8 亿参数，每 rank 静态 7.1–8.4 GiB）。基线各 rank 62.5 / 60.3 / 70.5 / 66.9 GiB，V4 为 36.7 / 35.1 / 39.6 / 43.9 GiB。同样峰值下 V4 每个 micro-batch 能放 14336 个 token（1.75×）。
  - **叠加 balance、让所有 rank 都接近 72 GiB：dim 4352**（22.1 亿参数，每 rank 静态 7.5–9.0 GiB）。基线叠加 balance 后各 rank 70.8 / 64.1 / 70.9 / 71.1（目标 r0，源 r2，池 4.25 GiB）；V4 叠加 balance 后 40.8 / 37.2 / 42.0 / 45.3（目标 r0，源 r3，池 1.75 GiB）。V4 叠加 balance 能放 13824 个 token（1.69×）。
- **5060 上已冒烟**：4 卡，dim 1024，2 步。切分与估算一致，rank 3 上是 stage 3、7、11、15，各 3 层。

### 9.2 怎么度量收益

- **全部 rank 的峰值 allocated**：每步在 titan 重置前记录，同时记 reserved、步间常驻、接收缓冲字节。收益同时报三个数：逐 rank 的差值、均值的差值、最大值的差值；叠加 balance 后另报一遍。显存的噪声底是 0：5060 上同配置重复运行，allocated 和 reserved 都逐字节一致；H100 上重跑一次 V4 确认。
- **容量**：每棵树在同一配置下能跑的最长 micro-batch（每次 3 步，按 512 递增到 OOM），不叠加和叠加 balance 各测一遍。
- **吞吐**：同配置下的 tps 用来检查没有退步；各自容量上限处的 tps 和 MFU 用来体现收益。可选：V4 把省下的显存用来换 selective AC，测 tps。
- **数值**：基线与 V4 各跑 100 步，同一暖缓存谱系的两份副本，loss 和 grad norm 逐位对比。
- **分解**：基线、基线加自持 send（`base_own_sends_probe.patch`）、V4 三格，与 5060 的分解对照。
- **模型验证**：各 rank 的预测值与实测值对照，作为第二个标定点（这次包含 stage 中间开 block 那一项）。

### 9.3 格子与顺序

| 顺序 | 格子 | 步数 | 用途 |
|---:|---|---:|---|
| 1 | 基线，dim 4096 | 3 | 确认撑到约 70 GiB，必要时按 ±256 调整 dim |
| 2 | V4，同配置 | 3 | 主结果 |
| 3 | V4，同配置再跑一次 | 3 | 噪声底 |
| 4 | 基线 + 自持 send | 3 | 分解 |
| 5 | V4 的 seq 依次加 512，直到 OOM；基线 seq 加 512 确认 OOM | 各 3 | 容量 |
| 6 | 基线与 V4 各 100 步 | 100 | 数值逐位 |
| 7 | 基线 + balance、V4 + balance，dim 4352 | 3 | 叠加 balance 的全部 rank（前提：#4764 已 rebase 到两棵树上，H100 机器装了 mooncake） |
| 8 | 可选：selective AC | 3 | 能搬的更多时 balance 的效果，以及用省下的显存换吞吐 |

- 脚本：`kit_pp_optimize_2026-09-24/run_ppmem_h100.sh`，参数为名字、树、缓存、seq、步数。可用 `PPMEM_DIM`、`PPMEM_LAYERS`、`PPMEM_BLOCK` 覆盖尺寸，用 `TORCHRUN` 指定环境。
- 按估算每步约 1.3 秒，全部格子不到一小时。
