# offload 与 balance 草案重写：统一激活存储（2026-09-25）

分支（fork）。2026-09-25 按用户的话同步了两个 draft PR：`k3_pp_offload` 从 `c73e17c03` 改为 `49117a146`，`k3_pp_balance` 从 `005cf4aee` 改为 `46692171b`；旧 head 存为 `backup/k3_pp_offload_pre_20260925`、`backup/k3_pp_balance_pre_20260925`。新 body 见 `PR_BODY_PP_OFFLOAD.md`、`PR_BODY_PP_BALANCE.md`，等用户粘贴。

| 分支 | head | 内容 |
|---|---|---|
| `pp_review_optimize` | `81b30fd88` | V4，再加输入梯度 send 提前释放 |
| `pp_offload_review1` | `49117a146` | 统一激活存储、host 后端、PP 计划 |
| `pp_balance_review1` | `46692171b` | mooncake 远程后端、（源，目标）配对 |

两个 review 分支的旧 head 已存为 `backup/pp_offload_review1_pre_20260925`、`backup/pp_balance_review1_pre_20260925`。

**所有 H100、GB300 的数字都是模型值**（`pp_prod_model_2026-09-24.py`），没有在 H100 上跑过。**实测只有 8 × RTX 5060。**

## 0. 结论

- **按报告复现，不另起一套。** 对照报告 §5.2 “Unified activation manager” 逐条实现（§1）。titan 已经在用 `torch_remat` 做函数粒度重算，它的 `saved_tensors_hooks` 本就是为 offload 留的接口：
  - host offload 和远程 offload 都是挂在这个接口上的存储后端，按张量选择；
  - AttnRes 的 block 按报告留在 GPU 上。
- **5060 实测**（46 层、block 6、8 次开 block 全在 stage 中间，pp8 × vp2，16 个 micro-batch；每组第 3 步的 loss 和 grad norm 与基线逐位一致）：
  - **关 AC、seq 512，全搬**：8 个 rank 峰值均值 7.60 → 3.08 GiB（−59%），最大值 8.92 → 3.41 GiB，各 rank 基本拉平；tps 330 → 153。
  - **关 AC、seq 1024**：基线 OOM（GPU 3 用满 15.4 GiB）；全搬后能跑，最大 5.52 GiB。
  - **FullAC、seq 2048**：均值 4.50 → 4.14 GiB，最大 5.21 → 5.09 GiB，tps 不变（435 对 438）。
  - **balance，3→7、1→6、2→5 各搬 8 个**：源 rank 各降 0.4–1.3 GiB；目标 rank 的 GPU 显存不变，因为 5060 没有 RDMA，池放在对端 host 上。
- **时机的影响（实测）**：搬得越多越省显存，但 host 链路跟不上时吞吐下降（关 AC、seq 512：搬 8、16、32 个时，tps 分别为 243、201、153）。lead 从 1 改成 2 没有换来吞吐，反而多占约 0.2 GiB。
- **模型（2.8T）**：
  - **分工**：host offload 降总量；远程 offload 只在各 rank 不均、且有 rank 还有余量时削峰。所有 rank 都超 HBM 时，balance 没有地方可搬。
  - **H100，FullAC**：offload 后 68.4–71.2 GiB，放得下；balance 叠上去不再有收益。
  - **H100，报告配方**：每层约 15 个单位可搬，host 链路 50 GB/s 时 69.6–74.5 GiB，刚好压在 HBM 上；降到 25 GB/s（两卡共用 PCIe 交换芯片）时链路饱和，offload 只能搬很少，放不下。

## 1. 与报告 §5.2 的逐条对照

| 报告原文 | 实现 |
|---|---|
| unified storage abstraction for activations, every tensor saved for the backward pass is associated with a pluggable storage backend | `torchtitan/distributed/activation_storage.py` 的 `ActivationStorage`：在 `torch_remat.saved_tensors_hooks` 上逐张量选择后端（留在设备、`HostBackend`、`RemoteBackend`） |
| Recomputation, quantization, and offload/remote-offload are storage policies, composable at tensor granularity | 重算：`torch_remat` 的 region 与 checkpoint（titan 的 `RegionAC` 和 FullAC）。offload、远程 offload：两个后端，按 (stage, micro-batch, layer) 选择。量化：**未实现**，但后端可以在 `put` 前量化、在 `get` 后反量化，接口已留 |
| policies declared via lightweight annotations on tensors, decoupled from the model code | 策略按 `current_saved_tensor_info()` 的 kind、在产生处捕获的 (stage, micro-batch, layer) 上下文以及张量性质来定，模型代码一行没改 |
| Recomputation is performed at function granularity, which supports cross-layer recomputation | 沿用 `torch_remat`（`remat.region`、`remat.checkpoint`、`RegionAC`） |
| all GPU memory is allocated on the main compute stream and managed within a single memory pool | 取回用的设备缓冲在计算流上分配；拷出期间一直持有源张量的引用，直到拷贝事件完成，不用 `record_stream`，所以显存都在计算流的池里 |
| activations are prefetched back at layer granularity and overlapped with computation | 反向中某层第一次 unpack 就预取下一层、释放上一层；每段反向的第一层，按 PP 计划在其反向之前 lead 个动作预取 |
| The block representation ... residing directly on the GPU | store 的 view（不连续）和 stage 自己的输入一律留在设备 |
| cache-based pipeline communication ... reaching the theoretical lower bound | V4，再加输入梯度 send 提前释放（`81b30fd88`）：等待点是本 rank 上第一个“其输入由接收方在消费完该梯度之后产出”的前向，由 `pipeline_order` 推出 |
| remotely offload activations to the memory of other PP ranks using the Mooncake Transfer Engine | `RemoteBackend`：目标 rank 预先注册一块池，各源 rank 平分；经已注册的 staging 槽，用 mooncake 的流序接口 `transfer_write_on_cuda`、`transfer_read_on_cuda` 在存储流上传输；池或 staging 满了就留在设备上 |

## 2. 代码

**`pp_review_optimize`（新增一个提交）**
- `stage.py`：`_grad_send_wait_points`、`_GradSendWaits`；stage 自己发出反向 send，在等待点 wait。

**`pp_offload_review1`**
- `torchtitan/distributed/activation_storage.py`（新文件）：`ActivationStorage`、`HostBackend`。
  - 同一张量被多处保存时，按对象加 weakref 去重。
  - 只搬独占整块存储的张量，视图、空张量和小于阈值的张量都不搬。
- `pipeline_parallel/activations.py`（新文件）：`PPOffloadKnobs` 和 `ActivationPlan`。
  - 计划按“保存量 × 从前向到反向相隔的动作数”排序，选出要搬的 stage-micro-batch。
  - 跳过前向与反向相隔太近的项。
  - 每段反向第一层的预取由计划触发。
- stage 前向套上管理器；反向结束时释放；零气泡类 schedule 把反向拆成两步时，在权重梯度之后释放。
- `KimiK3Model.Config.pp_offload` 打开这一功能。

**`pp_balance_review1`**
- `RemoteBackend`：池分配器沿用旧草案；新增 segment 预热重试；目标 rank 一直持有后端（否则引擎会被回收，segment 被撤销）。
- `PPBalanceKnobs`：配对、每个源搬几个、lead、池大小、staging 大小。
- 计划里远程先拿“最久”的一批，host offload 接着拿下一批。
- `KimiK3Model.Config.pp_balance` 打开这一功能。

**测试（CPU 52 项全过）**
- `test_activation_storage.py`：titan 的 FullAC 与 `remat.checkpoint` 下，梯度逐位一致；首层预取后没有临时取回。
- `test_kimi_k3_pp_offload.py`：2 进程 gloo，真实 Interleaved1F1B。host offload 搬 2 个和搬全部时，梯度与 loss 都逐位一致、没有临时取回；远程变体走 mooncake 的 TCP 同步接口，也逐位一致。
- `test_activation_storage_pool.py`：池分配器。
- 原有的 PP 测试全部保留并通过。

## 3. 5060 实测（kit：`kit_pp_optimize_2026-09-24/results/`）

46 层、block 6、dim 2048，pp8 × vp2，16 个 micro-batch；第 3 步每 rank 峰值 allocated（GiB）。

| 配置 | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | 最大 | 均值 | tps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullAC seq 2048，基线 | 4.09 | 3.89 | 4.49 | 4.59 | 4.97 | 4.12 | 4.60 | 5.21 | 5.21 | 4.50 | 435 |
| FullAC seq 2048，offload 全部 | 3.51 | 3.62 | 3.88 | 4.40 | 4.42 | 3.98 | 4.21 | 5.09 | 5.09 | 4.14 | 438 |
| 关 AC seq 512，基线 | 8.18 | 8.24 | 8.24 | 8.92 | 7.83 | 7.08 | 6.36 | 5.91 | 8.92 | 7.60 | 329 |
| 关 AC seq 512，offload 8 个 | 7.06 | 6.93 | 6.97 | 7.81 | 6.74 | 5.89 | 5.09 | 4.59 | 7.81 | 6.39 | 243 |
| 关 AC seq 512，offload 16 个 | 5.87 | 5.92 | 6.07 | 6.70 | 5.65 | 4.71 | 3.80 | 3.10 | 6.70 | 5.23 | 201 |
| 关 AC seq 512，offload 全部 | 2.94 | 2.93 | 2.98 | 3.41 | 3.21 | 3.05 | 3.04 | 3.10 | 3.41 | 3.08 | 153 |
| 关 AC seq 512，offload 全部，lead 2 | 3.13 | 3.12 | 3.16 | 3.62 | 3.45 | 3.29 | 3.55 | 3.26 | 3.62 | 3.32 | 154 |
| 关 AC seq 512，balance 8 个 | 8.18 | 6.98 | 7.23 | 8.50 | 7.83 | 7.08 | 6.36 | 5.91 | 8.50 | 7.26 | 151 |
| 关 AC seq 512，balance 8 个 + offload 8 个 | 7.06 | 5.97 | 6.33 | 7.38 | 6.74 | 5.89 | 5.08 | 4.59 | 7.38 | 6.13 | 128 |
| 关 AC seq 1024，基线 | OOM | | | | | | | | | | |
| 关 AC seq 1024，offload 全部 | 3.99 | 4.04 | 4.13 | 4.82 | 5.52 | 4.78 | 4.38 | 4.49 | 5.52 | 4.52 | 166 |

- 同一 seq 下各组的 loss 与 grad norm 逐位一致（FullAC：8.02849 / 6.59719 / 4.72565；关 AC seq 512：8.05476 / 6.58087 / 4.92833）。关 AC 的基线在暖缓存上重跑过一次，显存逐字节相同。
- FullAC 下 offload 全部时，最初 r7 反而升了 0.6 GiB：最后一段的前向后面紧跟着反向，搬出去等于马上又要取回，两份同时占着显存。计划加上“跳过前反向相隔太近的项”之后，所有 rank 都下降，上表就是修正后的结果。
- 5060 没有 NVLink，也没有 RDMA，8 张卡共用 PCIe；远程池在 host 上，走 TCP 回环。所以这里的 tps 只反映这台机器的带宽，不代表 H100。

## 4. 模型（2.8T，未实测；汇总 `prod_model_summary_2026-09-25.txt`，明细 `prod_*.out.txt`）

- **假设**：
  - 跨 rank 400 Gb/s：H100 上一半让给 EP 的 all-to-all，GB300 上 EP 在 NVL72 内，所以全部留给 PP；
  - host 链路：H100 每卡每方向 50 GB/s（另做了 25 GB/s 的敏感性），GB300 200 GB/s；
  - AttnRes 聚合用融合 kernel；
  - 两种配方：FullAC，以及报告的配方（逐元素算子重算、FP8 存储，每层约 15 个单位可搬）。
- **做法**：离散事件模拟，按层拷出、提前一层预取，预取来不及时计入等待。在 1% 的 step 时间预算内，搜索每个 rank 搬几个、lead 取多少、配对怎么配。

| 场景 | V4 + 提前释放，最小 / 均值 / 最大 | + offload | + balance（单独） | + 两者 |
|---|---|---|---|---|
| H100 PP8 × VP4，FullAC | 75.8 / 78.9 / 87.4 | 68.4 / 69.8 / 71.2 | 75.9 / 79.9 / 85.3 | 同 offload |
| H100 PP16 × VP2，FullAC | 67.9 / 79.7 / 92.8 | 64.3 / 70.5 / 75.9 | 74.4 / 79.9 / 86.8 | 同 offload |
| GB300 PP4 × VP4，FullAC | 87.6 / 90.8 / 93.8 | 79.7 / 80.3 / 80.8 | 89.4 / 91.7 / 93.0 | 同 offload |
| H100 PP8 × VP4，报告配方，host 50 | 196.8 / 231.1 / 256.7 | 69.6 / 71.4 / 74.5 | 207.1 / 232.4 / 249.2 | 同 offload |
| H100 PP8 × VP4，报告配方，host 25 | 同上 | 最大 248.2（链路饱和） | 最大 249.2 | 最大 248.2 |
| GB300 PP4 × VP4，报告配方 | 213.4 / 243.1 / 266.6 | 82.2 / 82.5 / 83.0 | 238.0 / 250.8 / 262.5 | 同 offload |

- **lead 的影响**（H100 PP8 × VP4，FullAC）：lead 1 到 16 都没有等待，但显存从 71.2 升到 75.6 GiB，所以 lead 取 1。host 25 GB/s、报告配方时，lead 从 1 加到 8，等待从 0.315 s 降到 0。
- **store 整块 offload**（不在报告里，只作对照）：H100 FullAC 能再降约 8 GiB，但 host 流量从 245 GB 增加到 2027 GB/step。

## 5. 限制与下一步

- **没有 H100 实测。** 下一步按 `PP_OPTIMIZE_REPORT_2026-09-24.md` §9 在 4 × H100 上跑。H100 有 NVSwitch；如果 host 上装了 RDMA 网卡，远程池会落在对端 GPU 上，这条路径在 5060 上测不到。
- **量化（FP8）没有实现**，只留了后端接口。
- **远程后端的 RDMA 路径没测过。** 5060 只测了 TCP。segment 预热重试和“目标 rank 一直持有后端”都是这次冒烟中修掉的问题。
- **参数靠人工给定。** 搬多少、lead 取多少，目前按模型结论手工设置；以后可以用第一步实测的动作时长来自动定。
- **组合覆盖不全。** 零气泡类 schedule 拆开的反向、TP/SP、EP、FSDP>1、CP 的组合都没测；K3 探针用的是 FullAC，还没用上 `RegionAC` 的 region 策略。
