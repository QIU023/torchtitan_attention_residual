# Elfie 10-01 的 2.78T 全模型结果：解读（2026-10-02 整理）

来源：
- pytorch/torchtitan#4272 的评论 [5941709055](https://github.com/pytorch/torchtitan/issues/4272#issuecomment-5941709055)，elfiegg 发于 2026-10-01 22:13 UTC，22:17 改过一次。
- Shuhua 的回复 [5942708609](https://github.com/pytorch/torchtitan/issues/4272#issuecomment-5942708609)，发于 23:33 UTC。

10-02 用 GitHub API 读取，上游代码核对到 main `33a2dc476`。下文表里 GB200 上的数字都是 Elfie 实测、照她原文转录的；带"估算"或"读代码"标记的是我们这边推出来的，没有实测。

## 结论

- **能跑，但只在小规模上：** 128 卡 GB200、seq 16、跑了 50 步。这只证明全模型能建起来、能前向反向、DistMuon 能更新。
- **代表性规模全失败：** 三个 256 卡配置（seq 2048 / 4096）都在第一次参数更新之前就失败了。
- **数值正确性还没有证据，原因有两个：**
  - 通过的那次 loss 比 ln V 还高，像是合成 token 或随机初始化；
  - 她的集群上，NCCL 网络插件会把通信结果弄错。
- **SiTU-GLU 编译：** 她说 97e673b77 "让 K3 可以编译"，打算 rebase 后验证。这个说法不对，97e673b77 不会编译 SiTU-GLU。Shuhua 回复里指向的 #5008 才是真正的修复。#5008 由 Jessica Zhong 在 10-01 23:11 UTC 开出，叠在已合入的 #4985 之上，10-02 01:38 UTC 合进 main（`6404b9a11`）。

## 1. 她的四次运行（GB200 实测，转录）

| 模型 / 模式 | GPU | PP / DP / EP | seq | micro-batch × 个数 → 全局 batch | 调度 / 重算 | 结果 |
|---|---:|---|---:|---|---|---|
| 2.78T QAT + DistMuon | 128 | 1 / 128 / 64 | 16 | 1 × 1 → 128 | 无 PP；SAC | 通过 50 步；loss 14.01 → 12.09；确认有更新、有 trace；峰值 140.3 GiB |
| 2.78T QAT + DistMuon | 256 | 8 / 32 / 32 | 2048 | 2 × 64 → 4096 | 1F1B；自定义 SAC | 失败：QAT 反向 OOM；零次更新 |
| 2.78T QAT + DistMuon | 256 | 8 / 32 / 32 | 2048 | 2 × 64 → 4096 | 1F1B；原生 Trainer，full AC | 失败：HybridEP 编译时找不到文件 / CWD；零次更新 |
| 2.78T DistMuon（不开 QAT） | 256 | 4 / 64 / 64 | 4096 | 1 × 8 → 512 | Interleaved1F1B；full AC | 失败：SiTU-GLU 重算 OOM；零次更新 |

## 2. 通过的那次说明了什么

**配置。**
- 不开 PP，FSDP 切 128 份；EP64 把 896 个专家分到每卡 14 个。
- 每步 128 条 × 16 token，共 2048 token。

**说明了什么。**
- 全模型在 128 卡上能建起来，前向、反向都能跑，DistMuon 确实更新了 50 步。
- 比 #4791 正文里那次 smoke 往前走了一步。那次只跑 2 步，用的是无状态 SGD 和模拟的 grouped GEMM，也是 seq 16。

**为什么不能当数值证据。**
- K3 词表 163840，ln V = 12.007。第一步 loss 14.01 比它还高，50 步后的 12.09 也只是贴到它上面。
- 如果是发布权重跑真实文本，第一步只该有几个 nat。所以有三种可能：
  1. 用的是合成 token；
  2. 随机初始化；
  3. 发布权重加真实数据，但权重没加载对。
- #4791 的 smoke 写明用了合成 token。这次评论没写数据和初始化。
- 在均匀随机 token 上，loss 的下界就是 ln V；降到 12.09 只说明更新方向对。
- 能读进来、loss 有限，都不等于加载正确。加载正确要看两样之一：
  - 真实文本上第一步的 loss；
  - 和参考实现对 logits。
- 她列的剩余工作里，参考数值对齐（reference numerical parity）也还没做。

**峰值 140.3 GiB（估算）。**
- seq 16 的激活可以忽略。比如 logits 只有 16 × 163840 × 4 B ≈ 10 MB。所以峰值基本都是静态部分。
- 2.78T / 128 ≈ 217 亿参数/卡，140.3 GiB ≈ 1507 亿字节，约合 6.9 字节/参数。
- 到 256 卡时静态部分减半。

## 3. 三个 256 卡配置

三次都是零次更新，第一步都没跑完。所以在全尺寸下，以下几样都还没有一步完整跑通过：
- PP（#4312 的路径）；
- HybridEP；
- QAT 的反向。

| 配置 | 每步 token | 失败点 | 她的归因 / 修复 |
|---|---:|---|---|
| PP8 / DP32 / EP32，seq 2048，1F1B，自定义 SAC | 4096 × 2048 ≈ 839 万 | QAT 反向 OOM | fake quant 的临时显存，AO #4917 按块分段；256 卡上还没验证 |
| 同上，原生 Trainer + full AC | 同上 | HybridEP 运行时编译找不到文件 / CWD | 排查中；单独跑 EP32、PP1 能过，私有 scratch 目录也没解决 |
| 不开 QAT，PP4 / DP64 / EP64，seq 4096，Interleaved1F1B，full AC | 512 × 4096 ≈ 210 万 | SiTU-GLU 重算 OOM | 编译 / 融合激活函数，她说单独和分布式的调试检查都过了；修复见第 5 节 |

## 4. 她列的八个问题（按严重程度排，附上由谁来修）

| 问题 | 她的证据 / 修复 | 状态 | 由谁来修 |
|---|---|---|---|
| 通信结果出错 | 不带优化器的重放能复现；`NCCL_NET_PLUGIN=none` 时正常。#4947 正文说出错的是跨域（cross-domain）的 rank | 要交给 NVIDIA 内部的 NCCL 团队 | NVIDIA NCCL |
| NCCL workspace 分配太晚，和 PyTorch 的缓存抢显存 | `NCCL_RUNTIME_CONNECT=0` 把建连提前到初始化，复现过的 OOM 解决了 | 她建议训练里默认打开；Shuhua 请她开个 issue，方便复现和修 | titan 决定默认值 |
| fake quant 临时显存太多 | 按块对齐分段（AO #4917，草稿） | 单项检查过了，256 卡没验证 | Elfie / torchao |
| SiTU-GLU 反向 OOM | torch.compile 融合激活函数 | #5008 已合（10-02 01:38 UTC，`6404b9a11`；第 5 节） | titan（Jessica Zhong） |
| 发布的打包 MXFP4 权重加载不对 | #4791（草稿，+3497 / −32，31 个文件），内容见表下 | 要和 Ivy Zhou 在 torch core 里的 checkpoint reader 对齐 | Elfie / torch core DCP |
| 加载视觉权重时多出通信 | #4947（草稿，+102 / −1）：拼接前把视觉 Q/K/V 显式设成 replicate | 回归测试过了；它只是避开那次多余的重分布，没修传输本身 | Elfie |
| HybridEP 运行时编译缺文件 / CWD | 单独跑 EP32、PP1 能过 | 排查中 | titan 的 `torchtitan/distributed/deepep/hybridep.py` 接入，或 HybridEP 本身 |
| profiler 开 shape 记录时挂住 | `record_shapes=False` 时 full128 能过 | 绕过去了，原因没查 | PyTorch profiler |

#4791 做了这几件事：
- MXFP4 解码成 BF16 主参数；
- 按 manifest 选出要做 QAT 的模块；
- `A_log` 去掉零填充，`dt_bias` reshape。

背景（NCCL 文档）：NCCL 从 2.22 起默认按需建连（`NCCL_RUNTIME_CONNECT=1`），第一次用到某条连接时才分配缓冲。titan 现在对环境变量的设置是：
- `run_train.sh` 第 40 行只设了 `PYTORCH_ALLOC_CONF=expandable_segments:True`；
- 代码里的 NCCL 变量（`torchtitan/distributed/utils.py` 第 304 到 328 行）只在 batch-invariant 模式下才设。

把 `NCCL_RUNTIME_CONNECT=0` 设成默认是有代价的：所有连接在 init 时就建好，多占一些显存，启动也更慢。

## 5. SiTU-GLU 编译：核对记录

**97e673b77 不会编译 SiTU-GLU。**
- 这个提交是上游 #4895 "[Compile] Remove TransformerBlock compilation"，Jessica Zhong 09-29 提交。它把标准 Trainer 里按 TransformerBlock 编译的功能整个删了。
- 在 `kimi_k3/model.py` 里它只删了 3 行，就是 "Kimi K3 does not support model compilation yet." 那条拒绝语句。删它是因为它守的那个选项已经没了，不是因为 K3 现在可以编译。

**main 现在怎么编译。**
- 用 `LocalCompileConfig.regions`（`torchtitan/distributed/local_compile.py`），按名字注册函数；名字没注册过就报 `Unknown compile.regions entries`。
- 在 `33a2dc476` 上注册了的有：gated_rmsnorm、loss、swiglu、cos_sin_rope，再加 qwen3_5 的 offset_rmsnorm。
- `SiTUGLU.__call__`（`models/common/activation.py`）没有注册。所以只 rebase 到 97e673b77 或 main，SiTU-GLU 仍然是 eager。

**#5008 才是修复。**
- 标题 "[Perf][Compile] Add local compilation for SiTUGLU"。写这份笔记时还没合，10-02 01:38 UTC 合进了 main（`6404b9a11`）。
- 改动是给 `SiTUGLU.__call__` 加上 `@local_compile("situglu", batch_invariant=True)`，并默认打开，加了 GPU 测试。
- 它叠在 #4985（cos/sin RoPE，10-01 23:57 UTC 已合）之上。
- 正文给的是 H100 上单个算子的耗时，不是显存。例如路由专家 [65536, 3072] 前向加反向，从 22.555 ms 降到 1.561 ms（14.45 倍）。
- 正文还说：路由路径仍然在整块带填充的缓冲上算，因为 SiTU-GLU 不读 offsets。

**显存（读代码推出，没实测）。**
- eager 下 SiTU-GLU 在 fp32 里算，为反向存下 6 个 fp32 张量，约 24 字节/元素：
  - 两个 tanh 的输出；
  - 两次乘标量的结果；
  - sigmoid 的输出；
  - 门那一路的乘积。
- 编译后，AOTAutograd 的分区器一般会在反向里重算逐元素的运算，预计只存两个 bf16 输入，约 4 字节/元素。
- 用到它的地方有三处，都在 `flavors.py`：dense FFN（`_feed_forward_config`）、路由专家、共享专家。
- 在 full AC 下，这是重算某一层和跑它的反向时的瞬时量。它随每卡收到的路由行数（含填充）增长。
- 到 10-02 为止，还没有人测过 #5008 对显存的实际影响。

## 6. 和我们几条线的关系

- **#4312（PP，09-26 已合）：** 在两个 PP 配置的路径上，但两个都在第一步前就挂在了别处。所以全尺寸 PP 既没出问题，也还没被验证。她的问题清单里没有 PP 相关的项。
- **PR A（`pp_review_optimize`）、#4656、#4764 / #4765：** 它们省的分别是跨 stage 的块、AttnRes 的激活和 PP 各 rank 之间的不均衡，和这次的 OOM（QAT 临时量、SiTU-GLU）不是同一处。等那两处修好以后，它们才决定还剩多少余量。
- **MoonEP（#4751）：** 不在她的栈上，她用的是 HybridEP。tracking issue 里 MoonEP 也只是 P1 的可选项。
- **DEP（#4381）、视觉 CP（#4380）：** 都属于多模态。全尺寸多模态还在她的剩余工作里。

## 7. 待定的事（等用户定，没动）

- SiTU-GLU：Shuhua 已经指向 #5008，不用我们再提。#5008 已经合进 main（`6404b9a11`），所以她 rebase 到这之后的 main 就有编译过的 SiTU-GLU；只 rebase 到 97e673b77 是不够的。
- 问清 128 卡那次的数据和初始化。要证明加载正确，建议做以下任一项：
  - 用发布权重在真实文本上跑一步前向，loss 应该在几个 nat；
  - 和 vLLM 对 logits。
- 以后读她的任何数值，先确认 NCCL 网络插件是怎么设的。

## 8. 她列的剩余工作（转述）

- 完成 full256 QAT 的性能 trace。
- 不开 QAT、编译 SiTU-GLU 后重试，另做一个缩小的 PP2 编译诊断。
- 给出代表性的吞吐、显存、DCP 重启恢复和真实数据上的表现。
- 完成参考数值对齐、长上下文 CP、原生低精度 GEMM，以及全尺寸多模态 / LoRA / QB 的验证。
- KDA 的 `A_log` / `dt_bias` 保持 FP32 这件事，在 #4272 里单独跟踪。

## 核对方法

- **GitHub API（未登录）：**
  - 评论 5941709055、5942708609；
  - torchtitan 的 PR #4791、#4947、#4985、#5008；
  - pytorch/ao#4917。
- **上游代码：** 在 torchtitan 子模块里 `git fetch upstream main`，得到 `33a2dc476`。看了：
  - `git show 97e673b77`；
  - `git grep 'local_compile("'`；
  - `local_compile.py`、`activation.py`；
  - `run_train.sh`、`distributed/utils.py`。
- **K3 全尺寸配置（`kimi_k3/flavors.py`）：**
  - dim 7168，93 层，AttnRes 每块 12 层；
  - 词表 163840，96 个头；
  - 896 个专家取 top-16，latent 维 3584，专家隐藏 3072，dense 隐藏 33792。
