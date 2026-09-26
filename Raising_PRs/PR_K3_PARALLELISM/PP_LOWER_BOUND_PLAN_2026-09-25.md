# PP cache 显存下界：pp_review_optimize 复审与后续 PR 计划（2026-09-25）

依据：
- AttnRes 原文（Kimi Team，arXiv 2603.15031）第 6 到 7 页：§4.1 的 "Pipeline communication""Cross-stage caching""Memory overhead" 三段，Fig. 3，Eq. 7/8，Algorithm 1。
- K3 报告 §5.2.2 "Memory-efficient Training"（`/tmp/k3report.txt` 第 1365 到 1394 行）：Unified activation manager、Memory-efficient Attention residual、Balancing activations across PP ranks 三段。
- 同日的 `LOWER_BOUND_AUDIT_2026-09-25.md` 已逐句对照原文，这里不重复，只用它的结论。

**数字说明：** 标"实测"的是 8 × RTX 5060 上跑出来的；其余显存数字都是模型值，来自 `pp_memory_model_v4b_2026-09-25.py`（输出 `pp_memory_model_v4b_2026-09-25.out.txt`）。模型只算 block 和 hidden，不含逐层激活和静态显存。H100、GB300 都没有实测。

## 0. 结论

1. **pp_review_optimize 的机制与原文一致，显存还没到下界。**
   - 机制：只做相邻传递，每跳只传接收方还没有的 block，反向用同样的办法（deposit）。原文没有跳发，我们也没有。
   - 状态：已 rebase 到 4312 的 head `7814d1f8b`，现在是 `9ac65f173`，CPU 单测 110 项通过；8 × 5060 上组合格跑 10 步 rc=0（实测，格子没固定 seed，只说明能跑通）。
   - 显存（模型值，H100 PP8×VP4 最重 rank）：当前分支 38.9 GiB，报告口径的下界 20.9 GiB，更紧的下界 17.4 GiB。
2. **剩下的差距是三项，从大到小：**
   - stage 中间开 block 时，模型用 `torch.cat` 复制出整条 stack，后面各层的 checkpoint 一直存着这份副本（open_copy）。去掉它：38.9 → 28.7。它只出现在开 block 的 rank 上，所以主要抬高最大值。
   - hidden 输出一直留到反向（hidden_out）：38.9 → 35.5。这是 torch PP 的通用行为，任何 PP 模型都有。
   - store 按本 rank 需要的最多行数一次分配（per_block）：38.9 → 36.3。
   - 三项都去掉是 21.5，比报告口径只多 0.6 GiB。多出的是输入梯度 send 的残差，要改 torch 的 schedule。
   - 再把每个 block 的释放点提前到"把它带进这个 rank 的 stage"（下称 bringer）的反向，就到更紧的 17.4。
   - PR A（§4.1、§4.2）做掉 per_block 和 open_copy，并在 bringer 反向时释放：最大值 21.4、均值 20.2，与报告口径的 20.9 / 19.2 相当；hidden_out 留给 torch，那边改完是 18.0 / 16.8。
3. **"统一内存管理器管 PP rank cache block"：报告原文没有这样写，但我们的实现必须这样做。**
   - 报告的统一激活管理器管的是 "every tensor saved for the backward pass"。
   - 对 block，报告只说 "generated once at the boundary layer and shared by all subsequent layers, residing directly on the GPU"，并说 AttnRes 整个放在 checkpoint 里，使每层保存的激活与标准残差相同。
   - 在我们的实现里，store 里的 block 就是各层 checkpoint 保存的张量。所以管理器必须认得它们：常驻 GPU、不能搬、计入每个 rank 的峰值、由 store 释放。现在两者互不知道（§4.3）。
4. **offload（#4765）和 balance（#4764）的后端没问题，计划不是最优（§5）。**
   - 计划不看每个 rank 的峰值出现在什么时候，每个 rank 搬同样多个。
   - balance 的配对、个数、池大小全靠手填，池在各源之间平分。报告的目标是 "balanced activation memory across PP ranks"，这要按每个 rank 的显存时间线算出来。
   - 两个草案都是按当前的 block 显存调的；block 显存降下来之后，各 rank 的轮廓会变，要重新建模。
   - 现在的 offload 会把 open_copy 的副本搬到 host，把带宽花在一份本不该存在的显存上。
5. **建议的顺序（§6）：** 先向 torch 提 issue（通用的那部分），再做 PR A（K3 的 block 显存到下界），然后重写 #4765 和 #4764 的计划。§6 列了需要你定的五件事。

## 1. 原文与报告的要求，逐条对照当前分支

| 出处 | 要求 | pp_review_optimize |
|---|---|---|
| 原文 Fig. 3、"Cross-stage caching" | 只做相邻传递，每跳只传增量 | 满足。layout 表保证不重复投递，总量是精确的最小值 |
| 原文 "The backward pass benefits from the same scheme" | 反向也只传增量 | 满足。deposit 在本 rank 上累加，只把 delta 的梯度发回 |
| 原文 "Memory overhead"：each block is stored exactly once across all V virtual stages | 每个 rank、每个 micro-batch，每个 block 只存一份 | 基本满足。delta 直接收进 store，stack 和 payload 都是 store 的 view。例外是开 block 时 `cat` 出的整条 stack（open_copy） |
| 原文同段：activation checkpointing eliminates all inter-block attention intermediates, and the checkpointed input p_l matches the memory size of the hidden state h_l it replaces | 每层 checkpoint 只多存一个 [T, D] 的 p_l，引用共享的 block，不存副本 | 不满足。开 block 之后，本 stage 后面各层的 checkpoint 存的是 `cat` 出来的 [T, N+1, D] |
| 报告：generated once at the boundary layer and shared by all subsequent layers, residing directly on the GPU | block 生成一次，后续各层共享，留在 GPU 上 | 生成后被 `cat` 复制，不是共享。留在 GPU 上这一条满足：offload 草案不搬 view |
| 报告：The AttnRes computation is entirely wrapped with checkpointing | 聚合在 checkpoint 里 | FullAC 下满足；选择性 AC 要靠 #4780 的 remat region |
| 报告：released as soon as the micro-batch finishes, reaching the theoretical lower bound | 本 rank 上这个 micro-batch 的最后一次反向时释放 | 释放点满足：store 在本 rank 第一个 stage 反向结束时释放。但 store 是按本 rank 需要的最多行数一次分配的，以后才到的 block 的行也从一开始就占着（per_block） |
| 报告 "Unified activation manager"：all GPU memory is allocated on the main compute stream and managed within a single memory pool | 单一缓存池，在计算流上分配 | store 用 `new_empty` 在计算流上分配，满足。与管理器的约定见 §4.3 |
| 原文 Algorithm 1（推理用的两阶段计算） | 块间一次批量注意力，块内用 online softmax 逐层合并 | 训练里不需要照搬。它的 online softmax 合并正是"对 block 列表聚合、不拼 stack"的算法（§4.1） |

## 2. pp_review_optimize 复审

分支 `pp_review_optimize` = `9ac65f173`，在 4312 的 `7814d1f8b` 之上三个提交（09-25 rebase，旧 head 备份为 `backup/pp_review_optimize_pre_20260925` = `81b30fd88`）：
- `f58088615`：delta 收进 rank store，stack 和 payload 都是 store 的 view；
- `91d3f4bdc`：所有接收缓冲按需分配；前向 send 在本 stage 对这个 micro-batch 的反向开始时 wait；
- `9ac65f173`：输入梯度的 send 在第一个能证明对端已收到的前向时 wait。

**验证：**
- 09-25：CPU 单测 110 项通过，含 4 进程 gloo 的 block 梯度测试；8 × 5060 组合格 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4` 跑 10 步 rc=0，loss 8.02765 → 3.45315（实测，smoke）。
- 09-24（rebase 前）：探针最重 rank 13.38 → 8.04 GiB，100 步 loss 和 grad norm 逐位一致（实测，`PP_OPTIMIZE_REPORT_2026-09-24.md` §2）。

**复审意见：**
1. **通用的部分应该进 torch，不该长期留在模型目录里。**
   - 09-24 的实测分解（该报告 §3）里，有两块对任何 PP 模型都成立：
     - send 到 step 末才 wait，Work 一直钉住被发送的张量。改成自己持有前向 send 之后，每 rank 省 0.52 到 1.30 GiB；
     - hidden 的接收缓冲按 micro-batch 常驻。改成按需分配之后，每 rank 再省 0.29 到 0.74 GiB。
   - 在 2.8T 的尺寸上，torch 常驻的 hidden 接收缓冲（输入和梯度各一份）是每 rank 112 到 128 个 [T, D]，即 12 到 14 GiB（模型值：分配规则 09-24 实测确认，大小按 8K × 7168 bf16 算）。同样尺寸的任何模型都要付这笔。
   - 现在这些都是在 `AttnResPipelineStage` 里重写 torch 的私有方法做的：V4 新重写了 `_setup_forward_recv_info`、`_setup_backward_recv_info` 和四个 `get_*_ops`，另外用到 `_batch_p2p`、`_make_tensor_from_meta`、`_PipelineScheduleRuntime`。
   - 按抽象规则（core 的缺口是上游 issue），这部分先向 torch 提 issue，附 09-24 的实测。K3 的 PR 里暂时保留这些重写，等 torch 有了对应接口再删。
2. **注释超标。** V4 新增 22 行注释和 docstring，大多在讲设计理由（比如 "torch's default keeps one per micro-batch across steps"）。按"默认不加注释"规则，进 PR 前删到只剩代码本身表达不了的约束（比如 `.data` 那一行）。
3. **store 和 stage 依赖"连续的行"。** 就地接收要求 delta 是 store 里连续的几行、stack 是前 N 行，所以有两处运行时检查。这是 [N, T, D] 整块缓冲带来的约束。改成按 block 分配之后，这两处检查和 `.data` 绕开版本号的写法都不再需要（§4.2）。
4. **payload 是 store 的 view，所以前向 send 会钉住整个 store 缓冲。** action-list runtime 下 send 在本 stage 反向时就 wait，这时 store 反正还在，不多占。单 stage schedule 下改发副本。按 block 分配之后，send 只钉住它发出的那几个 block。
5. **deposit 每次都新分配。** `deposit` 用 `clone()` 或 `prior + grad` 建新张量，可以改成原地加。量小，不改变结论，顺手改。
6. **聚合有较大的临时峰值。** `_apply_attention_residual` 先 `cat`，再整体转 fp32，前向时 values、keys 和两者的乘积三个 fp32 的 [T, N+1, D] 同时存在。2.8T 最后几层 N+1 = 9，eager 下约 5.9 GiB（模型值；compile 能融合掉多少没测）。它不在上面的常驻数字里，但会叠加到峰值上。

## 3. 差距（模型值，GiB，只含 block 和 hidden）

列的含义：
- V4：09-24 的 V4；+81b：再加 `81b30fd88`，即当前分支。
- +per_block、+open_copy、+hidden_out：各自单独在 +81b 上去掉一项；pb+oc：去掉前两项；all3：三项都去掉。
- PR A：去掉 per_block 和 open_copy，并在 bringer 反向时释放 block（§4.1、§4.2），hidden_out 和梯度 send 残差仍在。PR A+hidden：再去掉 hidden_out（torch 的通用改动，§4.4）。
- report：报告口径的下界（每个 block 每 rank 一份，micro-batch 结束时释放）；tight：更紧的下界（bringer 反向时释放）。

**最大值与均值（全部 rank）：**

| 切分 | V4 | +81b（当前） | +per_block | +open_copy | +hidden_out | pb+oc | all3 | PR A | PR A+hidden | report | tight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5060 探针 PP8×VP2，32 层，block 4，seq 3584 | 3.2 / 2.8 | 2.5 / 2.2 | 2.1 / 2.0 | 2.5 / 2.2 | 2.3 / 2.0 | 2.1 / 2.0 | 1.9 / 1.8 | 1.7 / 1.6 | 1.5 / 1.4 | 1.8 / 1.6 | 1.3 / 1.2 |
| 5060 PP8×VP2，46 层，block 6，seq 2048 | 2.0 / 1.7 | 1.7 / 1.4 | 1.5 / 1.2 | 1.3 / 1.2 | 1.6 / 1.2 | 1.1 / 1.1 | 1.0 / 0.9 | 0.9 / 0.8 | 0.8 / 0.7 | 0.9 / 0.8 | 0.7 / 0.6 |
| H100 PP8×VP4，93 层，block 12，M16 | 43.1 / 34.8 | 38.9 / 29.8 | 36.3 / 26.6 | 28.7 / 27.2 | 35.5 / 26.3 | 24.9 / 23.6 | 21.5 / 20.2 | 21.4 / 20.2 | 18.0 / 16.8 | 20.9 / 19.2 | 17.4 / 15.8 |
| H100 PP16×VP2，M32 | 54.9 / 45.8 | 47.1 / 34.8 | 41.8 / 31.2 | 37.0 / 32.0 | 43.8 / 31.3 | 30.4 / 28.5 | 27.9 / 25.8 | 23.6 / 21.7 | 21.0 / 18.9 | 26.6 / 23.5 | 19.6 / 16.8 |
| GB300 PP4×VP4，M16 | 28.2 / 26.6 | 19.7 / 16.5 | 17.9 / 14.9 | 14.7 / 13.9 | 18.0 / 14.7 | 12.7 / 12.1 | 11.0 / 10.4 | 10.9 / 10.4 | 9.3 / 8.7 | 10.4 / 9.6 | 8.6 / 7.9 |
| GB300 PP2×VP8，M16 | 33.8 / 30.0 | 15.9 / 13.6 | 15.6 / 13.0 | 11.3 / 11.1 | 14.0 / 11.8 | 10.3 / 10.2 | 8.6 / 8.5 | 9.4 / 9.4 | 7.8 / 7.6 | 8.0 / 8.0 | 7.1 / 7.1 |

每格是"最大值 / 均值"。93 层的切分都按 8K token × 7168、bf16 算。

**H100 PP8×VP4 的每个 rank：**

| rank | V4 | +81b | +per_block | +open_copy | +hidden_out | pb+oc | all3 | PR A | PR A+hidden | report | tight |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 41.0 | 36.4 | 36.0 | 25.2 | 32.2 | 23.1 | 18.8 | 19.7 | 15.4 | 18.2 | 14.8 |
| 1 | 31.3 | 26.8 | 24.3 | 26.8 | 22.8 | 24.3 | 20.3 | 20.9 | 17.0 | 19.2 | 15.9 |
| 2 | 30.8 | 26.4 | 23.4 | 26.4 | 22.5 | 23.4 | 19.9 | 20.0 | 16.5 | 18.6 | 15.2 |
| 3 | 30.4 | 25.9 | 22.5 | 25.9 | 22.3 | 22.5 | 19.5 | 19.1 | 16.1 | 17.9 | 14.5 |
| 4 | 43.1 | 38.9 | 36.3 | 28.7 | 35.5 | 24.9 | 21.5 | 21.4 | 18.0 | 20.9 | 17.4 |
| 5 | 33.9 | 28.5 | 24.4 | 28.5 | 25.4 | 24.4 | 21.2 | 20.9 | 17.7 | 20.2 | 16.7 |
| 6 | 33.8 | 28.1 | 23.5 | 28.1 | 25.2 | 23.5 | 20.6 | 20.0 | 17.1 | 19.6 | 16.1 |
| 7 | 33.8 | 27.7 | 22.6 | 27.7 | 24.9 | 22.6 | 19.9 | 19.1 | 16.4 | 18.9 | 15.4 |
| 最大 | 43.1 | 38.9 | 36.3 | 28.7 | 35.5 | 24.9 | 21.5 | 21.4 | 18.0 | 20.9 | 17.4 |
| 均值 | 34.8 | 29.8 | 26.6 | 27.2 | 26.3 | 23.6 | 20.2 | 20.2 | 16.8 | 19.2 | 15.8 |

**怎么读：**
- open_copy 只出现在 stage 中间开 block 的 rank 上（H100 PP8×VP4 是 rank 0 和 4）。去掉它，最大值降 10.2 GiB，均值只降 2.6 GiB。它是 rank 之间不均的主要来源。balance 可以把它搬到别的 rank，但那是花带宽搬一份本不该存在的副本。
- per_block 和 hidden_out 在每个 rank 上都降，均值和最大值降得差不多。
- 5060 的 32 层 block 4 探针看不出 open_copy。按核心的切分，它 8 次开 block 里有 7 次落在所在 stage 的最后一层，新 stack 后面没有层去存，只有 rank 0 在第 0 层开的那一次留下副本。
- 所以 5060 实测不另编层数，直接用生产的 93 层、block 12、PP8×VP4 切分，只缩小 dim 和 seq。这样 stage 划分、开 block 的位置（rank 0 和 4）和最重的 rank（4）都与 H100 PP8×VP4 的模型相同，每个 rank 的单位数一一对应。
- 这个探针在 dim 2048 时约 21 亿参数，每个 rank 的静态显存约 3.9 GiB（fp32 主参数、梯度和两份 Adam 状态），5060 放得下。
- 全部数字的明细（每个切分的每个 rank）在 `pp_memory_model_v4b_2026-09-25.out.txt`。

## 4. 设计：到下界要改什么

### 4.1 模型：block 列表，聚合不拼 stack（去掉 open_copy）

- `block_residual_TND`（一个 [T, N, D] 张量）改成 block 列表，每个元素是一个 [T, D]。
  - 开 block 时，把上一个 block 的 [T, D] 追加进列表。这只是新建一个 Python 列表，不复制张量。
- 聚合 `_apply_attention_residual` 改成逐 block 计算，用原文 Algorithm 1 第二阶段的 online softmax 合并：
  - 每个 block 算一个 [T] 的 score，维护每个 token 的最大值、分母和 fp32 的 [T, D] 累加；
  - 不拼 [T, N, D]，也不整体转 fp32。
- 反向有两种做法：
  - 交给 checkpoint 重算（FullAC，或 #4780 的 region）；
  - 或者写成一个自定义 autograd Function，只存 bf16 block 的引用和 [T, N] 的概率，反向时重算 score。这样 §2 第 6 条的临时峰值从 O(N·T·D) 的 fp32 降到 O(T·D)。
- 改完之后，每层 checkpoint 存的是列表里各 block 的引用（与 store 同一块存储）和一个 [T, D] 的 partial，正是原文 "p_l matches h_l" 的形态。
- 不开 PP 时也受益：不再每开一个 block 就复制一次 stack。
- **和 #4780 重叠：** #4780 在 `_apply_attention_residual` 的调用处套 remat region，签名一改就冲突。放在哪边见 §6。

### 4.2 PP：按 block 传、按 block 存、在 bringer 反向时释放（去掉 per_block，到更紧的下界）

- **线上格式从一个 [K, T, D] 改成 K 个 [T, D]。** stage 的输出是 `(hidden, b_i, …, b_j)`，torch PP 支持多输出，K 个 P2P 放在同一批里发。
  - 接收方每个 block 一块独立分配，收到的张量直接就是 store 的条目。不再需要连续的行，也不再需要 `.data`。
  - 转发的 block（delta 里不是本 stage 产生的）直接发同一个张量，不打包、不复制。
  - 本 stage 产生的 block 就是模型自己的 [T, D]，store 直接引用它，不再 `put` 复制一份。
- **每个 block 在一个 rank 上只有一个 bringer**，即第一个收到或产生它的 stage。
  - 按反向顺序，bringer 是这个 block 在本 rank 上的最后一个读者，它的 deposit 也在 bringer 反向时收走。
  - 所以 bringer 反向一结束就可以释放它，不必等整个 micro-batch 结束。这就是审计文档 §3 的更紧下界。按 block 存储之后，只是换了释放点，不需要别的机制。
- **前提是接收缓冲按需分配**（当前分支已有）。torch 默认的接收缓冲按 micro-batch 常驻，block 收进去就永远不会释放。

### 4.3 与统一激活管理器的约定

- #4765 草案的 `ActivationStorage` 现在靠"不是 view、独占整块存储"来判断一个被保存的张量能不能搬。
  - 当前分支的 block 是 store 的 view，所以碰巧不会被搬。
  - 按 block 分配之后，block 是连续、独占存储的 [T, D]，会被当成普通激活搬走。这违反报告的 "residing directly on the GPU"，也浪费带宽。
- 所以 store 要向管理器登记它的 block：常驻 GPU、被多个 checkpoint 共享、由 store 释放。管理器据此不搬、不重复计数，并在算每个 rank 的峰值时把 block 的时间线算进去（layout 表直接给得出来）。
- 管理器里已有按 id 保留的集合（`_keep`），登记可以先用它。更稳的是按存储登记，这样同一块存储的不同 view 也都认得出来。
- 分配仍在计算流上、同一个缓存池里，符合报告的 "a single memory pool"。

### 4.4 hidden 输出：通用 PP 改动，不放进 K3 的 PR

- torch 的 stage 把每个 micro-batch 的输出放在 `fwd_cache` 里一直到反向。反向只用它的 grad_fn，不用它的值。
- Megatron 在 send 之后把输出的存储换成一个元素的占位（`deallocate_output_tensor`），再用自己的 backward 绕过形状检查。
- 这对任何 PP 模型都成立，H100 PP8×VP4 最重 rank 上 3.4 GiB（PR A → PR A+hidden，模型值）。它应该和"接收缓冲按需分配""send 提前 wait"一起，作为 torch 的 issue 和 PR。
- 输入梯度 send 的残差（all3 比 report 多 0.6 GiB）同样属于 torch 的 schedule。

## 5. offload（#4765）和 balance（#4764）哪里不是最优

现状见 `PP_ACTIVATION_STORAGE_REPORT_2026-09-25.md`：两个草案按报告实现了统一激活存储，包括 `torch_remat.saved_tensors_hooks` 上的逐张量后端、host 后端、mooncake 远程后端和按层预取，5060 上数值逐位一致。后端的形态没有问题，问题在"搬什么、搬到哪"的计划。

1. **offload 的计划不看峰值出现的时间。**
   - `ActivationPlan` 按"保存的层数 × 从前向到反向相隔的动作数"排序，每个 rank 取同样多个（`PPOffloadKnobs.microbatches`）。
   - 一个条目只在"拷出完成"到"开始取回"这段时间里减显存。如果这段时间不覆盖本 rank 的峰值时刻，搬了也不降峰值，只花带宽。
   - 各 rank 的峰值不同（H100 PP8×VP4 仅 block 和 hidden 就在 25.9 到 38.9 GiB 之间，模型值），同一个个数对轻的 rank 是浪费，对重的 rank 可能不够。
   - 5060 上 FullAC 全搬时，r7 的峰值反而升了 0.6 GiB（实测），后来加了"跳过前向与反向相隔太近的项"才修好。这是同一个问题：计划没有对着时间线算。
2. **balance 全靠手填。**
   - 配对、每个源搬几个、池大小都是 knob，目标 rank 的池在各源之间平分。
   - 报告的目标是 "achieving balanced activation memory across PP ranks"。这需要由各 rank 的峰值算出来：谁超出、超出多少，谁有余量、余量多少。目标 rank 的池整个 step 都占着，所以只能用它自己峰值之上的空间。
3. **两者都是按当前的 block 显存调的。**
   - PR A 之后，最重 rank 的 block 和 hidden 从 38.9 降到 21.4 GiB（模型值），rank 之间也更平，因为 open_copy 只在个别 rank 上。
   - offload 要搬的量和 balance 的配对都会变。之前"H100 FullAC 下叠加 balance 没有收益"的结论要重算。
4. **现在的 offload 会搬 open_copy 的副本。** `cat` 出来的 stack 是连续、独占存储的，计划选中的 stage micro-batch 里，它会被拷到 host。PR A 之后这份副本不存在；在那之前，这部分带宽花在一份本不该存在的显存上。
5. **缺 FP8 后端。** 报告的配方是 FP8 量化加 offload。存储报告里"报告配方"那几行已经按 FP8 存储算了每层的量（约 15 个单位），但代码里没有 FP8 后端，接口只是留着。不补上它，那几行的数字在代码上不成立；host 链路降到 25 GB/s 时，即使按 FP8 算也饱和、放不下。

**改法（PR B 和 PR C 共用一个计划器）：**
- **时间线。** 每个 rank 在构建时就能从 `pipeline_order` 推出所有 rank 的动作序列。再加上：
  - 每个 (stage, micro-batch, layer) 保存的字节数（第一步实测，或由 AC 策略推出）；
  - block 的时间线（layout 表）；
  - 静态显存（第一步实测）。
  这样就得到每个 rank 的显存时间线。
- **目标水位。** 取各 rank 峰值的均值，但不低于每个 rank 自己搬不动的底。
- **贪心。**
  - 每次取峰值最高的 rank，在它的峰值时刻，找覆盖这一时刻、并且前向与反向之间留得出拷出和取回时间的条目。
  - 按"字节 ÷ 传输代价"选一个，分给代价最低、还有余量的后端：同节点的远程池优先，其次 host。
  - 更新时间线，重复，直到所有 rank 都不超过目标，或者带宽预算用完。带宽够不够，用第一步实测的动作时长来判断。
- **远程池。** 大小取目标 rank 峰值之上的余量，按各源的需要分，不再平分。
- 这样 offload 的个数、balance 的配对和池大小都由计划算出来，knob 只剩开关和带宽预算。

## 6. PR 顺序与需要你定的事

**顺序：**
1. **torch issue（通用）。** 内容：接收缓冲按 micro-batch 常驻、send 到 step 末才 wait（Work 钉住张量）、输出留到反向。附 09-24 的实测分解和 `stash_probe.py`。看回应再开 torch PR，走 PR-194033 那样的流程。
2. **PR A（K3，叠在 4312 上）。**
   - 内容：§4.1 的 block 列表和聚合、§4.2 的按 block 传输与存储，以及在 bringer 反向时释放。当前分支三个提交里的 torch 私有方法重写暂时保留。
   - 目标（模型值，H100 PP8×VP4 最重 rank）：21.4 GiB；torch 的 hidden 改动进来之后是 18.0 GiB。
3. **PR B（重写 #4765）。** 管理器与 store 的约定（§4.3）、按时间线的计划器（§5）、host 后端。FP8 后端放下一步。
4. **PR C（重写 #4764）。** 远程后端和均衡计划，与 PR B 共用计划器。

**需要你定的：**
1. **PR A 单独开，还是放进 #4765 的底部。** 现在 #4765 草案底部垫的是这三个提交 rebase 之前的版本（`14cba2237`、`202b6a974`、`81b30fd88`，在旧的 `901ef34de` 上），#4765 的 body 写的是 "The three rank-store commits are not part of #4312"。我建议单独开：它管 block 的传输和存储，#4765 管被保存的激活，是两件事，也好分开 review。但这要多一个 PR 和一个分支，需要你点头。
2. **更紧的下界放不放进 PR A。** 我建议放：按 block 存储之后，只是把释放点从"本 rank 的最后一次反向"换成"bringer 的反向"，没有别的机制，H100 PP8×VP4 最重 rank 再省约 3.5（PP16×VP2 上 6.8） GiB（模型值）。
3. **聚合接口改在哪个 PR。** 我建议放在 PR A：列表载体是 PP 显存的前提，#4780 只是在调用处套 region，谁后合并谁改一下签名。另一个选择是先在 #4780 里改接口，PR A 直接用。
4. **hidden 输出的释放。** 我建议走 torch（issue 先行），不在 K3 的 stage 里再重写一个 torch 的行为。另一个选择是先在 `AttnResPipelineStage` 里做，PR A 就能直接到 PR A+hidden 那一列。
5. **torch issue 现在就提吗。** 09-24 的 5060 实测已经够写 issue；H100 的数字要等机器。

## 7. 验证计划

- **5060（冒烟和同一性）：** 生产的 93 层、block 12、pp8×vp4、M16、FullAC、dim 2048、seq 2048（放不下就降 dim，层数和 block 不动），当前分支对 PR A 原型：
  - 每个 rank 的峰值 allocated，报均值和最大值，与模型预测对照（模型：block 和 hidden 的最大值 2.8 → 1.5 GiB，和 H100 PP8×VP4 的单位数相同，按 dim 2048、seq 2048 换算）；5060 上同配置重复运行，显存逐字节相同，噪声底是 0；
  - 同一份暖缓存上 100 步，loss 和 grad norm 逐位一致；
  - `test_kimi_k3_pp_block_grads`（4 进程 gloo）逐位一致。
- **H100（PR 里的数字）：** 按 `PP_OPTIMIZE_REPORT_2026-09-24.md` §9 的 4 × H100 方案，加一格 PR A。
- **offload 和 balance 的计划器：** 先在模型里比较新旧计划（同一带宽预算下各 rank 的峰值），再上 5060。

## 8. 修订（2026-09-26）

- **实测切分改为生产切分。** 09-25 版写的是 46 层、block 6，那是 09-24 为了在 16 个 stage 上凑出"stage 中间开 block"临时选的层数。生产的 93 层、block 12 在 PP8×VP4 上本身就是 8 次开 block 都在 stage 中间，不需要另编层数（§3、§7 已改）。
- **pp_review_optimize 的逐行审核** 见 `PP_REVIEW_OPTIMIZE_DIFF_AUDIT_2026-09-26.md`：行为上没有发现错误；µfmt 不通过、测试类放在 main guard 之后、注释超标，这几项做 PR A 时先修。
- **DEP（#4381）已 rebase 到 4312 的 `7814d1f8b` 并推送**，`k3_pp_mm` 和 `dep_review1` 都是 `232834a4d`，经过见 `PR_BODY_PP_MM_v3.md` 的状态部分。DEP 直接叠在 4312 上，与 PR A、PR B、PR C 这条显存线互不依赖。
