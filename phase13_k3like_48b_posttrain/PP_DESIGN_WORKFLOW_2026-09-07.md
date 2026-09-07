# Block AttnRes 的流水线并行：依赖、实现、踩坑与对 torch pipelining 的建议（2026-09-07）

一份把 PP 这条线从头到尾串起来的报告。顺序按讨论的逻辑走：先说 PP / VP 路径对 Block Attention
Residual 和 micro-batch 提出了什么要求；再说我们参考 Reku 公开笔记做的实现、一路撞上的 bug 与它们的
根因（双梯度路径、缓存里存的是什么）；最后落到 `torch.distributed.pipelining` 在 PP 这一维为 AttnRes
缺了哪些接口，给出建议。文末评估现有两张图并列出要补的图。

来源：`REVIEW_ANSWERS_PP_CP_2026-09-04.md`（设计史与 review 回答）、`PP_VP_REEXAMINATION_2026-07-30.md`、
`PP_ATTNRES_ADAPTER.md`、`phase3_attnres_pp_integration/handoff_status_20260421.md`、`PR_BODY_PP.md`、
新树 `torchtitan/models/kimi_k3/pipeline_stage.py` / `layout.py` 的 docstring，以及本周的三个修复。

---

## 0. 一页总览

```
Block AttnRes：每一层都要读"所有更早 block 的表示 + 当前未完成的 partial block"
      │
      ▼  PP 切层之后
要求 1：block 栈必须跟着 hidden 一起跨每一个 stage 边界（后面每个 stage 都要读前面每个 block）
要求 2：最终聚合（output_res_proj → output_res_norm）只能在持有 lm_head 的那个 stage 做
要求 3：栈随深度增长；stage 边界落在 block 中间时，线上传的是一个 partial block
      │
      ├─ 朴素传输：每跳发整栈 → 正确，但每跳字节随 stage 号线性增长
      └─ 跨 stage 缓存（报告 §4.1）：V 个虚拟 stage 共享一个 rank，跳上只发"接收方还没见过的 delta"
                │
                ▼  于是出现三个新依赖
        (a) 发送方和接收方要对"这跳带什么"达成一致 → 离线模拟得到路由表（BlockLayoutTables）
        (b) 缓存按 micro-batch 存 → 需要一个穿越 P2P 仍然稳定的 mb 键（schedule 的 chunk id）
        (c) 缓存里的 block 的梯度要从所有读它的后续 stage 流回提交它的 stage → 双梯度路径
                │
                ▼
        torch pipelining 只有"相邻 stage 之间：输出→下一 stage，梯度←下一 stage"这一种协议，
        缺：多消费者输出、同 rank 就地交付、submodule 可见的 chunk id / 生命周期、稠密 P2P 梯度缓冲、
            forward-only 与"无梯度输出"的契约 → §3 的建议
```

---

## 1. PP / VP 路径对 Block AttnRes 与 micro-batch 的依赖要求

### 1.1 模型侧的三条硬要求

| 要求 | 为什么 | 代码里的落点 |
| --- | --- | --- |
| 栈跨边界 | 每层 attend 的对象是所有更早 block 的表示，切层不改变这一点 | 每跳载荷 `(hidden_TD, delta_TND)`；模型 forward 拿到并返回**整栈**，对传输一无所知 |
| 聚合只在 head stage | `output_res_proj` / `output_res_norm` 是对整栈的一次聚合，属于持有 `lm_head` 的 stage | `kimi_k3_module_fqns_per_model_part` 把聚合模块和 `lm_head` 放在同一段 |
| partial block 上线 | 一个 block 的第一层先把上一个 block 合入栈再算；边界落在 block 中间时，线上是"栈 + 当前 partial" | `KimiK3TransformerBlock.forward` 的 `first_layer_in_block` 分支；stage 边界在 block 起点不需要任何特殊处理 |

Block 大小 12 与 stage 切分之间**没有整除要求**：K3 是 93 层 = 7×12 + 9，最后一个 block 本来就是
partial；debug flavor 特意用 33 层（35 个单元含 embed/head），让任何 pp 形状都切不齐，路由表按实际
切分（一次 all_gather 的 layer→stage 映射）构建，不再有 even-split 前置条件。

### 1.2 两种传输

- **朴素传输**：非末 stage 返回 `(hidden, block_stack)`，torch 原样搬运。正确，无需 adapter；`1F1B` 一 rank
  一 stage 时就是它。代价：stage `S` 发送迄今全部 block，每跳字节 ∝ S（论文 eq.7 的 `C(C-1)/2·N_p·d`）。
- **跨 stage 缓存**（论文 §4.1，Reku 的笔记）：P 个 rank、每 rank V 个虚拟 stage。stage `S` 提交的 block
  在线上只"新鲜"P−1 跳；从 `S+P` 起，每个接收 rank 在它的上一个虚拟 stage `S−P` 已经拿到过。所以每跳只
  发 delta，稳态每跳 = 最近 P−1 个 stage 的提交，**与深度无关**。这是一项省字节的优化，不是正确性组件，
  朴素传输始终是回退路径（`attn_res_cache=False`，同一份路由表的两个取值，也就是结果表里的 A/B）。

### 1.3 micro-batch 带来的依赖

| 依赖 | 内容 | 教训 |
| --- | --- | --- |
| mb 键 | 缓存按 micro-batch 存，forward/backward 都要知道当前是哪个 mb | 用 `id(tensor)` 做键会失效：NCCL 每次给新的接收缓冲，生产者和消费者的 id 永不相等，中间 stage 全部 miss。稳定的键是 schedule 自己的整数 `fwd_chunk_id` / `bwd_chunk_id` |
| 释放时机 | 一个 rank 的 V 个虚拟 stage 共享缓存，不能在第一个虚拟 stage 反向结束时就丢 | 旧 adapter 拖到 `step()` 结束才丢，缓存随在飞 mb 数增长；新版在该 rank **最后一个虚拟 stage 的 forward 之后**释放（那是最后一次读） |
| 数量约束 | `Interleaved1F1B` 要求 micro-batch 数 ≥ 虚拟 stage 数；虚拟 stage 数 = ceil((L+2)/lps) 且要能被 pp 整除 | k3mini 上 lps=2 → 12 段要 ≥12 个 mb；VP=4 要 ≥8 个 mb，所以要在更大的 batch 上比较 |
| 数值 | PP 在 **1 个 mb 时与无 PP 逐位相同**；多个 mb 时把一次 forward 拆成多次累加，求和顺序改变（step 1 约 4e-4 相对 loss、1.3% grad_norm）；**VP 不增加任何差异**（VP=1 与 VP=4 逐位相同） | 早期"透明"的结论只在单 mb 下测过，是平凡情形；诚实的说法是"单 mb 精确，多 mb 差一个 bf16 累加序的量" |

---

## 2. 从 Reku 的思路到我们的实现，以及一路的 bug

### 2.1 参考：Reku 的公开笔记（知乎，Kimi 基础设施工程师）

- 做法："在流水线并行通信之后加一个 adapter，把收到的 block 和 adapter 里缓存的 block 拼起来；反向类似，
  收到所有 block 的梯度，在 adapter 里累加，把累加后的缓冲发给下一个 stage。"
- 交错调度下 send/recv 开销在稳态被隐藏，只有 warmup / cooldown 露出一点通信。
- 代价："跨 stage 缓存改变了梯度的累加顺序，PP 配置一变，调试和精度对齐就变难。"
- 他说的是线上字节的不对称（稳态每跳常量），没有谈缓存内存的不对称；超深模型他推荐的回退是
  selective AC + activation offload，不是分布式缓存。

### 2.2 Phase 3（2026-04）的 adapter：四块缺失的拼图

| 拼图 | 做法 | 备注 |
| --- | --- | --- |
| 谁持有哪些 block、不发元数据 | `BlockLayoutTables`：两端各自离线模拟一个 mb 的 forward，按 stage 列出"提交 / 入口时缓存已有 / 这跳要带"的 block；delta = 累计 − 接收方缓存 | 当时的 layer→stage 映射拿不到全局，代码只验证等分默认值 → even-split 前置条件的来源 |
| 穿越 P2P 的 mb 键 | 包一层 `forward_one_chunk` / `backward_one_chunk`，把 chunk id 放进 thread-local；forward/backward 同线程同步执行，反向里的 hook 能读到 | reviewer 说的 "patch / thread" 之一；根因是 `PipelineStage` 不把 chunk id 交给 submodule |
| 反向 | 见 2.3 | reviewer 说的 "hook / Function" |
| 驱逐 | 打补丁在 `step()` 末从 rank 的最后一个虚拟 stage 丢弃 | 第二个 wrapper |

### 2.3 反向：六次尝试，留下的那个，以及"双梯度路径"

缓存里 block 的梯度必须从每个读过它的后续 stage 回到提交它的 stage。按时间顺序：

1. **在 `autograd.Function.backward` 里自己发 NCCL**（每 block isend/irecv）→ 死锁：autograd 引擎单线程深度
   优先，反向里阻塞的 `wait()` 卡住引擎，对端还没到匹配的 Function；还和 schedule 自己的 `SEND_B/RECV_B`
   在同一进程组上竞争。
2. **`backward_one_chunk` 之后在 autograd 外 flush** → 仍死锁：交错调度下各 rank 到达同一 mb 的反向时刻差
   很大，flush 出去的 P2P 没有对端。
3. **step 末一次性批量交换** → 不死锁，但要把每个 mb 的图保留到 step 末，内存 ∝ 在飞 mb 数。未跑就否。
4. **搭 PP 自己的 `SEND_B`**：收到的 block 保持挂在接收张量的 autograd 图上，梯度沿 schedule 已有的反向
   P2P 逐跳回流，零自定义集合通信（文件 1320 → 784 行）。这就是最终设计的 **通道 A**。它唯一失败的情形：
   rank 在虚拟 stage `v` 提交、在 `v+1` 从自己缓存读回的 block——消费者的反向走进了生产者的 forward 图并把
   它释放，生产者自己的反向随后经 `SEND_B` 到达时报 "backward through the graph a second time"。
5. **每个 stage `retain_graph=True`** → 正确，但 175M 模型 rank 7 多出 5 GiB，且随 V 与 mb 数增长。临时方案。
6. **`_LocalCacheAugment` + `_LocalCacheCapture` 两个 Function** → CPU 金丝雀通过，4 卡仍双反向：逐 Function
   追踪显示 Capture 与 Augment 在同一个 `backward_one_chunk` 里先后触发，autograd 仍从消费者侧走进了生产者图。
7. **缓存存 DETACHED 副本 + `_LocalCacheCapture` + 生产者挂上 tensor grad hook**（04-21）→ 通道 B：消费者的
   Capture 输入没有上游图可走；Capture 的反向把梯度存进 `(mb, producer stage, commit index)` 槽位并返回
   None；生产者自己的反向由 hook 弹出槽位并**相加**。内存回到朴素 PP + 缓存占用（175M PP4×VP2 rank 3：7.71
   对 7.45 GiB）；`expected_same_rank_captures` 的静态计数让 hook 能发现丢失的梯度并拒绝该步。

**"双梯度路径"就是**：跨 rank 走通道 A（PP 自己的反向 P2P，无新代码）；同 rank 走通道 B（一个字典槽位 +
一个 Function + 一个 hook）。**缓存里存的东西**：是 block 的 **detached 副本**（值），不是激活、不是图；
每个 stage 的激活仍只在它自己的 forward 图里，每个 forward 图每 mb 恰好被遍历一次——这正是 4/6 两次失败
的根因（消费者反向沿着挂着图的缓存条目走进了生产者图），detach 是承重的保证，`view` 和 "Function 返回
None" 都不够。

### 2.4 K3 移植（8 月）与 review 分支的子类重写（9 月）

- 布局来自模型 config 而不是模型上的标记属性；传输开关从环境变量变成 `pipeline_kimi_k3` 的参数
  （`functools.partial`，每个 rank 解析一致——环境变量非一致导出曾让各 rank 拓扑不同，集合通信挂死）；
  K3 需要的切分由 `kimi_k3_module_fqns_per_model_part` 自底向上给出，不再注入通用工具；even-split 门槛去掉
  （layer→stage 一次 `all_gather_object`）。
- **`AttnResPipelineStage(PipelineStage)`**（388 行，替代 1228 行的 adapter）把同一协议实现在 stage 自己的
  方法上：`forward_one_chunk` 从 rank 的 `RankStore` + 收到的 delta 组装整栈、跑 stage、留下本 stage 提交
  的 block、只发下一 rank 缺的；`backward_one_chunk` 读组装栈（stage 自己拥有的叶子）的梯度，收到的列作为
  delta 的梯度原路发回，来自 store 的列**存入 store（deposit）**；把 block 带上本 rank 的那个 stage 在自己
  反向前 `_retrieve_recv_grads` 里把 deposits 收进自己的入口梯度，而任何 schedule 都把它排在该 rank 更晚 stage
  的反向之后。**没有 hook、没有 Function、没有 detach 技巧**；路由表给出每个 block 应有的 deposit 数，丢梯度
  会 raise。核心 hook 只有一个：`pipeline_llm(..., stage_class=...)`。
- 数值：irregular debug 模型 2 到 32 个 stage 的每个 pp×vp 格，step 1 与单卡逐位相同（delta 传输与整栈
  传输都是）；老 adapter 的数值在一个 bf16 舍入内复现。

### 2.5 路上的 bug 一览（含本周）

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 中间 stage 全部 cache miss | 用 `id(tensor)` 做 mb 键，NCCL 每次新缓冲 | schedule 的 chunk id |
| NCCL 看门狗超时 | 自定义 P2P 放在 autograd 反向里 / 在 flush 里 | 搭 PP 自己的 `SEND_B`（通道 A） |
| "backward through the graph a second time" | 同 rank 缓存读回的 block 仍挂着生产者的图 | 缓存存 detached 副本；后来的子类版由 store 的 deposit/collect 取代 hook |
| rank 7 多 5 GiB | `retain_graph=True` 的临时方案 | 通道 B |
| 各 rank 拓扑不一致挂死 | 传输开关走环境变量，launcher 非一致导出 | `functools.partial` 参数 |
| 只能等分切层 | 交错调度下本地拿不到全局 layer→stage | 一次 `all_gather_object` |
| `Tensors for P2P must be non-overlapping and dense` | torch 用下一 stage 输入梯度的 stride 分配接收缓冲；以 cat/stack/slice 开头的 stage 得到 view 梯度 | 模型侧 `_DenseGradient`（前向恒等，反向 `.contiguous()`）；上游应改成稠密 `torch.empty(shape)` + 发送前 `.contiguous()` |
| LoRA 下 "no gradient arrived for the payload"（`4716f8ee6`） | 载荷上游没有可训练参数（冻结的 embedding 喂 block 0），下一 stage 没有它的梯度通道 | `_retrieve_recv_grads` 接受 None，deposits 丢弃 |
| verl 的 forward-only log-prob 步 `KeyError: 0`（`cea5c4f0f`） | `schedule.eval` 关掉 backward 后仍调用 `backward_one_chunk`，基类立即返回，子类却去读它从没填的梯度缓存 | 子类同样按 `has_backward` 返回并清掉该 chunk 的簿记 |
| PP 段的 HF 初始加载 `KeyError`（`7d3ba3553`） | adapter 用 layer 1 的张量做 layer 0 占位模板，不持有 layer 1 的 stage 没有 | 只在持有 layer 0 的 rank 合成，缺模板时按 config 形状 |
| 8 卡 PP8 的 "allowed dynamic shared memory" | 硬件/驱动，不是 PP 逻辑 | 换机器 |
| step 10 差百分之几 | Adam 首步 = lr·sign(g)，bf16 舍入让 0.2% 元素翻号并被陡降放大；fp32 总范数不改变 | 只用 step 1 与逐参数梯度做标准 |

---

## 3. `torch.distributed.pipelining` 缺的接口，以及建议

现状：`PipelineStage` 的协议只有"输出给下一 stage、梯度来自下一 stage"（`act_send_info` / `args_recv_info`
是一条链）；它的 `fwd_cache` / `bwd_cache` 只为**本 stage 自己**的反向保存本 mb 的输入输出（3.5 的回答），
既不为同 rank 的后续 stage 保活接收到的激活，也没有"非相邻 stage 消费同一输出"的概念。AttnRes 需要的是
一个**多消费者的跨 stage 边**。建议按价值排序：

1. **多消费者的 stage 输出。** 库应拥有"某个输出被哪些后续 stage 读"的路由表（`BlockLayoutTables` 就是
   这张表，只是算在库外），并按表决定每跳带什么、缓存什么。链式中继 + 沿途缓存是它的一种实现。
2. **同 rank 消费者的就地交付。** 交错调度下消费者常与生产者同 rank；库知道 `stage_index_to_group_rank`，
   同 rank 的输出应按引用交付、梯度由 schedule 合并——这正是通道 B，但不再需要 hook；子类版用 store
   deposit + `_retrieve_recv_grads` 已经证明"由调度顺序保证"是够的。
3. **submodule 可见的 chunk id 与生命周期。** 一个 context 对象（当前 mb），加 mb 结束 / step 结束回调，
   直接删掉两个 wrapper，并支持按 mb 释放。
4. **稠密的反向 P2P 缓冲。** 用 `torch.empty(shape)` 分配接收缓冲、发送前 `.contiguous()`；现在按 stride
   分配的做法在 stage 以 view 类算子开头时必然失败。
5. **forward-only 与"无梯度输出"的契约。** `has_backward=False` 时的 `backward_one_chunk` 行为写进子类
   契约；允许 stage 输出 `requires_grad=False`（对应 None 梯度），而不是假定每个输出都有梯度回来。

可写成一个短 RFC 挂在 4312 的讨论下，措辞沿用 review 里已经答过的 §3.2。

---

## 4. 现有两张图能不能用

| 图 | 内容 | 判断 |
| --- | --- | --- |
| `phase3_attnres_pp_integration/pp_adapter_flow.svg`（含 dark 版） | P=2×V=2 的完整走线：每个 rank 的 `RankLocalCache`、各 stage 的 recv/assemble/emit、`_install_augment_hook`、`_LocalCacheCapture`、`_keepalive_touch` | **前向路由部分可直接用**（stage/rank/缓存/delta 的走法没变）；**反向标注是旧 adapter 的**（hook / Capture / keepalive），要改成子类版：缓存读出的列 → store deposit；提交/带入该 block 的 stage 在 `_retrieve_recv_grads` 收集。建议改成一张"两版对照"或直接更新标签 |
| `Raising_PRs/PR_K3_PARALLELISM/pp_dual_gradient_bridge.svg` | 前向 delta 窗口（block 只在线上新鲜 P−1 跳）+ 反向通道 A/B + 计数（消费者 T−1−S_p，通道 B = V−1−v_p）+ 自检 | **计数与窗口完全有效**（`deposits_expected` 就是它）；同样只需把 "`_LocalCacheCapture.backward` deposits here / `_install_augment_hook`" 改成 "store deposit / collect before own backward" |

建议补的图（对应 §0 的三层）：

1. **依赖图**：栈随 stage 增长、partial block 落在边界、聚合只在 head stage——一张纵向的 stage 序列。
2. **一个 rank 的交错时间线**：V 个虚拟 stage 的 forward/backward 交错，store 里每个 mb 的 put / read /
   release 时刻，以及 deposit 与 collect 落在哪两个反向之间。
3. **协议对照**：torch 的相邻链（输出→下一 stage、梯度←下一 stage）对 AttnRes 的多消费者边，把 §3 的
   五条建议各标一个位置。
