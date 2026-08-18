# Overnight 计划,2026-08-19

先说结论:**这一夜 GPU 需要做的事很少**,因为四条主线里有两条卡在别处而不是算力上。所以计划的重心是把"能在无 GPU 情况下做完的对齐工作"做完,GPU 只跑一个真正需要它的验证。

## 四件事的现状(查过,不是回忆)

**① MoE 重构 —— 已完成并推送。** `moe.py` 92 → 53 行,只剩 `gate_up_combine` 覆盖,不再复制 `forward`。上游 `GroupedExperts` 的 `F.silu` 提成了 hook。矩阵 56 真通过 + 2 格环境 OOM(独占复跑各 20/20 证实非本改动)。已在 `dep_exp_impl` 顶端(`630059b71`),无待办。

**② 其他 PR —— 26 个 kit,只有 PR26 filed。** 四轴 kit 是 PR21(EP)/22(TP)/23(PP)/24(CP),都写着 "on top of #4025",与本轮切的 `Raising_PRs/diffs/` 一一对应。**PR24 有实质问题**:它通篇是 Ulysses 叙事(标题 "context parallelism (Ulysses head-sharding)"、"Ulysses head-parallelism, not ring/zigzag"),而本轮把 `kda_cp_mode` 默认改成了 KCP 并验证了 KCP 前向+反向在 cp=2/4/8。**这个 kit 描述的不是我们现在要提的东西。**

**③ 完整项目计划 —— 见下"路线"一节。**

**④ DeepEP/MoonEP —— draft,卡硬件不卡代码。** `moon_ep_dispatcher.py` 已按 `BaseEPTokenDispatcher` 的 ABC 写好(`dispatch`/`combine` 参数名与基类逐字一致,import 可选不破坏 collection),两个方法 `raise NotImplementedError`,因为 MoonEP 要 8×NVLink 而这台没有那个拓扑。**关键复用发现**:torchtitan 已经有这个 seam,DeepEP 已经通过它接进 inference 路径,`deep_ep.*` 已在 `pyproject.toml` 可选导入列表 —— 所以 MoonEP 只需一个子类加一行,不需要"引入依赖"。

## 这一夜做什么

### A. 无 GPU,优先(PR 内容对齐)

1. **重写 PR24 为 KCP 叙事**。现在的 Summary 说 Ulysses 是方案,实际 KCP 是默认、Ulysses 降为 A/B。要改的是:标题、Summary、evidence(换成 cp=2/4/8 的 parity 数字:cp=8 最差梯度误差 1.36e-02,承重情形是 cp>=4 因为那是第一个有中间 rank 的配置)、以及"两者按层同时生效"这个容易被误读的点(KDA 走 KCP、MLA 走 Ulysses,不是二选一)。
2. **PR23 补 DEP 的诚实定位**。kit 里提到 DEP,但本轮首次测出气泡隐藏为负(r=0.493 时 -0.95%,r=2.0 时 -2.08%),且真正的约束是 upfront 前缀而非 cost ratio。PR 里不该暗示隐藏已证明。
3. **四个 diff 与四个 kit 对表**。`diffs/` 是本轮按文件切的,kit 是更早写的;确认每个 kit 的 Scope 段与对应 diff 的文件集一致(PR24 的 Scope 写 "parallelize.py 加 kcp.py",而本轮 CP diff 还含 `vit_cp_plan.py` + `multimodal_model.py` + `moonvit.py` 的 dynamic CP 执行半)。

### B. 一个 GPU 作业(唯一真需要的)

**PR21(EP)的证据在新 MoE 结构下重测。** `moe.py` 从"复制 forward"变成"覆盖 hook"之后,EP 路径的代码变了(现在走基类的 `_grouped_mm` seam),而 PR21 的 evidence 是重构前测的。矩阵 56/58 证明它没坏,但 PR 里的具体数字应该来自现在的代码。跑 `ep2_fsdp2`、`ep2_fsdp2_tp2_cp2`、`ep8_fsdp8` 三格,10 步,取 loss 与 grad_norm。

不跑全矩阵:MoE hook 已经过一轮 58 格,这三格是为 PR 取数,不是回归。

### C. 不做,以及为什么

* **DeepEP/MoonEP 的实现**:要 8×NVLink。draft 已经把接口定死(参数名与 ABC 逐字一致),换机器后要验的两件事写在文件 docstring 里:token 守恒(与 `AllToAllTokenDispatcher` 同输入对比,含某专家收到零 token)、以及 `combine` 的梯度能否回到 `dispatch` 的输入(若 MoonEP 内核不带反向就需要 `autograd.Function`,而只做前向的包装会看起来正确却静默丢专家梯度 —— KCP 的手写 halo 已经付过这个代价)。
* **四轴 PR 的实际提交**:等 4025 合并。它的 11 个新提交含 602 行的 state-dict adapter 重写,现在提等于让四个 PR 穿过别人的 review 周期反复 rebase。
* **DEP 隐藏 / delta 偏置的 shape 依赖性**:两者卡在同一个显存约束上,一台大显存机器同时解开。

## 路线(完整项目计划)

按"什么阻塞什么"排,不按主题:

**现在就能推进(不依赖任何人)**
1. `GroupedExperts` 激活参数化的上游 PR —— 已做成 `gate_up_combine` hook 并验证,可独立提(只碰 `models/common`,gpt_oss 与 4025 都受益)
2. PR26 双侧 —— torchtitan 侧已 filed(4135);pytorch 侧两个 commit 在 fork `get-total-norm-dtype`,PASTE 块待点头
3. PR24/PR23 的叙事对齐(本夜 A 项)

**等 4025 合并**
4. 四轴 PR,PP 和 CP 优先 —— 它们各删掉 4025 里一句 `raise NotImplementedError`,这个叙事比"重构别人的模型"干净得多
5. 索引/命名对齐 —— 查证后确认**不需要**:他们改的是函数内部变量,传给模型 config 的键仍是 1-based 的 HF 值,与我们一致

**等一台大显存机器**
6. DEP 气泡隐藏(需 `mb >> pp` @ seq 4096)
7. delta 偏置是否随 shape 变化(48B carrier 在 8×15.5 GiB 上 seq 1024/512/256 全 OOM)
8. MoonEP 实现与测量(8×NVLink)

**需要一次设计对话**
9. `ShardingConfig` 能否表达 CP/PP —— 它六个 placement 字段说的是"模块参数与 IO 在 mesh 上怎么切",表达不了"这个轴切序列"或"这个模块跨 stage 切开"。这是我们 CP/PP 只能命令式、`parallelize.py` 1656 行对他们 99 行的最大单项来源。该上 RFC 问,答案决定 CP/PP 两个 PR 是按现状提还是重写

**本地技术债**
10. `parallelize.py` 按轴拆成三个文件,让四个 PR 真正零共享文件
11. Config 化收尾(`KimiK3AttnResModel` / `KimiK3MTPLayer` / `KimiK3Spec`)—— 会改 `self.config` 类型而 `register_topology(model.config)` 读它,需要能在尝试间跑矩阵的时段
12. `dep_exp_impl` → fork main 的 ff 合并 + DEP 代码去留(等人定)
