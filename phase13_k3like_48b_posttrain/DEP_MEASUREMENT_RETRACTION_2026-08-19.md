# DEP:之前的结论全部撤回,以及为什么

2026-08-19。这份文档存在的原因是:关于 DEP 气泡隐藏,我在 8-16、8-17 和今天写下过四条不同的
结论,**每一条都是在前提不成立的配置下测出来的**,而每一次我都先给了解释、后才发现前提有问题。
所以这里先列撤回,再列现在唯一站得住的东西。

## 撤回清单

| 写在哪 | 原话 | 为什么错 |
| --- | --- | --- |
| `DEP_BUBBLE_STATUS_2026-08-16.md`、PR23 kit | "binding constraint 是 upfront 前缀" | 从 pp=8 单点推的。pp4/mb64 下前缀只占 4/64,而 56 个 left synchronous |
| 同上 | "placed share ≈ (mb−pp)/mb" | mb 从 16 加到 64,placed 恒为 4,share 反而从 25% 掉到 6.3% |
| 今天口头 | "气泡供给 ≈ O(pp),是机制天花板" | 同一格有 14 个空闲 slot,只用了 4 个 —— 是规划器每 slot 只放一个 |
| 同上 | "要让 DEP 有用需要更差的调度,机制自相矛盾,换大卡也没用" | 建立在上一条之上,一并撤回 |
| `HANDOFF_2026-08-16.md` | "DEP 实验需要 ≥60 GiB 每卡" | 那是**无 AC** 下的结论。所有 flavor 的 `activation_checkpoint=None`;开 full AC 后 `pp2×vp2 seq4096 mb16` 本地就跑得动,内存 76.86% |

## 为什么会连错四次:观测本身不可信

三个独立的观测缺口叠在一起,每一个单独都足以导致错误结论。

**一、`idle_slots` 算了但从不打印。** 于是"调度里没有气泡"和"有气泡但我们没用上"在外部看来是同一件事。
今天补进日志后立刻分辨出是后者。

**二、`dep_cost_ratio.py` 已经跑不动。** `KimiK3Model(kimi_config)` 现在要带 `.kimi_config` 的
wrapper,config 化改动留下的漂移。所以我们一直在用手填的 `KIMI_VIT_BUBBLE_COST_RATIO=0.45`,
**没有人知道真实 r 是多少**。

**三、机制会被静默跳过,而日志读起来像成功。** 这一条最严重,有三个不同的形态:

* `KIMI_VIT_DEP_STAGES` 默认是 1,视觉塔与文本同 stage。`_dep_current_mb` 只在 `submod` 是
  `KimiK3ViTStage` 时设置,所以 `take()` 从不被调用,prefetch 的 `_hits`/`_misses` 恒为 0,
  而报告行只在有 hit/miss 时打 —— 于是它永不出现。我据此以为"计数器不存在",准备去补一个
  **本来就写好的**计数器(`vit_prefetch.py:83-84, 106-124`)。
* `KIMI_VIT_PREFETCH` 与 `KIMI_VIT_BUBBLE` 在代码里互斥,理由是测量诚实性:同时开会让 prefetch
  在计划 slot 到来前喂饱每个 micro-batch,气泡那边照样报"已放置",活是侧流干的 ——
  "a green occupancy number for the wrong mechanism"。这个守卫是对的。
* 加上 `KIMI_VIT_DEP_STAGES=2` 之后,prefetch 被**显式忽略**:
  "the run-ahead has no cross-stage form yet, and KIMI_VIT_DEP_STAGES=2 splits the tower"。

第三条的后果是:**"第二个 micro-batch 的 ViT 前向藏在第一个的文本前向后面"这条路径,在当前代码里
没有任何一种配置能跑通** —— `DEP_STAGES=1` 下取数路径断开,`DEP_STAGES=2` 下机制被关掉。

## 现在站得住的

**规划器的修复是对的。** 去掉"每空闲 slot 只放一个"之后,pp4/mb64 从 4 placed 变 8,
`test_dep_bubble.py` 17 项全过。诊断字段给出 `0 starved / 10 exhausted`,即约束是气泡与消费点的
时序,不是预算。

**backward 侧曾测到 64/64 满**(`0 drained, 0 forced, 64 slot(s) found nothing pending`),
但那是在 `DEP_STAGES=1` 下,**需要在真实 stage 切分下复验**。若成立,它解释了报告为什么敢说
"most of the ViT computation is hidden":反向约占 ViT 计算的三分之二,且它的约束宽得多 ——
forward 必须在消费点之前,backward 只要在梯度到达之后、optimizer step 之前。报告说
"the backward passes are handled analogously",analogous 的是手法,可行域不是。

**报告 5.2.3 是两级串联,不是一件事。** dynamic CP 先按 patch 把单张大图切到多设备、把多张大图
分到 sub-CP 组;DEP 藏的是**这之后的余量**。原文:"This reduces both the encoder latency of large
visual samples and the cross-device load imbalance, allowing the remaining encoder computation to
be hidden in pipeline bubbles."所以"大图长视频时 ViT 负载数倍于文本"是第一级的输入,不是 DEP
面对的 r —— 我此前把两者混为一谈。

## 几何约束(顺带查实)

`pp=4` 下 `layers_per_stage` 只有 2 可用:1 报 "virtual stages (15) must be divisible by
pipeline degree",4 报 "requires at least 2 stages per rank"。15 层几乎没有自由度,要扫 vp
深度得先加一个层数友好的 flavor。这与 multi-commit 那次(16 层 / 块 2 只容一种几何)同类。

## 下一步的正确顺序

先让"这个机制有没有在跑"变成一眼可见的事实,再谈隐藏率:

1. 在 `DEP_STAGES=2`(报告描述的那种切分)下跑通 bubble 路径,复验 backward 是否仍 64/64,
   并取 forward 在真实消费点位置下的 placed/exhausted;
2. 修 `dep_cost_ratio.py`,让 r 是测出来的而不是填出来的;
3. prefetch 的跨 stage 形式是缺口,不是 bug —— 要么实现,要么在文档里承认 forward 只有气泡一条路。

在 1 完成之前,任何关于"DEP 有没有用"的判断都不该写进 PR。

## 更根本的一条:我们实现的机制和报告的不是同一个(2026-08-20)

重读报告后,上面所有测量还有一个共同前提是错的。

**报告把 ViT 摊到所有 PP stage,我们放在一个。** 5.2.3 原文:DEP "splits ViT and text
training into separate stages and **balances vision forward and backward passes across PP
stages**"。Fig 11 的三行 micro-batch 编号是三个 PP rank,而 `ViT fwd` / `ViT bwd` 出现在同一条
Computation 时间线的两端 —— ViT 工作是摊开的,于是**每个 rank 都有 ViT 活可干,每个 rank 的气泡
都能用**。

我们的 `KIMI_VIT_DEP_STAGES` 默认 1,pp=4 时只有一个 rank 持有视觉工作,**另外三个 rank 的气泡对
ViT 不可达** —— 可用气泡池被结构性砍到 1/pp。之前测到的 `10 exhausted` 因此有了第二种解释:不是
"气泡与消费点时序错配"这么单纯,而是"能看到的气泡本来就只有四分之一"。

**结构上是支持的,只是从没跑过。** 实测 `DEP_STAGES` 1/2/4 在 pp=4 上都能建起来,16 步跑满:

    stages=1   roles ['both']                        1 个 rank 有视觉
    stages=2   roles ['head'] ['tail']               2 个
    stages=4   roles ['head'] ['body'] x2 ['tail']   4 个 -- 全覆盖

所以报告那种配置我们建得出来。问题是 `pipeline_adapter.py` 里 `dep_vision_stages() > 1` 那个
**无条件 `return`** 把气泡装配一起挡掉了(注释只谈 prefetch 没有跨 stage 形式,但 return 不区分)。
于是"塔摊开 + 气泡开启"这个组合 —— 也就是报告描述的东西 —— **我们一次都没跑过**。

**第二处偏差:重叠的对象。** Fig 11 显示 `ViT fwd` 压在开头的 `gather param` 上,`ViT bwd` 压在
结尾的 `reduce grad (reduce_scatter + onload + add + offload)` 上。那是 ZeRO 的 DP 集合通信,在步
边界上又长又确定。我们的规划器只认 1F1B 动作表里的 `None` 槽位,**看不到通信流上的窗口**。

合起来:我们实现的是"在单个 rank 的调度空隙里塞 encode",报告做的是"把 ViT 摊到所有 rank,并与步
边界的集合通信重叠"。名字相同,机制不同。这解释了为什么反复调 mb / cost_ratio / vp 都撞墙 ——
调的是错的那台机器上的旋钮。

### 修的顺序(判据先写下来)

1. 收窄那个 `return`,只跳过 prefetch,让气泡在 `DEP_STAGES>1` 下装配;
2. `vision_stage` 从单值改成集合 —— `consume_slot` 现在按 `stage_index != vision_stage` 跳过,
   塔摊开后会认错消费点,单独做第 1 项会得到一个"装上了但认错"的绿数字;
3. 在 `DEP_STAGES=pp` 下测 forward/backward 的 hidden share。**判据事先定死**:backward >= 90%,
   forward >= 25%(现状 12.5%,翻倍才算机制在工作)。判定只看 hidden share,不看 tps ——
   这台机器分辨不了 ±0.5%。

**forward 若达不到 25%,结论就是"forward 的气泡隐藏在我们的实现里不成立"**,而不是再找一层解释。
