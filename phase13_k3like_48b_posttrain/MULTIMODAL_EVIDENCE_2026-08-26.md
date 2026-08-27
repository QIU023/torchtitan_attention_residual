# 多模态证据:DEP 与动态 CP(2026-08-26)

判据见 `EVIDENCE_METHOD_2026-08-25.md`。三轴组合见 `CROSS_AXIS_EVIDENCE.md`。

树:`k3_on_4025`(上游基点 `30eb5e502`,与三个 PR 分支一致)。
flavor `kimi_k3_debugmodel`(24 层解码器,8 层 ViT)。单一 seed,每格自带预热,
每格断言 `Loading the checkpoint from`。

## 一、这两张表不需要任何 flavor

DEP 只在 `pp > 1` 生效,动态 CP 只在 `cp > 1` 生效。所以**每张表自己的 `dp1` 行就是
关闭侧** —— 与其它每张表用的是同一种关系,不必为"关掉某个特性"专门造 flavor。

原计划不是这样。原计划是拿 `kimi_k3_debugmodel` 对 `kimi_k3_debugmodel_dyncp`
(阈值 256 对 64)。**烟测推翻了它**,见第四节。

## 二、DEP 表(**只测了塔的划分,没测传输**)

> **这张表里 adapter 一次都没 wrap 过。** 三个 PP 格的 wrap 计数是 0,`pp8` 打了
> 8 次 "supports only Interleaved1F1B; running without the adapter" —— 它没设
> `layers-per-stage`,所以每 rank 只有一段,delta 传输全程走回退。
> 它证明的是**塔怎么划分**,不是 DEP 与传输一起工作。后者见第二之二节。

`mx3_mm_dep_0826_052146`,树 `2f9dd3098`,全局 batch 2048,微批 256/dp rank,
每个 PP 格 8 微批。**4 格 0 挂。**

| cell | step 1 | step 2 | step 3 | step 10 | s2 相对 dp1 | 塔的实际形态 |
|---|---|---|---|---|---|---|
| `dp1` | 12.43878 | 9.39110 | 7.80802 | 3.99723 | - | 无流水线 |
| `pp2` | **12.43878** | 9.40845 | 7.80171 | 3.99293 | 1.85e-3 | 一段,`roles []` |
| `pp4` | **12.43878** | 9.40845 | 7.80171 | 3.99293 | 1.85e-3 | `head` + `tail` |
| `pp8` | **12.43878** | 9.40845 | 7.80171 | 3.99095 | 1.85e-3 | `head` + `tail` |

三个 PP 格 step 1 与 dp1 逐位相同,s2 相对 dp1 都是 1.85e-3。

**`pp2` 与 `pp4` 在四个步上逐位相同,而前者塔占一段、后者塔切两段。**
塔跨段是纯调度改动,端到端确证了 `test_vit_stage_shares` 在塔本身算术上钉的
那条 bitwise 断言。`pp8` 只在 s10 上偏离(3.99095 vs 3.99293)。

### `pp2` 那一行现在不可达

这张表跑的时候,`vit_dep_stages` 放不下会**缩到放得下**;`pp2` 只有两段,塔的默认
份额 2 缩成了 1,于是给出了 clause 1 的对照。

之后按"显式失败不静默降级"改成了**报错**(`77a298ac5`),所以默认配置下这一行
不再产生。段数账:

| 形态 | 塔 | 文本 | 总段数 | 最小 pp |
|---|---|---|---|---|
| clause 1(塔独占一段) | 1 | >=1 | >=2 | pp2 |
| clause 2(默认,塔切两段) | 2 | >=1 | >=**3** | **pp4** |

卡在 schedule 自己的断言 `num_stages % pp_degree == 0` 上:3 段要 pp3,而 pp 又得
整除 world size。所以默认形态的最小可用配置是 **pp4**(4 段 = 2 塔 + 2 文本),
或 **pp2 x vp2**(同样 4 段)。

**贴 PR 时用 dp1 / pp4 / pp8 三行**;`pp2` 这行留在本文件作为"切分数值中性"的旁证,
并注明它是在缩段语义下测的。

## 二之二、DEP 与 delta 传输同时打开

`mx3_dep_cache_0826_221314`,树 `0242649e8`,同 batch,每格 8 微批。**5 格 0 挂。**

在 `0242649e8` 之前**这四个组合一个都跑不起来**。`pp8 x vp4` 死在:

    ValueError: layer 0 sits on stage 2, but the contiguous layout this adapter
    assumes puts it on stage 0 (n_layers=24, num_stages=24)

DEP 把塔的 stage 插在所有解码层前面,层 0 因此从 stage `n_vit` 开始;而
`layers-per-stage 1` 下 24 层要装进 22 个文本段,**段还不均匀**。两件事都不符合
"层 ell 在 stage ell // layers_per_stage"。守卫拒绝得对 —— 按错的布局路由块 delta,
只有梯度里看得见。它缺的不是宽容,是真实布局。

| cell | 段数(塔+文本) | step 1 | step 2 | step 3 | step 10 | s1 对 dp1 | s2 相对 |
|---|---|---|---|---|---|---|---|
| `dp1` | - | 12.43878 | 9.39110 | 7.80802 | 3.99723 | - | - |
| `pp2 x vp2` | 2+2 | **12.43878** | 9.40124 | 7.83331 | 4.00067 | 逐位相同 | 1.08e-3 |
| `pp2 x vp4` | 2+6 | **12.43878** | 9.39993 | 7.73236 | 4.01525 | 逐位相同 | 9.40e-4 |
| `pp4 x vp2` | 2+6 | **12.43878** | 9.39841 | 7.85824 | 3.98119 | 逐位相同 | 7.78e-4 |
| `pp8 x vp4` | 2+22(**不均匀**) | **12.43878** | 9.40560 | 7.76881 | 3.99359 | 逐位相同 | 1.54e-3 |

**四格前向逐位相同。** 层到 stage 的映射只要错一处,delta 就送到别的 stage,前向立刻
就不一样了 —— 包括 `pp8 x vp4` 那个不均匀的切分,正是原来的连续假设最不可能覆盖的。

s2 相对 dp1 是 7.8e-4 ~ 1.5e-3,与 PP 文本表 delta 组同量级(那边开 cache 的 vp 格
是 5.1e-5 ~ 7.0e-4,不开 vp 的 pp2 就有 3.5e-4)。反向不逐位是预期:送到的块相同,
求和顺序不同。

**wrap 计数确认跳过塔段生效**:rank 0 持 `[0, 8, 16]` 三段只包 2 段(stage 0 是塔
head),rank 2 持 `[2, 10, 18]` 三段全包。零回退警告。

修复的三处(`0242649e8`):布局从调用方给的全局 `layer_to_stage` 读,不再假设连续或
均匀;该映射由 `module_fqns_per_model_part` 得到 —— 原注释说"没有 collective 拿不到
全局映射",其实不需要,调用方自己就知道它要的切分;wrap 循环跳过塔段,并把链头判断
从 `stage_id == 0` 改成第一个**文本**段。守卫没有放松,反而变强:映射来自模块名,
校验来自 split 后活下来的模块,两个独立来源互相对账。

## 三、动态 CP 表

`mx3_mm_dyncp_0826_044433`,树 `2f9dd3098`,全局 batch 4096,微批 512/dp rank,
seq 512。**3 格 0 挂。**

| cell | step 1 | step 2 | step 3 | step 10 | s2 相对 dp1 | 实际切分 |
|---|---|---|---|---|---|---|
| `dp1` | 12.45537 | 9.46334 | 7.89917 | 4.01159 | - | 无(cp=1,不打印) |
| `cp2` | 12.45676 | 9.46086 | 7.64048 | 3.99404 | **2.62e-4** | 1 张大图/1 张,1 组 x 2 rank |
| `cp4` | 12.46410 | 9.42999 | 7.91165 | 4.07461 | **3.52e-3** | 1 张大图/1 张,1 组 x 4 rank |

两个 CP 格都确认**真的进了图内切分**,判据是

    Dynamic CP: %d large image(s) of %d over %d sub-CP group(s) of %d rank(s); min_patches=%d

不是从数值推断。与 PP 那次 adapter 的 wrap 日志同一个作用。

本表只有两个 CP 度数,而 cp2 -> cp4 偏差涨了 13 倍(2.62e-4 -> 3.52e-3)。
两个度数分不出趋势和单点,所以**度数跑到头**,见下一节。

## 三之二、动态 CP 跑到 cp8:没有趋势

`mx3_mm_dyncp8_0826_142531`,树 `77a298ac5`,全局 batch 4096,微批 1024/dp rank,
**seq 1024**。**4 格 0 挂。**

cp8 需要 `Q_LEN % (cp * 128) == 0`,所以 seq 必须抬到 1024。那使它成为**另一张表、
自带 dp1**,不能作为上一节的第四行 —— 一张表 = 一次 run,一个长度。

| cell | step 1 | step 2 | step 3 | step 10 | s2 相对 dp1 | 实际切分 |
|---|---|---|---|---|---|---|
| `dp1` | 12.46687 | 10.22638 | 7.53078 | 3.55408 | - | 无(cp=1,不打印) |
| `cp2` | 12.45999 | 10.12949 | 7.34380 | 3.57777 | 9.47e-3 | 1 组 x 2 rank |
| `cp4` | 12.49556 | 10.12967 | 7.52961 | 3.58007 | 9.46e-3 | 1 组 x 4 rank |
| `cp8` | 12.47423 | 10.20364 | 7.45319 | 3.49991 | **2.22e-3** | 1 组 x 8 rank |

**cp2 与 cp4 几乎完全相同(9.47e-3 对 9.46e-3),cp8 反而更小。** 三个度数上没有
单调上升,与文本侧 CP 从 cp4 到 cp8 饱和同形。

**上一节那个 13 倍是单点,不是趋势。** 这是 `EP_TEXT_EVIDENCE.md` 第二节那条教训的
反面用法:那次是"低度数的零变化不能外推成无影响"(ep2/ep4 不动,ep8 动了 1.9e-3),
这次是"低度数的上升不能外推成随度数恶化"。**两次都是度数跑到头才知道的。**

三个 CP 格都确认真的进了图内切分,判据同上,`sub-CP group` 的 rank 数随度数走
(2 / 4 / 8)。

与 PP 不同,**CP 的 step 1 不逐位相同**:CP 把序列切开,attention 的规约顺序随之改变,
前向就有差异。PP 只切层,前向的算术不变,所以那边九个格 step 1 与 dp1 逐位相同。

## 四、被推翻的前提:`_dyncp` flavor 是个 no-op

`kimi_k3_debugmodel_dyncp` 的 docstring 曾声称:

> debug dataloader 把图封顶在 256 patch,也就是生产阈值,所以默认下没有图够大,
> 路径是接好的但从不执行;这个 flavor 把阈值降到 64,矩阵才真的跑它。

实测两组:

    min_patches=256   Dynamic CP: 1 large image(s) of 1 over 1 sub-CP group(s) of 2 rank(s)
    min_patches=64    Dynamic CP: 1 large image(s) of 1 over 1 sub-CP group(s) of 2 rank(s)

除阈值数字外两行完全相同。`classify()` 是 `c >= min_patches`,那张图既 >=64 也
>=256,分组结果一模一样 —— **默认配置下动态 CP 本来就是活的**,64 什么也没改变。

如果按原计划直接铺矩阵,两张表会给出几乎相同的数字,而这很容易被读成
"动态 CP 数值中性" —— 一个完全错误、却看起来很可信的结论。

**"先单格烟测确认代码路径真的走到再铺矩阵"第二次救场**(第一次是 PP 的 adapter)。
该 flavor 已删除。

## 五、"默认打开"挖出的三个 bug

DEP 从 opt-in 改成默认打开(`ca641f4c8`)之后,三条此前**永远不进**的路径暴露了:

| # | 症状 | 后果 |
|---|---|---|
| 1 | wrapper 布局的死分支 | 给纯文本模型发匹配不到 child 的 vision 段 |
| 2 | 只要 DEP 开就扣一段文本预算 | 不管模型有没有塔 |
| 3 | `hasattr(model, "vision_encoder")` | 文本模型该属性**存在且为 None**,gate 通过 |

第 3 个最值得记。后果是纯文本 pp2 被切成

    stage 0: tok_embeddings
    stage 1: layers.0-23, norm, output

而不是两段各 12 层。**它能跑,loss 看着正常,流水线是错的,run 里没有任何东西会说。**
唯一的可见痕迹是"塔缩段"的 warning 出现在一个根本没有塔的 run 里 —— 烟测里那格
`text_pp2` 盯的正是这个。

`_holds_vision_tower` 早就用 `getattr(...) is not None` 且 docstring 记着这个坑,
FQN 注入那边没有。已补 CPU 单测钉住:撤掉修复它红,装回去绿。`_Model` 是**不存在**
该属性,真实模型是**存在且为 None**,新的 `_TextModel` 专测后一种形状。

## 六、复现

    matrix_scripts/mx3.sh             # 报 step 1/3/10,含 seed 断言与磁盘闸门
    matrix_scripts/dep_matrix.sh      # DEP 表
    matrix_scripts/dyncp_matrix.sh    # 动态 CP seq512 表
    matrix_scripts/dyncp8_matrix.sh   # 动态 CP seq1024 表(含 cp8)

`--comm.init-timeout-seconds 3600` 是这批表比早先表多的一个 flag:冷 KDA/tilelang
编译要七分多钟,超过 NCCL 默认 300 秒 watchdog,会在编译中途把 pp4 基线格打死。
它只抬高那个上限,不进任何数值路径。

## 七、气泡隐藏与 run-ahead:时间维度(2026-08-27)

树 `a0a14b63b`。逐步 wall-time 由日志时间戳计算(8-12 步取每步最小消抖),
比整数 tps 细两个数量级。同拓扑 pp4 + DEP,12 步。

### 7.1 bubble:调度层完美,debug 规模时间上是净亏

| 臂 | 每步中位 | 相对 | 调度 |
|---|---|---|---|
| off(塔反向内联) | 2.879 s | - | - |
| bubble | 3.000 s | **+4.2%** | **8/8 计划槽,0 兜底,0 强制** |

debug 塔只有 8 层 ViT,pp4 x 8mb 的空隙本来就小,而 deferred 机制的固定开销
(detach/hook/重放、每步重建 plan)是真实的 —— **塔太小,省的没有花的多**。

**边界声明**:本机只能证明"调度 100% + 数值中性 + 机制活着";
"隐藏在 wall-clock 上为正收益"在 debug 规模**不可证**(实测 -4%),它的经济性
要到塔占比真实的规模(2.8T)才可能反转,需要 profiler 级空闲区间占用证据。
`vit_bubble` 保持 opt-in 是正确默认。

### 7.2 run-ahead(vit_prefetch):第三个"有 writer 无 reader",修复后首跑

历史上从未执行过。两层无声缺陷,与 bubble 队列同族:

1. **消费缝没接**:`_vision_prefetcher` 有 writer 无 reader,`ensure()` 在前向
   路径上没有调用者。模型侧 `encode_images` / `_issue_on_vision_stream` /
   `_join_vision_stream` 都在,只缺老树 `multimodal_model.py:1426` 那一读。
   已从老树移植(`d49400544`):encode 前 `take(mb)`,取到就跳过内联 encode,
   服务完 `advance(mb, depth)` 发起后续。
2. **确认日志钉在错的类型上**:"prefetch installed" 只对 `KimiK3ViTStage` 打,
   而塔占一段(run-ahead 唯一支持的形态)时折叠布局从不构造该类 —— 恰好在
   它要确认的那种 run 上沉默。改为 `_holds_vision_tower`。
3. 首跑暴露第三层:`ensure()` 按老树约定把 features 包成**每图一项的 list**,
   折叠布局的 splice 要拼接流 —— `'list' object has no attribute 'to'`。
   消费端补上老树同款 `torch.cat`(`a0a14b63b`)。

修复后 A/B(都在 `vit_dep_stages=1`,run-ahead 的唯一合法形态;塔切分下它
被守卫显式忽略并告警,该守卫工作正常):

| 臂 | 每步中位 | s2 loss |
|---|---|---|
| off | 2.751 s | 9.98687 |
| prefetch depth=1 | 2.770 s(+0.7%,噪声级) | **9.98687(逐位相同)** |

`DEP vision prefetch installed: depth=1 on stage 0` 首次出现在日志里。

### 7.3 本周同类缺陷清单(全部已修)

| 机构 | writer | 缺的 reader | 修复 |
|---|---|---|---|
| cache adapter | 配置/marker | 七处布局适配 | `7606e3cc7` 等 |
| bubble 队列 | `_vision_grad_queue` | `cut_for_deferred_backward` 调用点 | `e973802e6` |
| prefetcher | `_vision_prefetcher` | `take()`/`advance()` | `d49400544` + `a0a14b63b` |

同一个移植断层:老树的消费端都在 `multimodal_model.py`,折叠布局没有该文件,
机构随移植过来、缝没接上,且全部无声。**判据永远是那行成功日志,不是"跑完没报错"。**
