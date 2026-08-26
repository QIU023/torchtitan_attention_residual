# PP 文本侧证据(2026-08-26 重测,adapter 首次真正运行)

判据见 `EVIDENCE_METHOD_2026-08-25.md`。**本文件此前的所有数字已作废两次**:
第一次因为换了上游基点和 batch;第二次因为它们测的是回退路径,不是这个 PR 的核心。

树:`k3_pp_text` @ `e5177bf0e`(已推 fork)。flavor `kimi_k3_debugmodel_text_32l`
(32 层,块大小 16 → 2 块)。全局 batch 4096,微批 256/dp rank。单一 seed,
每格自带预热,每格断言 `Loading the checkpoint from`。**默认组 10 格 + 回退组 7 格,0 挂。**

## 一、为什么此前的 PP 表全部作废

2026-08-25 之前(含老树的 58 格)每一个 PP 格都走**朴素传输** —— 整个块栈随 P2P 逐跳
传递。cross-stage cache adapter 与 `layout.py` 共约 900 行,**从未执行过一次**。

判据不是推断:代码里有一行专为此存在的日志

    # Say so on success, not only on fallback: the adapter is numerically
    # neutral by design, so without this line "wrapped" and "silently fell
    # back" are indistinguishable from the outside.
    logger.info("cross-stage cache adapter wrapped %d stage(s): %s", ...)

它在此前全部 26 个 measure 日志里出现 **0 次**。写这行的人预见到了正是这种情况。

老树同样:`CODE_REVIEW_20_VERDICTS_2026-08-16.md:10-18` 记着 "The 58-cell gate never
enters delta mode ... no matrix script sets `TORCHTITAN_ATTNRES_CACHE`"。差别是老树另有
独立的 delta 对照(该文档记 12 格 delta 与 naive 逐位相等),而 4025 这棵树上一次都没跑起来。

## 二、七处不匹配,同一个根源

adapter 和 `layout.py` 从未针对 4025 的折叠布局适配过。三层门挡着,门后还有四处接口错位。

| # | 假设 | 4025 的实际 | 修复 |
|---|---|---|---|
| 1 | `attn_res_cache` 可从 config 设 | 从未声明;退休 env 兜底后彻底不可达 | `982a9e037` |
| 2 | 默认关 | PP 下 delta 才该是默认,整栈是回退 | `ddc05c911` |
| 3 | `submod.num_blocks` / `layers_per_block` / `config.num_hidden_layers` | 三个都不存在(老树 wrapper 模型的属性) | `16b1c99c8` |
| 4 | 载体走 `blocks=` 关键字 | 第二位置参数 `block_residual_TND` | `7606e3cc7` |
| 5 | token 数 = `shape[0]*shape[1]` | 折叠成 `[T,D]`,乘出来是 T×D = 262144 | `7606e3cc7` |
| 6 | 可以把 adapter 塞进 `model_parts` | 破坏 `_skip_lm_head`,并给 checkpoint key 加前缀 | `7606e3cc7` |
| 7 | 模型返回"本段新增的块" | 返回**累积载体**,新增块是尾部差值 | `7606e3cc7` |

第 3 层通了第 4 层才会暴露,第 4 层通了第 5 层才会暴露 —— 逐层剥。每一层都靠**探针打印
实际形状**定位,不靠读 traceback 猜(traceback 指的是消费者,缺陷在生产者)。

第 6 条的 checkpoint 前缀此前从未暴露:adapter 一旦生效,存出的 key 与不开时对不上。

第 5 条的 262144 直接指认了自己:`256 × 1024 = T × D`。

## 三、主表(分支当前 head,不含 4135)

`mx3_cache_pp_0825_224734`

| cell | 段数 | world | 传输 | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|
| dp1 | - | 1 | - | 12.48548 | 7.92534 | 3.41439 |
| pp2 | 2 | 2 | 回退 | **12.48548** | 7.91227 | 3.35806 |
| pp4 | 4 | 4 | 回退 | **12.48548** | 7.91227 | 3.35881 |
| pp8 | 8 | 8 | 回退 | **12.48548** | 7.90930 | 3.40345 |
| pp2 x vp2 | 4 | 2 | **delta** | **12.48548** | 7.89923 | 3.42837 |
| pp2 x vp4 | 8 | 2 | **delta** | **12.48548** | 7.94984 | 3.39687 |
| pp4 x vp2 | 8 | 4 | **delta** | **12.48548** | 7.93775 | 3.28493 |
| pp4 x vp4 | 16 | 4 | **delta** | **12.48548** | 7.89573 | 3.33794 |
| pp8 x vp2 | 16 | 8 | **delta** | **12.48548** | 7.93965 | 3.25196 |
| pp8 x vp4 | 32 | 8 | **delta** | **12.48548** | 7.91517 | 3.38148 |

**九格 step-1 与 dp1 逐位相同**,4 到 32 段、2 到 8 卡,其中六格由 adapter 驱动。

wrap 确认(每格都查过,零回退警告):stage 归属严格是 Interleaved1F1B 的
`R, R+P, R+2P, ...` —— 例如 pp8 x vp4 是 `[R, R+8, R+16, R+24]`,pp4 x vp2 是 `[R, R+4]`。

**纯 1F1B 不适用**:每 rank 只有一段,没有可复用的 rank 内共享栈,机制本身无从谈起。
所以 pp2/pp4/pp8 走回退不是缺陷,是适用范围。

## 四、delta vs 回退,六对配对

`mx3_naive_pp_0826_001050`,同 batch、同流程,只有 flavor 把传输关掉。

| cell | step 1 | step 2 | step 3 | step 10 |
|---|---|---|---|---|
| pp2 x vp2 | 逐位相同 | 3.5e-4 | 1.6e-3 | 2.1e-2 |
| pp2 x vp4 | 逐位相同 | 3.5e-4 | 5.1e-3 | 1.7e-3 |
| pp4 x vp2 | 逐位相同 | 1.5e-4 | 3.2e-3 | 2.2e-2 |
| pp4 x vp4 | 逐位相同 | 4.9e-4 | 1.7e-3 | 1.9e-2 |
| pp8 x vp2 | 逐位相同 | 5.2e-4 | 3.8e-3 | 4.5e-2 |
| pp8 x vp4 | 逐位相同 | 6.4e-4 | 7.4e-4 | 6.6e-3 |

**前向逐位相同,反向不逐位。** 文件里两句看似矛盾的注释("numerically neutral by
design" 与 "not bitwise against the naive transport")都成立且兼容:送到的块相同,
求和顺序不同。

一条结构性观察:**回退组六格的 s3 只有两个值**(7.90930 / 7.91227),
**adapter 组六格六个不同的值**。前者是 grad-norm 分组的 signature,后者多出的那层
差异来自 delta 打包随拓扑变。

## 五、内存:adapter 确实在省东西

同拓扑、同调度,只差传输方式。per-rank 峰值(GiB):

| pp8 x vp4 | 各 rank 峰值 | 最大 | 极差 |
|---|---|---|---|
| delta | 2.62 x6, **6.57, 6.60** | 6.60 | 3.98 |
| 回退 | 2.66 x6, **7.83, 8.50** | 8.50 | 5.84 |

| pp8 x vp2 | 最大 | 极差 |
|---|---|---|
| delta | 6.03 | 4.05 |
| 回退 | 7.92 | 5.75 |

**峰值降 22–24%,极差收窄三分之一。六个低占用 rank 几乎不动,省的全在两个高占用
rank 上** —— 正是"块栈朝末端累积"被 delta 消掉的形状。这是 wrap 之外、独立于数值的
功能证据:wrap 了但没生效的话内存分布不会变。

方法学限制:`memory:` 那行不带 rank 前缀,所以只用顺序无关的量(峰值集合、最大值、
极差),不声称"哪个 rank"。

## 六、与老树的差异,待解释

老树 `PP_STATUS_2026-08-14.md` 记 **12 格 delta 与 naive 逐位相等**(含反向);
我们测到前向逐位、**反向 s2 在 1.5e-4 ~ 6.4e-4**。两者不一致。

可能的原因:老树那 12 格在 bf16 下恰好舍入到同一个值;或那些配置下 delta 集合与整栈
一致(delta 为空)。**未验证,记为待查**,不作为任何一侧的结论。

## 七、复现

    matrix_scripts/mx3.sh              # 报 step 1/3/10,含 seed 断言与磁盘闸门
    matrix_scripts/pp_cache_matrix.sh  # 默认组 10 格
    matrix_scripts/naive_pp.sh         # 回退组 7 格
    cleanup_scripts/mem_profile.py     # per-rank 峰值内存
