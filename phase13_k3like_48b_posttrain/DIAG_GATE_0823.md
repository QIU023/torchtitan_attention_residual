# 30 格门的两类问题定位(2026-08-23)

门:`/workspace/gate_notp_0823_211610`,19 过 11 挂。冻结副本 `/workspace/tt_frozen2`。

## 一、"数值不对齐"的主体是测量假象,不是并行实现

同一个格、同一个 seed ckpt、同样 `--debug.seed 42 --debug.deterministic`,连跑三次:

| 运行 | step-1 loss |
|---|---|
| dp1_rep1 | 12.59459 |
| dp1_rep2 | 12.58783 |
| dp1_rep3 | 12.58783 |
| pp2_rep1 | 12.58810 |
| pp2_rep2 | 12.58783 |

机制确认:换全新 `TORCHINDUCTOR_CACHE_DIR` 重跑 dp1,精确复现 **12.59459**;
暖 cache 稳定 **12.58783**。两个值各自可复现 —— 是 inductor 编译缓存的冷热差,
不是随机性。

**单格自身的冷热差 6.8e-3,大于我之前在追的跨格差**(fsdp2 2.3e-3、cp2 9.3e-3)。
门把所有格串行写进同一个 cache 目录,谁先编译某个图形状谁付冷启动代价,所以:

* `text_pp2` = 12.52035 异常,而 `text_pp4/pp8` 逐位相同 —— pp2 是门里**第一个** PP 格;
* `mm_pp2/pp4/pp8` 全部逐位相同 —— 轮到 mm 臂时 PP 的图已经被 text 臂烤热。

这条根因能解释所有没出问题的格。

**判据受影响,必须告知**:"跨格 step-1 loss 相同"在当前 harness 下无法与编译缓存
状态分离。门要改成每格先跑一次丢弃的预热、再跑计量的那次(或每格独立 cache 且各跑两次取第二次)。

### 与老树对比:新树更紧,不是回归

| 格 | 老树 Δ vs dp1 | 新树 Δ vs dp1 |
|---|---|---|
| fsdp2 | 3.56e-3 | 2.27e-3 |
| cp2 | 1.57e-2 | 9.27e-3 |

(老树数据:`MATRIX_18_SDPA_2026-08-09.md`。)老树同样从来没有跨格逐位相同。

### 数据侧的独立事实

上游 `components/data/packing.py:91` 自带 TODO:
`Consider packing before DP sharding so ranks receive similarly filled rows.`
打包用 `FirstFitPackIterDataset`,发生在 DP 分片**之后**。实测 dp1 与 dp2:
real_label_tokens 都是 2048、fill 都是 1.0000,但 input_checksum 219115 vs 220114
—— **token 总数相同、token 内容不同**。所以改变 dp 度的格本来就在不同数据上训练。
这是上游数据管线的性质,不是迁移引入的。CP 不在 batch 轴
(`"batch": dp_replicate * dp_shard`),所以 cp 格的数据与 dp1 相同。

## 二、mm 8 卡挂:确定性复现,是迁移回归

`mm_fsdp2_pp2_cp2` / `mm_ep2_fsdp2_pp2_cp2`。先前判断的"端口 TIME_WAIT 残留"**是错的** ——
text 臂同拓扑 8 卡通过、`mm_pp8`/`mm_ep8_fsdp8` 也通过,残留 socket 不会挑臂。
单独重跑确定性复现;栈顶是 `ProcessGroupNCCL::recv -> initNCCLComm ->
broadcastUniqueNCCLID`,即 PP 的点对点 recv 在建通信子而对端没有发出配对的 send。
去掉 FSDP 只留 `mm_pp2_cp2`(4 卡)同样挂 —— 与 dp 无关,是 PP+CP+视觉塔。

老树的 `mm_fsdp2_pp2_cp2` 是通过的(三跑逐位相同 12.04848),所以这是迁移回归。

老树 `HANDOFF_CP_HANG_2026-08-04.md` 记录过同一类故障,结论是:
**集合通信的进入条件必须由 mesh 决定,不能由数据决定。**

对照下来,老树写好的两处我只搬了一半:

| | 老树 `multimodal_model.py` | 新树 `model.py` |
|---|---|---|
| `_exchange_sentinel_counts` | line 983,在 `if pixel_values is None` **之前**,`if cp_active` 门控 | line 800,在早退(765-778)**之后** |
| PP 元组容错 | `_keep_tower_alive`(736-752) | 直接调 `add_zero_valued_dependency` |

老树该处注释逐字描述了现在的现象:

    the decision to exchange them must not depend on data. A batch carrying
    no images at all is a normal occurrence, and letting that rank return
    early leaves its CP peers waiting in the collective forever -- a
    100-second NCCL watchdog timeout, not an error. Gate on the mesh.

新树的早退分支只处理了塔自身的 FSDP all-gather,没处理 sentinel 交换这条集合通信。
观察到的栈在 PP recv 而非 all_reduce,与此一致:卡在 sentinel 交换里的 rank 永远
产不出 stage 输出,下游 stage 的 recv 因而阻塞 —— 我们看到的是受害者,不是肇事者。

### 实测定位到的真正成因(2026-08-24)

先前两条推断都被实测推翻,记录下来:

* "端口 TIME_WAIT 残留" —— 错。text 同拓扑 8 卡通过,残留 socket 不挑臂。
* "`mm_pp2_cp2`(4 卡)同样挂,所以与 dp 无关" —— 错。那次是我自己 `pkill` 杀的
  (SIGTERM 时间戳与我执行 pkill 的时刻一致)。
* "sentinel 交换没做 mesh 门控导致挂起" —— 机制不成立:`_prepare_multimodal_embeds`
  只在 `tok_embeddings is not None` 时进入,即只有 stage 0;同一 stage 内两个 CP rank
  看到同一个 microbatch,对称。

NCCL watchdog 给出了事实:

    Rank 0/1: Last enqueued NCCL work: 8, last completed: 7
    Rank 2/3: Last enqueued NCCL work: 7, last completed: 7
    OpType=ALLGATHER, NumelIn=32768, NumelOut=65536, PG GUID 0(default_pg)

pp2 x cp2 下 rank 0/1 是持有 `vision_encoder` 的 stage 0,rank 2/3 是 stage 1。
**只有持塔的 stage 发起了这个 ALLGATHER,而它发在跨 stage 的 default_pg 上。**

成因在 `model.py` 的 `_encode_images`(dynamic CP 路径):

    parts = [torch.empty_like(mine) for _ in range(cp_size)]
    dist.all_gather(parts, mine.contiguous())        # 没有 group=

分配 `cp_size`(2)个输出却发在 default_pg 上 —— 与观测到的 NumelOut = 2 x NumelIn
完全吻合。该文件其余每一处集合通信都显式传了 `group=`。

**能解释所有没失败的格**:

| 格 | 是否进入 dynamic CP 分区 | 结果 | 原因 |
|---|---|---|---|
| `mm_cp2`(2 卡) | 是(1 of 1 image across 2) | 过 | default_pg 恰好等于 CP 组,漏传无害 |
| `mm_fsdp2` / `mm_pp8` / `mm_ep8_fsdp8` | 否 | 过 | 无 CP,提前返回 |
| text 各格 | 否 | 过 | 无视觉塔 |
| `mm_*_pp*_cp2`(4/8 卡) | 是 | 挂 | default_pg 跨 stage |

顺带更正:该处代码注释称"debug 数据下每张图都低于阈值,路径接通但不触发" ——
实测**触发了**(失败格与 `mm_cp2` 日志里都有 `Dynamic CP partitioning`)。

### 已改(对齐老树的两条约束)

老树的 `_encode_images_cp_sharded` 是按图轮转分片,与新树按 band 切分的 dynamic CP
是不同算法,不能照抄;但它给出的两条约束正是被违反的 ——
显式传 `group=`,以及用**可微**的 all-gather(`funcol.all_gather_tensor(..., group=group)`,
老树注释:"its transpose is the reduce-scatter that hands each rank the gradient")。
新树自身已在用 `dist_nn`,对齐为:

    parts = dist_nn.all_gather(mine.contiguous(), group=group)

原写法 `dist.all_gather` 既漏 group **又不可微**,视觉塔 CP 分支拿不到梯度。

### 验证

同一格从 363 秒 NCCL watchdog 超时变为 1 分钟内快速失败在**另一个**断言上,
说明塔的 all-gather 已能完成、执行推进到了解码器:

| 格 | 修复前 | 修复后 |
|---|---|---|
| `pp2_cp2_dp1`(4 卡) | NCCL 死锁 | AssertionError: attention_masks |
| `fsdp2_cp2`(4 卡) | - | SIGABRT |
| `fsdp2_pp2`(4 卡) | - | **过** 12.58270 |
| `fsdp2_pp2_cp2`(8 卡) | NCCL 死锁 | AssertionError: attention_masks |

**下一个待查**:`attention_masks must be instance of BlockMask, got NoneType`。
只在 CP 与第二个轴同时在场时出现;`mm_cp2`(单独 cp2,2 卡)通过。未定位。

## 三、LoRA 臂 0/10

全部 `Missing key in checkpoint state_dict: layers.11.attention.wkv_b.lora_a`。
成因已确认:DCP 的 planner 要求 ckpt 里存在每一个 key,比
`load_state_dict(strict=False)` 严格,所以稠密 seed 喂不了 LoRA 臂。
已在门脚本里改为稠密臂与 LoRA 臂各自一个 seed(`seed_dense` / `seed_lora`),未重跑。

## 四、状态

| 项 | 状态 |
|---|---|
| 数值散布根因 | 已定位并实测确认(编译缓存冷热) |
| 门的方法学 | **需要改**:每格预热后再计量 |
| mm 8 卡挂 | 已定位到两处与老树的偏离,未修 |
| LoRA seed | 脚本已改,未重跑 |
| 与老树对比 | 新树数值更紧;mm+PP+CP 是回归 |

---

## 五、按老树重搬 dynamic CP(2026-08-24)

用户要求:遇到 block/miss 只允许从老树看,不允许自写。据此重搬。

### 搬了什么

| 项 | 来源 | 说明 |
|---|---|---|
| `_build_cp_subgroups` | 老树 `parallelize.py:570` | 逐字搬,接到 `apply_cp_kimi_k3`,设 `model._cp_subgroups` |
| `_encode_images` 主体 | 老树 `_encode_images_dynamic_cp` | 结构照搬 |
| `_PlainGradBoundary` | 老树 `multimodal_model.py:122` | 逐字搬 |

原来自写的 driver 缺的不只是性能,还有正确性要件:

* `subgroup_layout` / `balance_images` —— 报告 §5.2.3 的后半(子 CP 组 + 负载均衡),
  目的是 "preventing the communication fraction from growing with scale"。
  这两个函数**本来就在新树的 `vit_cp_plan.py` 里**(整文件搬过),我没调用,是死代码。
* `n_passes = max(per_sub)` + 空 pass 补齐 —— 各子组必须跑相同次数,否则子组间集合通信错位。
* bands 非递增断言 —— 保证 padding 落在尾部 rank,前缀截取才成立。
* `merged_tokens(h,w,kh,kw)` 截取 —— 不是 `counts//merge`,投影器折叠时间维。
* `funcol.all_gather_tensor(..., group=group)` —— 组是**子组**,且可微。

删掉了我自造的 `make_cp_patch_plan`(老树 0 处),搬完后无人引用。

### 校对:老树有而新树没接的其余部分

| 文件 | 死代码 | 判定 |
|---|---|---|
| `vit_cp_plan.py` | `stage_*` / `pack_*` / `unpack_*` | 老树 DEP 用它们按**静态形状**跨 stage 传 patch;新树 DEP 把塔放在 stage 0,patch 不跨边界。**架构差异,待跑 DEP 确认** |
| `sharding.py` | TP 声明族 | TP 暂停 |
| `vision_encoder.py` | 老树自己的 ViT 块 | 新树继承上游 `MoonViTEncoder` |

### 搬运过程中暴露的自写偏差

`vision_encoder.forward` 覆写从**传入的 grid** 算位置编码;老树塔用的是
**`cp_plan.full_grid`**(老树 `vision_encoder.py:286-289`)。之前"能跑"只是因为我的
driver 恰好传的就是完整 grid;改成老树约定(传分片 grid `[[t, band, w]]`)后立刻
`size of tensor a (128) must match b (0)`。已按老树改为读 `cp_plan.full_grid`。

### 当前 blocker:AC 与 CP plan 状态

修完之后 CP 各格统一停在:

    AssertionError: attention_masks must be instance of BlockMask, got NoneType
    (backward, torch/utils/checkpoint.py unpack_hook -> vision_encoder.py super().forward)

是**反向重算**触发的。`set_cp_patch_plan(plan)` + `finally: set_cp_patch_plan(None)`
在前向结束即清空,AC 重算时塔看不到 plan,回退到上游 `VisionAttention` 并断言。

**老树用的是完全相同的 try/finally**,它不触发是因为**老树不对视觉塔做 AC** ——
老树只 "Applied activation checkpointing to KimiK3TransformerBlock stack"。
而新树的 `ac_policy.apply(model.vision_encoder)` 来自**上游 4025 自己的 commit
`f925dad99 "fix activation checkpointing"`**,不是我加的。

所以这是上游 AC 选择与老树 CP 设计的交互,老树里不存在这个情形。
从老树能得到的唯一答案是"不要对塔做 AC"。**未自行改动,等指示。**

验证状态:`fsdp2_pp2` 12.58270 通过(不涉及视觉 CP,搬运未影响);
5 个 CP 格全部停在上述断言。

---

## 六、dynamic CP 与 activation checkpointing 不兼容(老树潜伏缺陷,已修)

### 为什么现在才出现

| | flavor 里的 AC |
|---|---|
| 老树 `config_registry.py` | `activation_checkpoint=None` |
| 新树(上游 4025) | `activation_checkpoint=SelectiveAC.Config()` |

门本身不传 AC,继承 flavor 默认。**老树 58 格从未在 AC 打开的情况下跑过 CP。**

新树的默认来自上游 commit `f925dad99 "fix activation checkpointing"`
(JavaZeroo, 2026-08-19),它在实现 AC 支持的同时把 debug flavor 的默认从 `None`
翻成了 `SelectiveAC`,并新增 `ac_policy.apply(model.vision_encoder)` ——
即**把 AC 也加到视觉塔上**。

### 缺陷

`SelectiveAC.apply()` 用 `ptd_checkpoint_wrapper` 包住 `layers` 的每个子模块;
重算时用**保存的入参**重放 forward。而 plan 是通过模块状态传的:

    self.set_cp_patch_plan(cp_plan)     # 写 block.attn._cp_plan
    try:  ...blocks...
    finally: self.set_cp_patch_plan(None)

前向结束即清空,反向重算时塔读到 `None`,于是走回上游 `VisionAttention`,
撞上 `attention_masks must be instance of BlockMask, got NoneType`。

**老树是完全相同的写法**(`vision_encoder.py:733/800/808`),只是 AC 从没打开,
所以这是老树的潜伏缺陷,被上游翻默认值后暴露。且它不能靠"不清空"绕过:
`n_passes` 循环里每个 pass 用不同的 plan,块是复用的,留着上一个 plan 会让
重算用错 plan —— 那是静默的错误结果,比崩溃更糟。

### 修法:plan 走入参,不走模块状态

checkpoint 保存并重放调用的**参数**,所以只要 plan 是参数就天然正确。
`VisionTransformerBlock.forward` 本来就把 `attention_mask` 透传给 attention,
沿同一条路加一个可选的 `cp_plan`:

* `models/common/vision_encoder.py`:`VisionAttention.forward` 与
  `VisionTransformerBlock.forward` 各加 `cp_plan: object | None = None`,
  基类忽略它、块透传给 attn。**加性改动,其他模型不传即行为不变。**
* `kimi_k3/vision_encoder.py`:`KimiK3VisionCPAttention.forward` 从参数取 plan;
  删掉 `_cp_plan` 状态槽、`set_cp_patch_plan()` 和 try/finally。

模块状态整个消失,AC 的问题随之消失。

### 验证:AC 开与关逐位相同

| 格 | AC 开 | AC 关 |
|---|---|---|
| `cp2`(2 卡) | 12.60396 | 12.60396 |
| `pp2_cp2_dp1`(4 卡) | 12.60396 | 12.60396 |
| `fsdp2_cp2`(4 卡) | 12.60754 | 12.60754 |
| `fsdp2_pp2`(4 卡) | 12.58270 | 12.58270 |
| `fsdp2_pp2_cp2`(8 卡) | 12.60754 | 12.60754 |

AC 本就应当不改数值,现在确实不改。另外两对 PP/非 PP 也逐位相同
(`cp2` = `pp2_cp2_dp1`,`fsdp2_cp2` = `fsdp2_pp2_cp2`)。

**待办**:同样的修法要移植回老树 —— 老树块的签名是 `block(x_LD, seq_bounds, freqs_cis)`,
需要沿它自己的透传链加 `cp_plan`。

---

## 七、DEP:多 stage 那一半还没接(更正第五节的判断)

第五节把 `vit_cp_plan.py` 里 `stage_*` / `pack_*` 一组死代码判为"架构差异",**偏乐观了**。

报告 §5.2.3 "Encoder computation in PP bubbles" 的 DEP 有两半:

1. "splits ViT and text training into separate stages"
2. "**balances vision forward and backward passes across PP stages**"

对照:

| | 老树 | 新树 |
|---|---|---|
| 塔与文本分成独立 stage | 有 | **有**(`fqns = [[embed, "vision_encoder"]] + fqns`) |
| 塔**跨多个 stage** 切分 | 有(`_dep_bounds`,"First share: patch_embed + early blocks") | **没接**:`dep_vision_stages()` 返回 1 |
| 跨 stage 传 patch 的静态形状补齐 | `pack_stage_patches` / `stage_patch_capacity` | 因上一行而未使用 |

`pack_stage_patches` 的存在理由是 PP 的 send/recv 需要静态形状,所以把 patch 补齐到
固定 capacity —— **只有塔跨 stage 时才需要**。新树塔整个在一个 stage 上,所以现在用不到。

新树的多 stage 机器(`block_bounds`,`pipeline_adapter.py:1145`)是存在的,但只接在
wrapper 布局那条分支上;`dep_vision_stages()` 的 docstring 自己写着 ">1 是目标,
需要实测来定"。

结论:这不是架构差异,是**多 stage DEP 还没接到新布局上**。属于待办,不属于已完成。

---

## 八、DEP clause 1 有真实数值差(2026-08-24,待定位)

干净对照(dp2 x pp4,每格独立 cache + 独立 dump-folder + 预热,排除缓存与 resume):

| | step1 | step2 | step3 |
|---|---|---|---|
| dep_off | 12.46615 | 12.46615 | 10.44790 |
| n_vit=1 | 12.47875 | 12.47875 | 10.60172 |

step1 差 1.3e-2,step3 差 1.5e-1(随步数放大)。**缓存已排除,差异真实。**

DEP clause 1(塔独占一个 stage)是纯调度改变,**不应改 step-1 loss**。布局日志确认
DEP 生效:n_vit=1 的 stage 0 = `[tok_embeddings, vision_encoder]`,文本 24 层重分到
后 3 个 stage;dep_off 则是 `vision_encoder` 与 layers.0-5 挤在 stage 0。

**与老树冲突**:老树 `DEP_30L_RESOLVED_2026-08-10` 记录 DEP off / n_vit=1 / n_vit=2
三者 **identical**。所以这是新树迁移引入的回归,不是设计问题。

注意 dep_off 与 n_vit=1 的 stage **切分不同**(文本层边界移动),这本身在同一 seed ckpt
下也不该改 step1 loss —— 待查是 DEP 的塔归属改了数值,还是文本层重分本身在新树里
就不 behaviour-free(若是后者,PP 的 behaviour-free 性质也要重验)。

`n_vit=2` 另有拓扑门槛:`num_stages must be divisible by pp_degree`,老树用 30L flavor
+ 特定 pp/vp 跑,24L debugmodel + pp4 不满足。定位 clause 1 之后再处理。

**状态:DEP 端到端未通过。塔的 share 分解单元测试逐位等价(atol=0),但整机 DEP
改数值。** 定位中。
