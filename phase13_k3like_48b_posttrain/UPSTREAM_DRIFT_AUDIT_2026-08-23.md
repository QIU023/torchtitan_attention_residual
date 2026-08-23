# 声明式实现相对上游的偏离审计(2026-08-23)

把我们自己写的每一个 `ShardingConfig` helper 和上游 `common/` 里的对应物逐个比对。
目的不是"看起来像",而是找出**我们发明了而上游已有**的地方 —— 这轮的六次错误归因里,
每一次的正确答案都在上游代码里,而我在自己发明解释。

## 结论速览

| # | 我们的 | 上游的 | 判定 |
|---|---|---|---|
| 1 | `norm` / `lm_head` 的声明 | `set_decoder_sharding_config` | **完全一致** |
| 2 | `_vocab_parallel_embedding` 的 `out_dst` | 同上 | **差一个字段** |
| 3 | `_tp_replicate()` | `invariant_norm_config()` / `vision_invariant_linear_config()` | **系统性偏离** |
| 4 | `_vit_colwise` / `_vit_rowwise` | `vision_colwise_config` / `vision_scaled_bias_rowwise_config` | **有意偏离,有实测理由** |
| 5 | `_tp_shard(dim)` | 无对应 | 我们特有,2 处使用 |

## 1. norm 与 lm_head:一致,但是手抄的

实测:`ShardingConfig` 对象逐字段相等。

但我们是**手抄进 `model.py`**,而不是调用 `set_decoder_sharding_config`。
上游那个函数一次设置 `tok_embeddings` / `norm` / `lm_head` 三项,而我们的
`KimiK3Model.Config` 与 `KimiK3AttnResModel.Config` **恰好就用这三个字段名**
(注释里写着 "carry the names core's set_decoder_sharding_config writes to")。

**也就是说我们已经为调用它做好了准备,却仍然抄了一遍。**

## 2. 词表 embedding 的 out_dst:R 对 I

唯一不同的字段:

| | `tok_embeddings.out_dst_shardings` 的 tp |
|---|---|
| 我们 | `R`(Replicate) |
| 上游(`enable_sp=False`) | `I`(invariant) |

含义不同:上游把跨 rank 求和**推迟到 norm**(`pre_lm_head_norm_config` 的 `out_dst` 是 R),
我们在 embedding 就做掉。两条路最终都归约,但**归约点不同,求和次序因此不同**。

我们这一版是 08-21 修 5.6 倍 grad_norm 膨胀时写的,当时按"补齐五件套"抄了上游,
但 `out_dst` 抄成了 R。**修复本身是对的(缺的是输入侧声明),但这一个字段跟着歪了。**

未验证:换成 I 是否仍然正确(理论上 norm 会补上那次归约)。要换必须跑 58 格。

## 3. `_tp_replicate()`:两处系统性偏离

我们的定义(`model.py:127`),用在 **19 处**:

    ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})

上游同类(`invariant_norm_config` / `vision_invariant_linear_config`):

    state:   {"weight": {DP: R, TP: I}, "bias": {DP: R, TP: I}}
    in_src:  {"input": {DP: V, TP: I}}
    in_dst:  {"input": {DP: V, TP: I}}
    out_src: {DP: V, TP: I}
    out_dst: {DP: V, TP: I}

两处差异:

* **权重的 tp 我们写 `R`,上游写 `I`。** R 是"复制"这个动作,I 是"不要碰"。
* **我们完全不声明激活边界,上游把四个边界都声明成 I。**

第二条的后果是:我们的模块对声明式驱动器而言"没有输入输出约定",
边界行为完全交给别处(命令式计划、`distribute_module`、或什么都不做)。
上游的写法把"这个模块在 TP 上什么都不做"显式表达出来。

**这是本次审计里最值得改的一处** —— 19 个使用点,而且它正是我们还留着一批
`NoParallel` 命令式条目的原因:声明里没有边界,边界只能由计划提供。

## 4. MoonViT 的两个 helper:有意偏离,理由已实测

我写的 `_vit_colwise` / `_vit_rowwise`(`moonvit.py`)与上游 vision helper 有两处不同:

| | 我们 | 上游 |
|---|---|---|
| `in_src` / `in_dst` | **不声明** | 声明 |
| mesh 轴 | 只有 TP | DP + TP |

第一条是**必须的**,而且有实测依据:声明输入边界会让框架把输入提升成 DTensor,
而 `common/linear.py` 的 `Linear.forward` 自己 `to_local()` 权重,两者相撞
-> `aten.mm.default got mixed`。core 的 `colwise_config` / `rowwise_config`
(dense 那一套)的 `in_src`/`in_dst` **也都是 None**,所以我们的写法和 dense 系一致,
只是和 vision 系不一致。

**未解释的是:上游的 vision 系为什么能声明输入边界而不撞同一个问题。**
`muse_glimmer` 用的是同一个 `Linear`。这一条我没有答案,标为未验证 ——
可能是它们的激活本来就是 DTensor 且 mesh 相容,也可能那条路在上游也没被跑过。

第二条(只用 TP 轴)是因为塔只在 tp_mesh 上被并行化,加 DP 轴会让权重落到
(fsdp, tp) 而激活在 (tp,) 上。

## 5. `_tp_shard(dim)`:我们特有

    ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.S(dim))})

只声明权重切分、不声明边界的 colwise/rowwise。上游没有对应物 ——
它的 `colwise_config`/`rowwise_config` 都带 `out_src`。

现在只剩 2 处使用(`attn_res_model.py:412` 的 AttnRes 投影、`model.py:471`)。
数量小,但同样是"边界靠别处"的写法。

## 该做什么

按性价比:

1. **调用 `set_decoder_sharding_config` 代替手抄的三项**,并顺带解决第 2 条的 `out_dst`。
   一次改动收两处,而且是"用上游的函数"而不是"抄上游的值"。
2. **把 `_tp_replicate()` 换成上游 invariant 系的形状**(权重 I + 四个边界 I)。
   19 个使用点,风险最大,但它是我们还留着 `NoParallel` 条目的根源。
3. `_tp_shard` 的 2 处,跟着 2 一起处理。

1 和 2 都会改变归约点或边界行为,**都必须跑 58 格**,而且判据是逐位 ——
如果数值变了,要能说清是"归约点移动"还是"切分错了",不能又去改判据。

## 一条方法上的记录

这份审计能做出来,是因为先把上游 `common/` 下的 helper **列全**(`decoder_sharding.py` 12 个、
`vision_encoder_sharding.py` 5 个),再逐个找我们的对应物。

这轮六次错误归因的共同形态,正是**没列全就下结论**:
"上游没有 LoRA 模板"(有 `components/lora.py`)、
"上游没有 vision 模板"(有 `common/vision_encoder.py`,170 行)、
"MoonViT 要从零 config 化"(同上)。

**"上游没有对应物"这种结论,必须来自把候选列全,不能来自试过一个。**
