# MoonViT 声明式迁移的范围(2026-08-23)

写这份是因为我对它的规模判断错了两次,方向相反。记下正确的形态和它为什么值得做。

## 两次错判

**第一次**:"视觉侧几乎没开始,连命令式都没做。"

错。`_apply_tp_moonvit_mlp` 同时处理 MLP 和注意力头,`vit_tp_heads` 默认 True。
我只看了函数名没读函数体。

**第二次**:"MoonViT 要从零把 935 行子树 config 化,规模比 AttnRes trunk 大得多。"

也错。上游有 **`torchtitan/models/common/vision_encoder.py`,只有 170 行**,提供
`VisionMLP` / `VisionAttention` / `VisionTransformerBlock` 三个带 Config 的 `Module`。
正确路径是**采用共享 block**,不是自己 config 化。我只看了 `muse_glimmer/` 和
`vision_encoder_sharding.py`,漏了 common 下的实现。

两次的共同点与本轮另外三次一样:**没把候选列全就下结论**。

## 现状基线(实测,不是推断)

`kimi_k3_debugmodel_report_arch`,dp2 + tp2,mesh 两维依次 (fsdp, tp):

| 参数 | tp 轴 |
|---|---|
| `encoder.blocks.N.mlp.fc0.weight` | `Shard(0)` |
| `encoder.blocks.N.mlp.fc1.weight` | `Shard(1)` |
| `encoder.blocks.N.wqkv.weight` | `Replicate()` |
| `encoder.blocks.N.wo.weight` | `Replicate()` |
| 其余(norms / patch_embed / mm_projector) | `Replicate()` |

**只有 MLP 被 TP 切。** 注意力头切分的分支存在且开关默认打开,但条件是
`num_heads % tp_size == 0`,而这个 flavor 是 3 头 / tp=2,走了复制分支。

## 阻塞点:融合的 wqkv

`moonvit.py:390` 是一个融合 `nn.Linear`,前向里 `.view(L, 3, num_heads, head_dim)` 拆开。
上游 `VisionAttention` 的 docstring 写明了为什么不这样做:

> Separate q/k/v projections (**clean per-head ColwiseParallel under TP**)

**上游拆 QKV 的理由,正是我们切不动的原因。** 而官方 checkpoint 侧是融合的
(`test_hf_key_map.py:159` 有 `vision_tower.encoder.blocks.3.wqkv.weight`),
所以拆分不是改名,要在 state_dict_adapter 里做融合<->拆分的映射。

## 建议的两步

### 4a:只迁 MLP

`MoonViTMLP` -> `common.VisionMLP`,挂 `vision_colwise_config()` /
`vision_scaled_bias_rowwise_config()`,删掉 `_apply_tp_moonvit_mlp` 里那段逐 block 的
`plan[...]`。

* checkpoint 影响:`fc0`/`fc1` -> `linear_fc1`/`linear_fc2`,两个名字
* 等价判据:上表的 `Shard(0)` / `Shard(1)` 必须保持
* 风险可控,是"共享 block 能不能套上 MoonViT"的可行性验证

注意 `fc0`/`fc1` 现在是**裸 `nn.Linear`**(`moonvit.py:367-368`,`bias=False`),
声明式驱动器本来就够不到它们 —— 所以这一步同时解决"挂不上声明"的问题。

### 4b:拆 QKV,解锁按头切分

`MoonViTEncoderLayer` -> `common.VisionTransformerBlock`,`wqkv` -> `wq`/`wk`/`wv`,
state_dict_adapter 加映射,挂 `set_vision_transformer_block_sharding_config`。

收益是实质的:这一步之后视觉注意力才能真正 TP 切分,也就是
`parallelize.py:752` 注释里搁置的那件事,以及 report 5.2.3 要的
"genuinely parallel vision tower"。

**但它跨模型定义、checkpoint 映射、并行三层,而 58 格里 40 格是多模态臂。**
应当有自己的 gate,不要和 4a 合并。

## 与 LoRAConverter 的关系

无关。converter 的四个阻塞里,第一个(根 Config 树)已由 `cfd84b87c` 解决,
其余三个是:FFN 字段名 `w1/w2/w3` 对目标名 `gate/up/down_proj`、
末段匹配无法区分 MLA 与 KDA 的 `q_proj`(实测 18 个匹配里 6 个落在 `delta_attention`)、
以及 mxfp4 packed base 无对应。

第二条需要给 `LoRAConverter.target_modules` 加带点后缀匹配 —— 我们自己的 `apply_lora`
早就支持,上游没有。那是**加功能**,与 optimizer / dataloader 那两处**修缺陷**性质不同,
应当单独提。


## 4a 试做过了,卡在边界约定(2026-08-23)

把 `MoonViTMLP` 改成带 `Config` 的 `Module`、两个 linear 从 `Linear.Config` 构建、
挂上声明、删掉 `_apply_tp_moonvit_mlp` 里的两条 `plan[...]`。逐步撞到四层问题,
前三层都修掉了,第四层是结构性的。

| # | 现象 | 处理 |
|---|---|---|
| 1 | `Linear.weight is already a DTensor with placements (Replicate(),)` | `apply_tp` 里的 `distribute_module(vision_tower)` 在驱动器之前把整塔变成 Replicate。改成在 `apply_tp` 之前先对塔跑一次驱动器 |
| 2 | `input DTensor has placements (Shard(dim=1),), but in_src expects` | 上游 `vision_scaled_bias_rowwise_config` 假定 3-D `[B,L,D]`(特征轴 2);MoonViT 是 varlen packed 的 2-D `[L,D]`(特征轴 1)。写了 2-D 变体 |
| 3 | `SpmdLayout has multiple mesh axes sharding tensor dim 1` | `dense_activation_placement` 同时给 CP 轴 `S(1)`,与 TP 撞在同一张量维。改用只含 DP/TP 两轴的 `SpmdLayout`,和上游视觉 helper 一致 |
| 4 | **`aten.mm.default got mixed torch.Tensor and DTensor`** | **未解决** |

第 4 层不是配置写错。实测三件事同时成立:

* placement **正确**:`fc0` = `Shard(0)`,`fc1` = `Shard(1)`,与迁移前基线一致
* forward wrapper **装上了**:报错位置在 `protocols/module.py:291 forward_with_redistribution` 内部
* **plain 输入没有被提升成 DTensor**

也就是说声明生效了、参数切对了,但输入侧的重分布没有把 plain 张量抬进 DTensor。

### 根因:MoonViT 是 plain 张量边界

文本侧的声明式能工作,是因为残差流早已翻成 DTensor 端到端。MoonViT 相反 ——
`apply_tp` 里 `distribute_module` + 每个 style 的 `use_local_output=False` 就是为了让
**模块边界始终是 plain 张量**(fla 内核、PP send/recv、AttnRes 的 `torch.stack` 都不吃 DTensor)。

声明式的输入重分布假定流里已经是 DTensor。要让 4a 成立,得先把视觉塔的边界约定翻成
DTensor —— 那不是"迁移 MLP",那是改整个塔的张量约定,规模等同于文本侧当初那次翻转。

**这与层内两个 AttnRes norm 迁不动是同一个根因**:声明式词汇表描述的是 DTensor 流上的
placement,而这两处的流是 plain。

### 上面这个"根因"是错的,当天就被自己的测量推翻

写完上一节后我去测了当前(命令式)树上 MLP 实际收到什么:

    [MLPIN] x=DT(Replicate(),)  fc0.w=DT(Shard(dim=0),)

**输入本来就是 DTensor。** `encode_images` 在进塔之前就做了提升
(`multimodal_model.py:292-296`:`DTensor.from_local(packed, tp_mesh, (Replicate(),))`),
所以"MoonViT 全程 plain 边界"不成立。

而且 `apply_tp` 注释里给 plain 约定的三个理由,逐条核下来只有一个沾边:

| 理由 | 对视觉塔是否成立 |
|---|---|
| fla 内核不吃 DTensor | **否** —— MoonViT 全文 0 处 fla,用的是 `F.scaled_dot_product_attention` |
| AttnRes 的 `torch.stack` | **否** —— 塔里 0 处 `torch.stack` |
| PP send/recv | 仅在 DEP 的 stage 边界,不是塔内部 |

## 做完了(2026-08-23),根因是 Linear.forward 自己拆权重

第六次尝试才测对。`torchtitan/models/common/linear.py`:

    def forward(self, input):
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return F.linear(input, weight, bias)

**`Linear.forward` 自己把权重 `to_local()`。** 所以只要声明了 `in_src`/`in_dst`,
框架就会把输入提升成 DTensor,它撞上本地权重 -> `aten.mm.default got mixed`。

core 的 `colwise_config` / `rowwise_config` 的 `in_src`/`in_dst` **都是 `None`**,正是为此。
**加输入声明才是错的**,与边界约定、mesh 维度都无关。

最终形态(`91feba654`):`MoonViTMLP` 成为带 `Config` 的 `Module`,两个 linear 从
`Linear.Config` 构建,只声明**权重切分与输出布局**;`_apply_tp_moonvit_mlp` 里那两条
`plan[...]` 删除;驱动器在 `apply_tp` 之前先跑一次塔,以免 `distribute_module` 抢先。

验证:placement 与基线一致(`fc0`=`Shard(0)`、`fc1`=`Shard(1)`),
两个多模态臂十步数值与迁移前完全相同,429 单测通过。
(该文件里 4 个 N814 lint 警告是预先存在的。)

## 曾经卡住的地方,以及五个被推翻的结论

TP-only 声明版本的最后一次测量:

| 观测 | 命令式(基线) | 声明式 |
|---|---|---|
| 权重(静止) | `(_StridedShard(0,sf=2), Shard(0))` | **相同** |
| MLP 输入 | `DT(Replicate(),)` | **相同** |
| `fc0.weight`(前向中) | `DT(Shard(dim=0),)` | **相同** |
| 结果 | 训练正常 | `aten.mm.default got mixed` |

**输入、权重、placement 三者都与基线一致,但 `Linear` 内部的 mm 仍报 mixed。**
探针打在 `MoonViTMLP.forward` 开头,即 `fc0` 被调用之前;所以 mix 发生在
`Linear.forward_with_redistribution` 对输入做重分布**之后**。

那个猜测(1 轴 `SpmdLayout` 在 2 维 mesh 上把 DTensor 掉成本地张量)**也是错的**。
真正的位置在 `Linear.forward` 自身,见上一节。输入重分布从不拆 DTensor ——
`_redistribute_inputs` 只做提升、断言、重分布三件事。

### 过程中修掉的三层(都是真问题,重做时会再遇到)

| # | 现象 | 处理 |
|---|---|---|
| 1 | `weight is already a DTensor with placements (Replicate(),)` | `apply_tp` 的 `distribute_module(vision_tower)` 抢在驱动器前面。在 `apply_tp` 之前先对塔跑一次 `_drive_declarative_sharding` |
| 2 | `input DTensor has placements (Shard(dim=1),), but in_src expects` | core 的 `vision_scaled_bias_rowwise_config` 假定 3-D `[B,L,D]`;MoonViT 是 varlen packed 的 2-D `[L,D]` |
| 3 | `SpmdLayout has multiple mesh axes sharding tensor dim 1` | `dense_activation_placement` 同时给 CP 轴 `S(1)`,与 TP 撞在同一张量维 |

还有一个自己制造的:替换类头时留下了**重复的 `forward`**,后定义的覆盖前者,
导致探针整整两轮没有输出而我以为"forward 没被调用"。

### 这一节该记的教训

**同一个问题上连续五次下结论、五次被自己的下一个测量推翻:**

1. "视觉侧几乎没开始" -> `_apply_tp_moonvit_mlp` 同时处理 MLP 和注意力头
2. "注意力头是切的" -> 实测 `wqkv`/`wo` 都是 `Replicate`(3 头 / tp2 除不尽)
3. "MoonViT 要从零 config 化" -> core 有 170 行的 `vision_encoder.py`
4. "边界是 plain,要翻整个塔" -> 实测输入本来就是 DTensor
5. "是 mesh 维度不匹配" -> 实测两版权重与输入完全一致

每一次都是**读了一段能解释现象的代码就下结论**,而没有先跑那个能一锤定音的观测。
正确的顺序是:先测当前行为,再改;不是先改,再用报错反推当前行为。

树已还原到 `cfd84b87c`。命令式实现是正确的、有实测基线,不构成阻塞;
这条线要继续,从上面"下一个该测的"那一步开始。


## 按头切分那条路:gate 从未执行,实测是好的(2026-08-23)

`_apply_tp_moonvit_mlp` 里 `shard_heads` 那条分支(`wo` 走 RowwiseParallel、
q/k/v `to_local(grad_placements=[Partial()])` 后切片)的条件是
`num_heads >= tp_size and num_heads % tp_size == 0`。

| flavor | 视觉塔头数 | tp2 下 |
|---|---|---|
| `kimi_k3_debugmodel_report_arch`(**gate 用**) | 3 | `3%2!=0` -> 复制 |
| `kimi_k3_mini_vl` | 4 | -> **按头切** |
| `MoonViTConfig` 默认(官方) | **12** | -> 按头切 |

**所以 58 格从来没有执行过这条分支。** 官方配置是 12 头,真实训练会走它。

用 `kimi_k3_mini_vl`(4 头)在 dp2+tp2 下实测:

    [VIT] encoder.blocks.N.wo.weight    (Shard(dim=0), Shard(dim=1))   <- tp 轴切了
    [VIT] encoder.blocks.N.wqkv.weight  (Shard(dim=0), Replicate())    <- 按设计保持复制

5 步训练正常。TP 是否透明,用查词表缺陷那次的同一把尺子对照:

| | step 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| dp4(无 TP) | 25.588 | 19.156 | 19.588 | 16.095 | 17.581 |
| dp2+tp2(按头切) | 24.761 | 18.914 | 19.902 | 16.405 | 17.348 |

差 1-3%,与 llama3 对照的量级一致(切分改变初始化时的 RNG 消耗),
**不是词表那种 5.6 倍膨胀**。grad_norm 绝对值大是这个 flavor 的规模/lr 特性。

### 这改变了拆 wqkv 的性价比

现有方案(复制 wqkv + 投影后切片)**能工作且 TP 透明**。所以拆成 `wq`/`wk`/`wv` 的收益只剩:

* 省掉冗余的 qkv matmul —— 每个 rank 现在算全部头再丢掉别人的
* 结构与 core 的 `VisionAttention` 一致,PR 更容易过

代价是**改官方 checkpoint 的加载映射**(released 侧是融合的 `wqkv`)。

**这是性能与一致性的改动,不是修缺陷。** 先前我把它写成"解锁视觉注意力的 TP 切分",
那是错的 —— 切分已经解锁了,只是 gate 的 flavor 头数不整除所以看不到。
