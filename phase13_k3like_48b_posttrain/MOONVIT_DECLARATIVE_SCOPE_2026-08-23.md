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
