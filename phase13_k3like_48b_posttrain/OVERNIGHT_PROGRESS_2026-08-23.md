# Overnight 进度(2026-08-23,滚动追加)

树:`/workspace/tt_4025/torchtitan`,分支 `k3_on_4025`,起点 `dee45e357`。
计划见 `OVERNIGHT_2026-08-23_B_MIGRATION.md`。判据见同文件。

新母树基线(`kimi_k3_debugmodel`,dp2,3 步,seed 42 deterministic):
loss 12.51502 / 11.35441 / 9.89706。
文本基线(`kimi_k3_debugmodel_text`,dp2):loss 12.40265 / 9.97898。

## CP 分支 — 完成

| commit | 内容 |
|---|---|
| `03e1fdb1d` | KDA 的两条 CP 路径(kcp / ulysses) |
| `dfc04cbb9` | 接进 parallelize;core 加 `cp_via_sharding_config`;文本 flavor |
| `79abe69b2` | CP x 视觉塔 |

实测:
* all-to-all(折叠布局 3D):放置逐元素精确、round-trip 逐位、backward 正确,**cp2 + cp4**
* KDA CP vs 非 CP:**cp2/cp4 双模式 max_abs ~1e-6**(bf16 ulp 量级)
* 文本 cp2 训练:loss 12.45262 -> 10.25607
* 多模态 cp2 训练:loss 12.45567 -> 11.37354

### 上游接口改动(PR 里要单独说明)

1. **`Decoder.Config.cp_via_sharding_config`(新,默认 True)**。
   `validate_cp_backend` 自己的 docstring 写着它是给"在 ShardingConfig 里声明 CP
   的模型"调的,但 `Decoder.Config.update_from_config` 无条件调。KDA 跑 fla triton
   kernel,不走 DTensor 派发,**任何 ShardingConfig 都够不到它**,所以 K3 不可能是
   声明式 CP。声明式模型行为不变。

### K3 自己新增的前提条件

2. **CP 下禁止 load balancer**。默认 `headtail` 会跨 rank 置换 token;Ulysses 的
   all-to-all 按 rank 顺序重组序列,KDA 的递归按 rank r -> r+1 传状态,**两者都要求
   rank r 持有第 r 段连续序列**。置换后形状仍然对得上,数值静默错。

### 两个必须记下的机制

3. **mask**:core 切 CP mask 是按 **ring attention** 的形状(q 本地、kv 全局),
   而 Ulysses 每个 rank 重建完整序列 -> 尺寸不匹配。K3 禁止 sample packing,
   所以完整序列上就是纯因果 mask,在 Ulysses 路径里重建。**packed 序列需要把全局
   边界传下来,现在没有。**
4. **视觉塔必须留在每个 rank 的图里**。局部分片没有图像 token 的 rank 不消费任何
   embedding,塔就拿不到梯度,FSDP 少发一次 reduce_scatter -> 死锁(实测 300s
   watchdog,一个 rank 卡在 reduce_scatter,另一个已领先两个 collective)。
   加一个精确的零解决,embeddings 不变。

## 上游 vs 我们(已验证,不是读文档)

* PyTorch 的 `_ContextParallel` 实现的是 **ring attention**
  (`_templated_ring_attention` / `_RotateMethod`),**不是 Ulysses**。
  torchtitan 只在 docstring 里提它。
* 所以 MLA 的 Ulysses 和 KDA 的 KCP **上游都没有**。

## PP 分支 — 主体完成(`51f861fd5`)

搬过去的文件:`pipeline_adapter.py` 1702、`dep_bubble_{plan,backward,runtime}.py` 707、
`attn_res.py`、`layout.py`、`knobs.py`。唯一的重命名是 `vision_tower` -> `vision_encoder`(3 处)。

### 修掉的三个缺陷(都在他们的树上,都静默)

1. **块残差不跨 stage**。每个 stage 的 forward 都从 `new_zeros(T, 0, D)` 开始,
   stage 0 完成的块被丢弃,stage 1 训的是另一个模型 —— 全程没有任何形状报错。
   实测(同 tokens/step):无 PP `12.46284 / 9.62380 / 7.44679`,
   pp2 `12.48449 / 11.93899 / 9.30017`。现在残差是 stage 的第二个输出/输入。
2. **最终聚合在每个 stage 都跑**。非末段的 `output_res_proj` 是 None(和 `norm`/`lm_head`
   一样),现在只在持有 head 的那一段跑。
3. **FQN 注入在这个模型上直接 return 了** —— 它从扁平 config 的 `num_hidden_layers` 取层数,
   而这棵树把 layers 本身放在 config 里。没有任何提示:切分静默退回 core 的默认,
   末段就没有 AttnRes 聚合模块。同时把历史拼写(`embed_tokens`)映射回 core 的
   (`tok_embeddings`)——否则 FQN 匹配不到任何子模块,而 core 会把每个不匹配的子模块置 None,
   结果是一个没有 head 的 stage。

### 决定性验证:stage 接口逐位精确

`pp_stage_parity_4025.py`:把模型按 core 的做法切成两段(留一半层,其余模块置 None),
手工串起来,**不经调度、不经 loss、不经微批**。

    stage-split vs whole model: max_abs=0.000e+00 rel=0.000e+00

**模型侧的 PP 是对的。**端到端 pp2 与无 PP 的 loss 仍有差异
(`12.53727 / 9.87605 / 8.55103` vs `12.46284 / 9.62380 / 7.44679`),
但既然前向逐位相同,差异来自调度/微批切分/loss 汇报,不在模型里。
**这一条按 58 格用新树自己的基线判,不在这里下结论。**

## 最重要的发现:冷缓存上的第一次运行,数值与之后所有次不同

同一份代码、同一 seed、`--debug.deterministic`,只看 inductor 缓存目录的新旧:

| 同一个全新缓存目录 | step-1 loss | grad_norm |
|---|---|---|
| 第 1 次(冷) | **12.40963** | 19.0000 |
| 第 2 次 | 12.38712 | 19.2500 |
| 第 3 次 | 12.38712 | 19.2500 |

两个互不相干的全新目录,首次都给 12.40963;之后一律 12.38712。**完全可复现。**

### 我先后猜错两次,记下来

1. "是 `max_autotune` 造成的" —— **错**。`distributed/utils.py:227` 在
   `debug.deterministic` 下已经把 `max_autotune` 和 `coordinate_descent_tuning`
   关掉并重新编译 flex。
2. "是我那个不走 `set_determinism` 的探针污染了共享缓存" —— **也错**。
   在同一个新目录上先跑两次训练,第二次就已经变了,污染发生在那之后。

真正的规律只有一条,而且是实测的:**冷编译与命中缓存产生不同的数值。**
具体是 inductor 哪一层造成的没有诊断。

### 对 58 格的后果(这才是要紧的)

从冷缓存开跑的矩阵,**头几格会和其余格不同**,而这会被读成"某个并行轴改变了数值"。
缓解办法便宜:**跑矩阵前先用一次性 run 把缓存热起来**,或固定一个已预热的
`TORCHINDUCTOR_CACHE_DIR`。旧树没这个问题是因为它用 SDPA;新 core 已禁止
语言模型用 SDPA,所以新树必须显式处理。

## EP 分支 — 推进但未完成(`995f4b151`)

* 之前:`model.parallelize` 之后 **680 个参数全是 plain**(树上零声明)
* 现在:调用上游的 `set_decoder_sharding_config` / `set_moe_sharding_config`,
  声明落地、dispatcher 运行,失败点移到 grouped GEMM:
  `matrix batch sizes have to match` —— 本地专家数与 offsets 在 dispatch 与
  `inner_experts` 之间某处不一致。**未诊断**,EP 仍挡在 NotImplementedError 后面。

## 回归(HEAD,全部 rc=0)

| 格 | step1 / step2 loss |
|---|---|
| text dp2 | 12.38712 / 10.07772 |
| text cp2 | 12.45262 / 10.25607 |
| mm cp2 | 12.45567 / 11.43767 |
| text pp2 | 12.53727 / 9.87605 |
| text cp4 / mm cp4 | 12.42091 / 12.39273(3 步全过) |

## 未完成 / 跳过的

1. **EP 的 grouped GEMM 不匹配** —— 见上,未诊断。
2. **TP** —— 需要 MLA/FFN/KDA 的声明(根级和 MoE 已做)。KDA 只能声明为 invariant,
   fla kernel 不走 DTensor。
3. **vit dynamic CP / DEP** —— 文件已搬(`vit_cp_plan.py` / `dep_bubble_*`),未接线未测。
4. **PP 端到端与无 PP 的 loss 差异** —— 模型侧 stage 接口已证明逐位精确
   (`max_abs=0.000e+00`),差异在调度/微批/汇报,未定位。
5. **LoRA** —— 未开始。
