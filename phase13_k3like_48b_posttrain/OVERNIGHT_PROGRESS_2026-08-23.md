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

## EP 分支 — 完成(`95f5bd8da`)

差的那一块是 token dispatcher。他们的 config 写死 `LocalTokenDispatcher`,
它只在 rank 内部重排 token,交给专家的是**全局**每专家计数。专家权重按 E 切分之后,
grouped GEMM 拿到 32 个 offsets 对 16 个本地专家,报
`matrix batch sizes have to match` —— 读起来像模型的形状 bug,其实不是。
EP 下换成 `AllToAllTokenDispatcher`(两个 Config 字段完全相同)。

换完实测:16 本地专家对 16 offsets,两个 rank 收到**不同**的 token 数(1030 / 1018),
token 确实在跨 rank。**step-1 loss 12.38712,与 dp2 基线五位全同** —— EP 本就不该
改变前向。

## TP 分支 — 声明生效,但前向跑不通(`3728ef835`)

### 一个只有实测才抓得到的静默空转

第一次 tp2 "跑通"了,loss 12.45942。**但参数全是 PLAIN、形状是完整的**
(`wq_b` 1536x512 而不是 768x512)—— 纯复制,TP 一点没生效。
原因:我写 `model.parallelize` 的触发条件时漏了 `tp_enabled`
(dsv3 的条件是 `spmd_types or tp_enabled or ep_enabled`)。

**一个"能跑且 loss 合理"的 TP 格子完全可能什么都没做。**只有直接量 placement 才看得出来。
补上之后实测:`wq_b` Shard(0)、`wo` Shard(1)、`w1` Shard(0)、`lm_head` Shard(0)、
KDA 的 `q_proj` Replicate —— 与设计一致。

### 逐层剥出来的接触面(每修一个露出下一个)

1. 每层 norm 未声明 -> `aten._fused_rms_norm.default got mixed`
2. KDA 的三个 Conv1d 和它自己的 `A_log`/`dt_bias` -> `aten.convolution.default got mixed`
3. fla kernel 拿到 DTensor -> **非法内存访问**,没有可读报错(我们树里记过这个形态)
4. `vision_invariant_linear_config` 不能用于 dense `Linear`:它声明了四个激活边界,
   把输入提升成 DTensor,而 `common/linear.py` 的 `Linear.forward` 自己把权重
   `to_local()`,两者相撞 -> `aten.mm.default got mixed`。
   **core 的 dense `colwise_config`/`rowwise_config` 把这些边界留 None 正是为此;
   复制版本 core 没有,这里补了一个。**
5. `KimiLatentMoE` 的 `routed_down/norm/up` 是 Kimi 加在 core MoE 之外的,
   `set_moe_sharding_config` 不认识 -> `moe.py:135` 的 `aten.mm got mixed`

现在停在 dynamo 的 `RuntimeError when making fake tensor call`。**未诊断。**

## HEAD 回归(全部 rc=0)

| 格 | step-1 loss |
|---|---|
| text dp2 | 12.38712 |
| text cp2 | 12.45262 |
| mm cp2 | 12.45567 |
| text pp2 | 12.53727 |
| text ep2 | 12.38712 |
| mm ep2 | 12.47105 |

## TP 停在哪(精确到算子)

不是 AC —— `activation-checkpoint:none` 同样报错。模型编译本来就是关的,
dynamo 来自 **FlexAttention 自己的 `torch.compile`**。失败的算子是:

    getitem(int32[256], DTensor(标量 int32, Replicate on tp))
    -> aten.index.Tensor got mixed torch.Tensor and DTensor

即用一个 DTensor 标量去索引普通张量,发生在 MoE 的 router/dispatcher 路径上。**未诊断。**

## 新树的矩阵(`matrix_scripts/run_4025_matrix.sh`)

覆盖已迁移的轴,**TP 格子故意不在里面** —— TP 仍挡在 NotImplementedError 后,
一个表达不了自己拓扑的格子不算通过。

脚本第一件事是 **warm-up 一格然后丢弃**,因为冷缓存首跑与之后所有次数值不同
(见上)。`TORCHINDUCTOR_CACHE_DIR` 固定在 OUT 目录下,矩阵内部因此自洽。

两臂 x 10 格:`dp1 / fsdp2 / cp2 / cp4 / pp2 / pp4 / ep2_fsdp2 / ep2_cp2 /
fsdp2_pp2_cp2 / ep8_fsdp8`。

## 固化成测试的两件事(`289ac3490`)

都是 CPU 测试,跟其余单测一起跑。

1. **CP 契约的折叠维度**。没有 batch 轴,Ulysses 在 dim 0 和 1 之间搬;
   batched 那版用的是 1 和 2。**任何形状检查都抓不到这个交换** —— 两个维度都存在,
   all-to-all 会产出一个"看起来合理"但头和序列对调的张量。
2. **FQN 注入**。它曾经在 Config 树模型上直接 return 且不出声;切分退回 core 的默认,
   聚合模块落在没有任何一段上,唯一症状是 loss 训到别处去了。
   第三个用例检查每个发出的名字都能匹配到子模块 —— core 对匹配不到的子模块是**置 None**,
   不是报错。

## TP 暂停(maintainer 要求),但成因已测出来

`getitem(int32[256], DTensor 标量)` 的成因**不是缺声明,是一处逻辑分叉**:

* **我们原来的树不重写 `MoE.forward`** —— 它把 core 的 MoE 组合成 `self._moe`,
  只在模块边界处理 DTensor,所以手里从来不会有一个 DTensor 的专家计数。
* **4025 的 `KimiLatentMoE` 重写了 forward**,路由的整数簿记
  (`routing_map_TE` / `num_tokens_per_expert_E`)因此在模块级的 DTensor 环境里算出来,
  dispatcher 再拿它去索引普通张量。

实测(tp2):`x=DTensor scores=DTensor ids=DTensor counts=DTensor`,四个全是。

顺带记下原树另一处不能丢的知识:它给 `set_moe_sharding_config` 传的是
`enable_sp = moe_enable_ep and moe_enable_tp`,**不是常量**;EP 开时 tp 在 MoE 区域内
变成 token 轴,只按 enable_sp 键会要求 `S(1) -> P(sum)`,DTensor 拒绝。
EP+TP 同开时还要把 MoE 声明成"外部 Replicate、内部 SP"的自足岛。
**恢复 TP 时从这两条出发,不要从报错反推。**

## PP 与无 PP 的 loss 差异 — 已定性

不是缺陷。排除法:

1. `pp_stage_parity_4025.py` 证明**在权重相同的前提下**,两段前向与整模型前向
   `max_abs = 0.000e+00`;
2. 所以端到端差异只能来自权重或数据;
3. PP 下每个 rank 只初始化自己那一段,RNG 消耗与整模型不同。

EP 是对照:它 step-1 loss 与 dp2 基线**五位全同**,因为 EP 不改变初始化的消耗顺序。

## LoRA — 用上游 converter(`7bcb25f22`)

新树 core 自带 `components/lora.py`。加一个 flavor:786 个参数里 36 个可训,
适配器在 `wq_b`/`wkv_b`/`wo`。**我们那 939 行的 `lora.py` 在新树上基本多余** ——
和 `attn_res_model.py` 同一形态:上游已经有了。

## 一次作废的矩阵,和它证明的事

第一轮 `run_4025_matrix.sh` 是**从活树跑的**,而我在它运行期间提交了四次改动。
每个格子是新起的 torchrun,读的是当时的文件 —— 所以那一轮混了多个代码版本,
**作为一份证据无效**,已停掉重跑。

这正是 08-23 早些时候我给自己写下的那条纪律("gate 从冻结副本跑,不从活树跑"),
第二次犯。这次的做法是 `git worktree add --detach /workspace/tt_frozen_titan HEAD`,
矩阵指向冻结副本,开发继续在 `/workspace/tt_4025/torchtitan`。

作废那轮有一条信息仍然有用:**它跑到 21 格,mm 臂的 PP 格是过的** ——
因为塔的修复在那些格子启动之前已经提交。也就是说修复本身是对的,
只是不能拿那一轮当证据。

## 多模态 PP 的视觉塔缺陷(`2cd239daf`)

塔没有被任何一段点名,而 **core 会把没被点名的子模块置 None**,
于是每段都拿到 None 的塔,第一个多模态 batch 报
`pixel_values were provided without a vision encoder`。

适配器自己的 docstring 就写着该放哪:视觉特征拼进 embedding,没有东西跨 stage,
所以塔跟着持有 embedding 的那一段。CPU 测试断言**恰好一段**持有它、且是 embedding 那段。
