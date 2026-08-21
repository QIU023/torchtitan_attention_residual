# `spmd_types` 缺口清点(2026-08-20)

工具:`matrix_scripts/probe_plain_params.py`。它不改树 —— 钩住 FSDP 的两个入口,在那里
把整棵树的参数按拥有模块归类。

## 为什么要清点

上游 PR-4218 之后 CP 强制要求 `spmd_backend='spmd_types'`,而我们的树在那个后端下起不来:

    ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on the
    full SPMD mesh (e.g. via distribute_module). Got plain tensor for parameter 'weight'.

这条消息只点名一个参数,不足以据此排工作量。

## 结果:文本侧 `kimi_k3_mini_kcp`

592 个参数,**590 个是 plain,2 个是 DTensor**。

| 拥有模块 | plain | 参数名 |
|---|---|---|
| Linear | **280** | weight |
| RMSNorm | **117** | weight |
| KimiSiTUGroupedExperts | 60 | w1_EFD / w2_EDF / w3_EFD |
| ShortConvolution | 45 | weight |
| AttnResProjection | 42 | weight |
| KimiDeltaAttention | 30 | A_log / dt_bias |
| FusedRMSNormGated | 15 | weight |
| Embedding | 1 | weight |

唯二两个 DTensor 是 `FSDPAttnResProjection` 和 `FSDPRMSNorm` —— **我们自己包的 FSDP 专用壳**。
也就是说,今天能变成 DTensor 的只有手工特殊处理过的,常规路径一个都没有。

视觉塔(`kimi_k3_debugmodel_report_arch`,在 `apply_fsdp_to_vision_encoder` 处):
30 个参数全 plain,零 DTensor。

## 决定方向的那一问:没声明,而不是没消费

**537 个持参模块里,带 `sharding_config` 的是 0 个。**

树里确实有 26 处 `sharding_config =`,但都在 MoE 的子配置和 moonvit 的局部路径上,
**没有覆盖 decoder 的任何常规模块**。

这一问必须单独测,因为两种情况的修法相反:

* 声明了没人消费 -> 接上 `model.parallelize(parallel_dims)` 即可,是接线活;
* **从没声明 -> 要为整棵树写声明**,对应上游 `set_llama3_sharding_config` 那一整套。

是后者。`Linear` 和 `RMSNorm` 用的就是上游的类、自带 `parallelize()`,机制齐备 ——
缺的纯粹是把 `sharding_config` 填上。

## 工作量排序

分布很集中:**Linear + RMSNorm = 397/590(67%)**。这两类覆盖掉,缺口降到三分之一。

剩下的要逐个想:

* `KimiSiTUGroupedExperts` —— EP 路径,专家维不是常规分片;
* `KimiDeltaAttention` 的 `A_log`/`dt_bias` —— 非常规形状,不是 Linear 那种 in/out;
* `ShortConvolution` —— fla 的模块,不是我们的类,不能直接加声明;
* `FusedRMSNormGated` —— 同上。

## 探针踩的两个坑

两个都不报错,只是**什么都不输出**,很容易被读成"没有问题":

1. **钩 `parallelize_kimi_k3` 无效** —— train spec 在 import 时按值存了函数,改模块属性到不了它。
2. **钩单个 `apply_fsdp_*` helper 无效** —— 哪个 helper 先撞取决于 flavor(多模态先死在视觉塔,
   文本先死在 decoder),钩一个就在另一个 flavor 上静默地什么都没测到。

现在两个入口都钩,并按 helper 分别清点。

## 第一次转换尝试的四个发现(2026-08-21)

norm 切片没做成,但把路探清楚了。四条都不是读代码能得出的:

**一、施加逻辑早就有,不用新写。** `parallelize.py::_drive_declarative_sharding`(149 行调用)
已经在遍历并施加声明。我一度另写了一个 `parallelize_declared`,是重复造轮子,已删。
它旁边的注释写着"只 ACTIVATES 已有的声明,不新增" —— **缺口正是"没有声明可激活"**。

**二、声明必须填在那次施加之前。** 我最初填在 FSDP 之前,晚了:诊断显示
`applied 0, skipped {'already': 631}` —— 631 个模块已被标记 parallelize 过,
包括我刚声明的那些。

**三、上游的"在 Config 上声明"这条路走不通。** `KimiK3AttnResModel`(flavor 实际构造的类)
调 `nn.Module.__init__` 后**直接从扁平的 `KimiK3Config` 建模块**,从不构造
`KimiK3Model.Config` 那棵树,所以没有 `.norm` / `.layers[i].input_layernorm` 可填。
声明只能打在构造好的模块实例上 —— 同一份契约,晚一步施加。

**四、117 个 "RMSNorm" 不是同一个类。** 按 `torchtitan.models.common.nn_modules.RMSNorm`
匹配只命中 **20 个**;其余 97 个是 fla 的同名类(KDA 层用的)。参数表按类名归类,
**同名不同类看起来完全一样**。这直接改变工作量:fla 的模块不是我们的,加不了声明,
和 `ShortConvolution`、`FusedRMSNormGated` 是同一类问题。

### 仍未解决

已声明的那 20 个 norm 施加后**仍是 plain**。原因未查明 —— 可能是驱动器跳过了它们
(`_already_distributed` 或 ep 标记),也可能是 `norm_config` 的 state_shardings 需要的 mesh
在那个时点不可用。**下次从这里开始**,而不是从头。

### 安全属性守住了

声明只在 `spmd_backend == "spmd_types"` 分支执行,`partial_dtensor` 一行都不走。
已验证:cp2 冒烟 step 1 = 7.71140 / 3.3554,与基线逐位相同;427 单测通过。
