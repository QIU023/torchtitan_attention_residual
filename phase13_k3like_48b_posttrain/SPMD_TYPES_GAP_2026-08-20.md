# `spmd_types` 缺口清点(2026-08-20)

> **2026-08-21 重要更正:本文的核心判据是错的。** 下面按 "DTensor vs plain" 计数,
> 并把 plain 当成阻塞证据。**它不是。** 对照实验:`llama3_debugmodel` 在同一个 torch、
> 同样的 `spmd_types` 下**训练正常**,而它的参数是 **0 DTensor / 45 plain**。
> 所以"590 个 plain"这个说法量错了东西,详见文末"判据更正"。
> 缺口本身仍然是真的 —— 我们失败、llama3 成功 —— 但差别不在 DTensor 上。

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

### 追到了具体函数(2026-08-21 续)

排除法一步步收紧,每步都是实测:

* **时机不是问题** —— 声明时零个 norm 已被 parallelize;
* **驱动器确实进了** —— `entered parallelize() on 26 outermost Modules:
  {'KimiAttnResDecoderLayer': 21, 'RMSNorm': 2, ...}`;
* **声明确实读得到** —— `Module.parallelize` 读的就是 `_sharding_config`,而且先递归子模块;
* **不是漏声明** —— 驱动器之后直接数:`RMSNorm params dtensor=0 plain=118,
  modules without a declaration=0`。**118 个全有声明,全是 plain。**

(`declare_norm_sharding` 报 20 是因为它跳过已有声明的;另外 98 个早就有。)

落点因此是 `Module._distribute_states` 里的分支:`spmd_backend == "spmd_types"` 走的是
**`_spmd_distribute_state`**,和 partial_dtensor 的 `distribute_tensor` 是两条不同的路。
它调 `spmd_distribute_tensor` 并配 `set_current_spmd_mesh`,而结果没有变成 DTensor。

### 下次从这里开始:先证伪判据本身

有一个可能推翻整条推理的点:**`spmd_types` 也许用类型标注而不是 DTensor**,那么探针里的
`isinstance(param, DTensor)` 就是错的判据,"全是 plain" 也就不构成问题。

但 FSDP 的报错明确说要 DTensor。两者必须对齐,而**先分清这一点再动手**,否则可能在修一个
不存在的问题 —— 这一轮已经有两次归因错在"没先证伪前提"上。

### 安全属性守住了

声明只在 `spmd_backend == "spmd_types"` 分支执行,`partial_dtensor` 一行都不走。
已验证:cp2 冒烟 step 1 = 7.71140 / 3.3554,与基线逐位相同;427 单测通过。


## 判据更正(2026-08-21)

追查一路收窄到 `Module._distribute_states` 的 spmd 分支后,发现两条互相矛盾的事实:

* `spmd_distribute_tensor` 用 `spmd.shard` 切出**本地分片并返回 plain tensor**,从不造 DTensor;
* FSDP 的 `_init_sharding_spec` 却在 `is_spmd_mesh and not is_dtensor` 时抛错。

若两者都成立,所有模型在 spmd_types + FSDP 下都该失败 —— 而它是上游默认。**这个矛盾说明
我某个前提错了**,于是做对照实验而不是继续推理:

| 模型 | spmd_types + FSDP | 参数形态 |
|---|---|---|
| `llama3_debugmodel` | **训练正常** | 0 DTensor / 45 plain |
| `kimi_k3_mini_kcp` | 报错 | 0 DTensor / 590 plain |

**两边参数形态一样,结果不同。** 所以 `isinstance(param, DTensor)` 不是区分标准,
"plain" 也不构成缺陷。差别在那些 plain 张量**是否携带 spmd 类型标注** ——
由 `_spmd_distribute_state` 经 `spmd.assert_type` 附加,而 `isinstance` 看不见它。

同时被这个对照推翻的还有另一个刚成形的假设:**不是 torch 与 torchtitan 的版本错位**。
同一个 torch,llama3 通、我们不通。

`probe_plain_params.py` 里那句 `<-- blocks spmd_types` 已删。**一个会自信地给出错误结论的
探针,比没有探针更糟**;它当时让"590 plain"看起来像已证实的诊断。

### 下次从这里开始

量"参数是否带 spmd 标注",不是量 DTensor。llama3 是现成的参考实现:它一行
`model.parallelize(parallel_dims)` 覆盖整棵树,而我们只覆盖了零散几处 —— 那才是真正的差别,
也是最初"零声明"那个观察真正指向的东西。


## 修复:非 TP 配置已通(2026-08-21)

`spmd_types` 现在能跑,**CP 也能跑** —— 这是上游 PR-4218 要求的组合,也是这条线的目标。

### 判据换对之后

| 判据 | llama3(可跑) | kimi_k3(不可跑) |
|---|---|---|
| `isinstance(p, DTensor)` | 0 DTensor / 45 plain | 2 / 590 |
| **`has_local_type(p)`** | **0 无标注** | **143 无标注** |

用 DTensor 量,两边看起来一样;用 `has_local_type` 量,差别一目了然。**换判据也让先前的工作显出效果**:
RMSNorm 117 个、AttnResProjection 42 个、KimiDeltaAttention 30 个早已带标注,只是用错尺子看不见。

### 三处改动

1. **`annotate_untyped_params`** —— 给声明到不了的参数补 replicate 标注:fla 的
   `ShortConvolution`/`FusedRMSNormGated`(根本不是 torchtitan `Module`)、EP 的专家权重、
   少数在已声明子树外的 Linear 和 embedding。**replicate 不是占位**:未分片的参数本来就在每个轴上
   复制,标注只是把已为真的事实告诉 FSDP —— 所以数值一点不动。
2. **`drop_declarations_on_distributed`** —— 摘掉命令式 TP 已分发模块上的声明。驱动器的
   `_already_distributed` 守卫只在子树根成立,而 `Module.parallelize()` 一旦调用就带着递归走遍
   全树、不带守卫,于是撞上 `assert_type() does not support DTensor`。
   `partial_dtensor` 下同样的递归无害,因为 `_distribute_states` 对 DTensor 另有一条只校验的分支。
3. **`verify_params_distributed` 按后端选判据** —— 它要求"TP 后全是 DTensor",而 spmd_types 下
   参数**本来就该是本地张量加标注**,那个要求会拒掉正确状态(实测拒掉 80 个)。改成
   "DTensor 或带标注",保护力不变:没标注的本地张量照样会在 `clip_grad_norm_` 里炸。

### 验证

| 检查 | 结果 |
|---|---|
| cp2 + spmd_types | **与 partial_dtensor 逐位相同**(7.71140 / 3.3554 …) |
| 文本 flavor | 通(141 补标注) |
| 多模态 flavor | 通(105 补标注),两个 FSDP 入口都过 |
| partial_dtensor(含 TP) | 数值不变,且**完全不进新代码** |

### 仍不通:TP + spmd_types

TP 格失败在 FSDP:`Expected param's DTensor mesh to be the same mesh passed to fully_shard`。

**命令式 TP 造的 DTensor 在 tp mesh 上,而 spmd_types 的 FSDP 要参数在完整 SPMD 存储 mesh 上。**
这不是能绕的接线问题 —— 它就是"TP 也必须声明化"本身,是独立且更大的下一块。

所以现状是:**非 TP 的 CP 配置已经兼容上游要求,带 TP 的还不行。**


## 剩余两块,各自独立(2026-08-21)

`spmd_types` 下现在**无 TP 无 EP 的 CP 配置全部可用且与 partial_dtensor 逐位一致**。
21 格里能通的是 `cp2`、`fsdp2_pp2_cp2`、`cp4`;剩下四格分属两个独立问题。

### TP:命令式计划与 spmd_types 互斥

`Expected param's DTensor mesh to be the same mesh passed to fully_shard` ——
命令式 TP 把参数放在 tp mesh,spmd_types 的 FSDP 要完整 SPMD 存储 mesh。

试过声明化,**回退了**,因为发现三个必须先处理对的点:

* KDA 层的模块叫 `delta_attention.*`,后缀匹配 `attention.q_proj` 会**连它一起命中**;
* KDA 层在命令式计划里**根本不做 TP**(`pass`,保持复制),照抄 MLA 的 colwise/rowwise
  会给它强加错误分片;
* LoRA 包装要跟进 `.base`。

回退的直接理由不是"没写完",而是它有**现实危害**:那段代码在 spmd_types 下跳过了命令式 TP,
自己却声明了 0 个,等于**静默地完全没有 TP** —— 比报错更糟,报错至少会停下来。

好消息是规模比看上去小:TP 计划 13 个条目里真正分片的只有 6 个,其余都是 replicate,
而补标注扫描已经覆盖。

### EP:专家侧缺 mesh_dims

`spmd_types parameters require fully_shard() to be called with both a named full
DeviceMesh and dp_mesh_dims`。密集侧用 `resolve_fsdp_mesh` 同时算出 mesh 和 dp_mesh_dims,
**专家侧没有对应物** —— `apply_fsdp` 收 `edp_mesh` 但从没有人传 `edp_mesh_dims`。

### 判据按后端分叉,已修三处

`_distribute_states`、`verify_params_distributed`、`verify_ep_applied` —— 三处都在用
"是不是 DTensor"当判据,而 spmd_types 下正确状态就是本地张量。**每一处都是保留问题、只换证据**,
并各自补了守护新分支的测试。第三处(EP)的等价证据是本地形状:专家维被切,dim 0 应缩小 ep 倍。

## 一条工作方法上的教训

取证跑到一半改被测的树,发生了**两次**。每个格子是新进程、导入当前代码,所以那种跑混了版本,
产出的证据需要靠记忆加注解才能读 —— 我两次都直接删掉了产物,而不是留着。

不是注意力问题,是节奏错了:**代码改动全部做完并过单测,再冻结树开跑,跑期间只读不写。**
