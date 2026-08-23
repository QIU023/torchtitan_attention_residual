# 四个 PR 的分支切分(2026-08-23)

树:`k3_on_4025`,基于 `dee45e357`(4025 的 head)。13 个 commit。
4025 merge 进 upstream/main 之后 rebase,再按这里切分支。

**TP 已按 maintainer 要求暂停**,但它的 commit 仍在树上、并挡在
`NotImplementedError` 后面,所以其余三个分支不受影响。

---

## 共享前置(必须先落,或并入第一个 PR)

| commit | 内容 | 为什么是共享的 |
|---|---|---|
| (在 `dfc04cbb9` 里) | `kimi_k3_debugmodel_text` flavor | **三个臂的 text 格都要它**。参照树只有多模态一个 flavor,text 臂失败无法归因到解码器还是塔 |
| `7bcb25f22` | `kimi_k3_debugmodel_lora` flavor | 用 core 的 `LoRAConverter`,零模型代码 |
| `289ac3490` 的一部分 | `vit_cp_plan.py` / `vit_prefetch.py` / `vision_preprocess.py` | 已搬、import 干净,**未接线**。属于 CP 分支的 vit 那一半 |

建议:把两个 flavor 单独切一个很小的 "add a text-only and a LoRA debug flavor" PR
先发。它零风险、零依赖,而且让后面三个 PR 的测试有共同的落脚点。

---

## PR-CP:context parallel

| commit | |
|---|---|
| `03e1fdb1d` | KDA 的两条 CP 路径(kcp / ulysses) |
| `dfc04cbb9` | 接进 parallelize;`Decoder.Config.cp_via_sharding_config`;text flavor |
| `79abe69b2` | CP x 视觉塔 |
| `289ac3490`(部分) | `tests/test_cp_contracts.py` |

**文件**:新增 `sharding.py` / `dtensor_ops.py` / `kcp.py`;改 `kda.py`(CP 路径)、
`model.py`(MLA Ulysses、CP 下的视觉 scatter、`cp_via_sharding_config`、
load balancer 前提)、`parallelize.py`(`apply_cp_kimi_k3`)、
**`common/decoder.py`(唯一的上游接口改动)**。

**上游接口改动要单独说明**:`Decoder.Config.cp_via_sharding_config`,默认 True。
`validate_cp_backend` 自己的 docstring 写着它是给"在 ShardingConfig 里声明 CP 的模型"
调的,但基类是无条件调的;KDA 跑 fla triton kernel,不走 DTensor 派发,
**任何 ShardingConfig 都够不到它**。声明式模型行为不变。

**未完成**:vit dynamic CP(`vit_cp_plan.py` 已搬,未接线)。

---

## PR-PP:pipeline parallel

| commit | |
|---|---|
| `51f861fd5` | PP,块残差作为 stage 载荷 |
| `2cd239daf` | 视觉塔的 stage 归属 |
| `289ac3490`(部分) | `tests/test_pp_fqn_injection.py` |

**文件**:新增 `pipeline_adapter.py` / `dep_bubble_{plan,backward,runtime}.py` /
`attn_res.py` / `layout.py` / `knobs.py`;改 `model.py`(块残差进出 stage、
head 守卫)、`parallelize.py`、`__init__.py`(注册 `pipelining_fn`)。

**这个 PR 修的是 4025 树上三个静默缺陷**,值得在正文里逐条给数字:

1. 块残差不跨 stage(无 PP `7.44679` vs pp2 `9.30017`,step 3,同批次)
2. 最终聚合在每段都跑
3. FQN 注入在 Config 树模型上直接 return 且不出声

**最强的一条证据**:`pp_stage_parity_4025.py` —— 按 core 的做法把模型切两段、
手工串起来,不经调度/loss/微批,`max_abs = 0.000e+00`。

**未完成**:DEP 未接线未测(文件已搬)。

---

## PR-EP:expert parallel(不含 MoonEP)

| commit | |
|---|---|
| `9f24178d9` | 稀疏 FSDP mesh、`ep_degree`、`model.parallelize` 调用 |
| `995f4b151` | 根级与 MoE 的声明(用上游 helper) |
| `95f5bd8da` | token dispatcher 切换 |

**文件**:`model.py`(`_set_sharding_config`)、`parallelize.py`、`moe.py`。

**正文的核心**:`LocalTokenDispatcher` 只在 rank 内重排,交给专家的是全局每专家计数;
专家权重按 E 切分后,grouped GEMM 拿到 32 个 offsets 对 16 个本地专家,报
`matrix batch sizes have to match` —— 读起来像模型的形状 bug。换成
`AllToAllTokenDispatcher` 后:16 对 16,两个 rank 收到 1030 / 1018 个不同的 token,
**step-1 loss 与 dp2 基线五位全同**。

**与 PR-TP 的耦合**:`995f4b151` 的声明是 EP 和 TP 共用的。EP 先发,TP 后面 rebase。

---

## PR-TP:暂停

| commit | |
|---|---|
| `3728ef835` | MLA/FFN/KDA 的声明 |
| `55acd725c` / `109b631ec` | 阻塞点的记录 |

留在树上、挡在 `NotImplementedError` 后。恢复时**从成因出发**:
`KimiLatentMoE` 重写了 `MoE.forward`,把路由的整数簿记放进模块级 DTensor 环境;
参照实现是组合 core 的 MoE、只在边界碰 DTensor。

---

## 冲突面

`model.py` 和 `parallelize.py` 被三个分支都碰,但落点不同:

| 文件 | CP | PP | EP |
|---|---|---|---|
| `parallelize.py` | `apply_cp_kimi_k3` + 删 unsupported 一行 | `pp_enabled` 传参 + 删一行 | mesh/ep_degree/parallelize + 删一行 |
| `model.py` | MLA Ulysses、视觉 scatter、CP 前提 | 块残差 stage 载荷、head 守卫 | `_set_sharding_config` |

**唯一的真冲突是 `unsupported_parallelisms` 那个 list,三个分支各删一行。**
一分钟的手解。

---

## 顺序

1. flavor 小 PR(text + LoRA)
2. **CP 与 PP 并行** —— 各自主体在独立文件里
3. EP
4. TP(解除暂停后)
