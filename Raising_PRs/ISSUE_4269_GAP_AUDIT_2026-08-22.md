# 对 upstream issue 4269 的差距审计(2026-08-22)

上游开了两个 K3 tracking issue,其中 **4269**(MXFP4-weight / MXFP8-activation QAT
与 checkpoint 互操作)当前**无人认领**,而它的 scope 与我们已有的实现高度重叠。

这份文档逐条核对 **spec 的要求 vs 我们代码里真实的样子**,每条给文件与行号。
我们的实现写在 spec 之前,所以"重叠"不等于"满足",下面区分开。

## 结论速览

| # | spec 里程碑 | 我们 | 性质 |
|---|---|---|---|
| 1 | Freeze checkpoint format against compressed-tensors | ❌ | 缺口,Gate 1 |
| 2 | Training primitives with STE backward | ✅ | spec 本身要的就是 fake-quant,见第 6 节 |
| 3 | Integrate QAT without modifying expert implementations | ✅ **Gate 已跑,三条全过** | 可直接声称 |
| 4 | Packed checkpoint import / export | ⚠️ import 有且流式;**export 明确未实现** | 半 |
| 5 | Validate across FSDP / EP / mixed topologies | ⚠️ FSDP / TP / EP+TP 全过;**零 token 专家、ragged routing、DCP resume 未测** | 半 |

关于第 2 条要更正一句:早先版本写"我们是 fake-quant,需坦白 scope"。读完完整 spec 后
这句不成立 —— spec 原文就是 "Stateless 1x32 MXFP4 weight **fake quantization**" 加
"Forward pass derives MXFP4 **quantize/dequantize view**",权重侧要的正是我们做的。
真正短的是激活侧的 grouped GEMM,而那条 spec 明写归 torchao,且 "emulated" 可接受。

## 1. 量化集的来源:抄下来的,不是解析出来的

spec 明确 "Do not hard-code w1/w2/w3"。

`torchtitan/models/kimi_k3/quant_scope.py` 的 docstring 说明来源是 released
`quantization_config`(:9-24),但代码里落成一个常量:

    # Verbatim from the released config's quantization_config.ignore. Kept as the
    OFFICIAL_IGNORE_PATTERNS: tuple[str, ...] = (...)      # :42-45
    _EXTRA_IGNORE_PATTERNS: tuple[str, ...] = (...)        # :57

我们没有硬编码 `w1/w2/w3`,但硬编码了 ignore 列表 —— **同一类问题,只是换了一端**。
spec 要的是从 pinned HF `quantization_config` + safetensors index **解析**。

**要补的活**:把 `OFFICIAL_IGNORE_PATTERNS` 换成解析器,常量降级为 fallback 或测试基准。

## 2. codec 与 compressed-tensors 的字节兼容:没有对照

Gate 1 原话是 "packed codes and E8M0 scales match the reference serializer"。

`torchtitan/models/kimi_k3/packed_mxfp4.py` 有 `quantize_mxfp4`(:122)、
`dequantize_mxfp4`(:68)、`load_packed_experts`(:159),测试是
`tests/test_packed_mxfp4_load.py`。

全仓 grep `compressed_tensors` / `compressed-tensors` **零命中**。

**也就是说我们对着自己的实现测往返,从未对着 reference serializer 测字节。**
往返自洽不能证明字节兼容 —— 一个自洽但 layout 不同的编码同样能往返成功。

**要补的活**:引入 compressed-tensors 作为测试期依赖,对同一权重比较 packed codes
与 E8M0 scale 的字节。

## 3. 分片导入不全聚:有,而且是我们的强项

spec 第 3 条要 "initialize sharded BF16 masters without gathering all experts on
one rank"。896 专家 x 92 层不可能单 rank 聚合。

`phase13_k3like_48b_posttrain/stream_quantize_mxfp4_dcp.py` docstring 原文:

> Streams tensor-by-tensor: reads each key individually from the source DCP
> checkpoint (peak RAM ~= one tensor + the growing output dict of PACKED tensors)
> ... **no rank ever materializes the full bf16 model**

配合 `lora.py` 的 meta packed-layout 构建(`quantize_base_mxfp4` 的 meta 分支,
:186-209),这是完整的 quantize-then-shard 路径。

**这条可以直接声称,而且是里程碑里最难的一条。**

## 4. 反向导出:没有,而且代码明确拒绝

`torchtitan/models/kimi_k3/state_dict_adapter.py:144` `_check_not_packed` 直接 raise:

    "HF checkpoint contains quantized/packed tensors ...; the MXFP4/packed unpack
     path is not implemented yet. Refusing to silently treat packed weights as
     ordinary values."

只有 bf16 -> packed 的正向(`stream_quantize_mxfp4_dcp.py`)。
**DCP -> HF packed safetensors 的 repack 完全没有。**

这个 raise 本身是对的(K3_RELEASE_IMPACT 的检查表里就写着"never treat packed
weights as plain tensors"),但它意味着 export 是从零开始的新活,不是补个分支。

## 5. TorchAO 的分工线:不争

spec 把 E2M1 fake-quant + STE + MXFP8 grouped-GEMM 划给 torchao。

姿势:**titan 侧的 converter / FQN / checkpoint 由我们做;torchao 缺的 primitive
我们可以去那边提**。那是第二个仓的 credit,不是让地盘。

## 6. MXFP8 激活:是 fake-quant,必须主动说

`torchtitan/models/kimi_k3/mxfp4_qat.py` 的 docstring 第 7-12 行原文:

> MXFP4 (weight) + MXFP8 (activation) **fake-quant** QAT for Kimi K3.
> ... **EMULATED** fake-quant so QAT runs on any GPU (**fake-quant is bf16 compute**)

`_ACT_ELEM = torch.float8_e4m3fn`(:36)是 MXFP8 的 elem dtype,但走的是
`_fake_quant_mx`(:60)—— quant 后立刻 dequant,反向 STE(:83)。
**没有真的 MXFP8 grouped-GEMM。**

所以我们能声称的是"MXFP8 激活的 QAT 语义",不是"MXFP8 激活的 kernel"。
reviewer 自己发现这一点的代价,远大于我们主动写一句。

## 验证覆盖:这次才第一次跑

这四样(MXFP4 QAT / packed 导入 / MXFP8 激活 / quantile)**都不在 58 格 gate 里** ——
gate 三臂的 flavor 是 `mini_block_attn_res` / `report_arch` / `report_arch_lora`,
没有一个开 `mxfp4_qat` 或 quantile。上一次矩阵级验证停在 2026-08-05 的旧树。

2026-08-22 补跑(10 步,`--debug.seed 42 --debug.deterministic`,4 卡):

| 格 | dp4 | dp2+tp2 |
|---|---|---|
| `kimi_k3_mini_qat_mxfp4` | ✅ | ✅ |
| `kimi_k3_debugmodel_report_arch_qat` | — | ✅ |
| `kimi_k3_debugmodel_gated_qlora_mxfp4`(packed) | ✅ | ❌ |
| `kimi_k3_mini_quantile_balance` | ✅ | ❌ -> 修复后 ✅ |

两个失败都报 `aten.add.Tensor got mixed torch.Tensor and DTensor`,但**是两个不同的
缺陷**,只是错误信息相同:

* **quantile**(`quantile_balance.py:46`,前向 hook):router gate 迁到声明式后输出是
  DTensor,而 `expert_bias_E` 在同一行被 `to_local()`,两者在 `topk_with_cutoff` 的
  加法里相撞。实测 `scores=DTensor(Replicate())`,每个 rank 持有全部 token,所以
  对称地 `to_local()` 即可,**已修并验证**(dp2+tp2 与 ep2+dp2+tp2 均通过,
  dp4 数值不变)。
* **packed-MXFP4**(`attn_res_model.py:269`):`KimiMoE.forward` 主动把输出拆成 plain
  ("this model's boundary convention is plain tensors",`model.py:2080-2091`),
  之后 `latent.from_latent` 与 `+ shared_experts(x)` 又把它变回 DTensor ——
  **但那是偶然的**,取决于最后一个算子恰好是 DTensor 算子。packed 路径把权重
  dequant 成本地张量,于是整条链保持 plain,撞上 DTensor 的残差流。
  **在 `afc3e4287`(TP 迁移之前)上复现,同样失败 —— 是既有缺陷,不是迁移引入的。**

## 里程碑 3 的 Gate:已跑,三条全过(2026-08-22)

spec 的 Gate 3 原话是 "200-step debug run has finite loss/gradients, decreasing
tail loss, no unexpected parameters/scales in DCP state"。此前从未跑过。

`kimi_k3_mini_qat_mxfp4`,dp2+tp2,200 步,`--debug.seed 42 --debug.deterministic`:

| 判据 | 结果 |
|---|---|
| 有限 loss / 梯度 | 全程无 nan/inf |
| 尾部 loss 下降 | 7.70829 -> 2.00759,单调 |
| DCP 无意外参数/scale | 9525 个 key 里 qdata/scale 命中 0 |

第三条在 QAT flavor 上"命中 0"是意料之中(fake-quant 不产生 packed 参数),
所以又在真正有 packed 参数的 `kimi_k3_debugmodel_gated_qlora_mxfp4` 上验了一次
(dp2+tp2,12 步,interval 5 以取到中途 checkpoint):

| | |
|---|---|
| 中途 checkpoint 内容 | `train_state` / `optimizer`(615)/ `dataloader` / `lr_scheduler` + 模型 |
| packed/scale 在 **model** state | 14 个(它们是模型的存储形态,应该在) |
| packed/scale 在 **optimizer** state | **0** |
| optimizer 的实际内容 | 210 个 LoRA adapter + AttnRes 的 alpha,别无其他 |

spec 的 "Packed weights and E8M0 scales materialize only for import/export,
never installed as optimizer parameters" 满足。optimizer 里那批非 LoRA 的条目是
`*_res_alpha`,即 `lora.py` 记的 alpha-fullparam exception,是设计上就该训的。

(最后一步的 checkpoint 是 model-only,那是 torchtitan 的 `last_save_model_only`
默认行为,不是缺陷 —— 第一次查的时候我拿了 step-10 这个末步存档,看到
optimizer 为 0 差点当成缺失。)

## 崩溃修复:六个,全部修完(2026-08-22)

补验第一次就把这条路径上串着的六个缺陷全暴露了,逐个剥出来:

| # | 缺陷 | 位置 |
|---|---|---|
| 1 | router gate 输出 DTensor 撞未拆的 bias | `quantile_balance.py` |
| 2 | 无 latent MoE 时输出 plain 撞 DTensor 残差流 | `KimiMoE.forward` |
| 3 | 事后 `from_local` 与内部 `to_local` 的反向错位 | 第一次修法本身 |
| 4 | gated 变体:alpha 拆本地使梯度变成 DTensor | `_scalar_local` |
| 5 | rowwise packed base 收到 Replicate 输入 | `_forward_packed_tp` |
| 6 | Replicate base 被当 rowwise 注册,packed 权重被错切 | `_register_lora_tp`(自己引入) |

**六个全是崩溃,不是静默错误** —— 没有任何一次训练因它们产出过错误数字。
这与词表那个缺陷性质相反,后者是静默的。

因为 QAT 是 fake-quant(bf16 计算),这六个全在 layout / placement 层,
碰不到数值语义。**真正能让数字静默错掉的只有 Gate 1(字节对照),而那一条我们没有。**

验证:58 格 0 挂且与上轮逐格逐位相同;429 单测;9 格特性矩阵全过,
其中 6 个回归格逐位不变。

## 里程碑 5 的一部分:ragged routing 已验,零 token 未达(2026-08-22)

Gate 5 要 "Exercise FSDP, EP, zero-token experts, ragged routing, mixed ranks"。

用一个临时前向 hook 统计 router 的 top-k 分配(`kimi_k3_mini_qat_mxfp4`,
dp_shard 4 + ep2,4 卡):

| seq_len | 每专家 token 数 | 零 token 专家 |
|---|---|---|
| 4096 | min 519 / max 1231(2.4x 不均衡) | 0 |
| 128 | min 17 / max 51(3.0x 不均衡) | 0 |

**ragged routing 覆盖到了** —— 分配显著不均且训练照常。

**零 token 专家没能自然触发。** 继续压小 token 数会撞上 `KDA training requires
chunk mode (T > 64)`,所以这个 flavor 上做不到:8 个专家 top-2,可行的最小规模仍给
每个专家几十个 token。

要覆盖它得**人为把 router 偏置成某个专家永不被选**。那是有意义的测试,但形态是单测
(构造路由后驱动一次 MoE 前向),不是训练矩阵里的一格。记为待办,不要写成"已验证"。

同样未测的:mixed image/text rank batches、DCP resume 后与不中断运行的一致性。
前者 58 格的多模态臂间接覆盖了一部分,但不是 Gate 5 说的那种刻意构造。

## 认领建议

4269 无人认领,而第 3 条(分片不全聚)是里程碑里最难的一条且我们已经做完。
**先认领并说明覆盖与缺口,而不是直接发 PR** —— 认领成本低,能避免别人重复做第 3 条;
而直接发 PR 会在 Gate 1(字节兼容)和第 4 条(export)上立刻被问住。

不需要等 4025:4269 是独立 issue,我们的 QAT 是 wrapper 形式,不依赖 4025 的模型结构。
