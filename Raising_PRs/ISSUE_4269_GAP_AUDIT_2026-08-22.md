# 对 upstream issue 4269 的差距审计(2026-08-22)

上游开了两个 K3 tracking issue,其中 **4269**(MXFP4-weight / MXFP8-activation QAT
与 checkpoint 互操作)当前**无人认领**,而它的 scope 与我们已有的实现高度重叠。

这份文档逐条核对 **spec 的要求 vs 我们代码里真实的样子**,每条给文件与行号。
我们的实现写在 spec 之前,所以"重叠"不等于"满足",下面区分开。

## 结论速览

| # | spec 里程碑 | 我们 | 性质 |
|---|---|---|---|
| 1 | Freeze checkpoint format against compressed-tensors | ❌ | 缺口,Gate 1 |
| 2 | Training primitives with STE backward | ✅(emulated fake-quant) | 需坦白 scope |
| 3 | Integrate QAT without modifying expert implementations | ✅ | 可直接声称 |
| 4 | Packed checkpoint import / export | ⚠️ import 有且流式;**export 明确未实现** | 半 |
| 5 | Validate across FSDP / EP / mixed topologies | ⚠️ QAT 过 FSDP+TP;**packed x TP 挂** | 半 |

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

## 认领建议

4269 无人认领,而第 3 条(分片不全聚)是里程碑里最难的一条且我们已经做完。
**先认领并说明覆盖与缺口,而不是直接发 PR** —— 认领成本低,能避免别人重复做第 3 条;
而直接发 PR 会在 Gate 1(字节兼容)和第 4 条(export)上立刻被问住。

不需要等 4025:4269 是独立 issue,我们的 QAT 是 wrapper 形式,不依赖 4025 的模型结构。
