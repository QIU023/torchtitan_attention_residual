# LoRA / QLoRA 并行矩阵与 QAT 落地(2026-08-28)

判据见 `EVIDENCE_METHOD_2026-08-25.md`。树:新树 `main` @ `a1dd321de`。
flavor 均为多模态 debug;batch 4096,微批 512/dp rank,seq 512,seed 42,
单 seed checkpoint,预热读第二遍。矩阵目录 `mx3_lora_mm_0828_*` /
`mx3_qlora_mm_0828_*`。

## 一、LoRA 表(`kimi_k3_debugmodel_lora`,4/4 通过)

| cell | world | s1 | s3 | s10 |
|---|---|---|---|---|
| dp1 | 1 | 12.45474 | 11.94849 | 10.71402 |
| dp2 | 2 | 12.46697 | 11.82895 | 10.53144 |
| tp2 | 2 | 12.45324 | 11.93854 | 10.72911 |
| fsdp2×tp2 | 4 | 12.46155 | 11.89166 | 10.47442 |

- 缓降形态(12.45→10.7/10 步)是 LoRA 只训适配器的预期。
- **tp2 格的发现**:上游 `_lora_adapter_sharding` 只认 colwise/rowwise,
  K3 的 TP-invariant 基(`_tp_replicate_config`:wq_a/wkv_a/routed_down/
  routed_up)命中 rowwise 断言。修复 = 第三案:复制基 → 双适配器同复制
  (`a418226cd`)。修后 tp2 与 dp1 的 s1 差 1.5e-3,s3 差 1.0e-2,家族正常。

## 二、QLoRA 表(`kimi_k3_debugmodel_qlora_mxfp4`,2 通过 + 2 设计性守卫)

| cell | world | 判定 | s1 | s3 | s10 |
|---|---|---|---|---|---|
| dp1 | 1 | 通过 | 12.45288 | 11.96454 | 10.68903 |
| dp2 | 2 | 通过 | 12.45138 | 11.87566 | 10.37547 |
| tp2 | 2 | 设计性守卫 | - | - | - |
| fsdp2×tp2 | 4 | 设计性守卫 | - | - | - |

- 与 LoRA 同 seed 轨迹贴近(dp1 s3:11.965 vs 11.948),QLoRA 有损设计内。
- TP 格死于 `NotImplementedError: quantize_base='mxfp4' does not support a
  TP-sharded base yet`——守卫是 `a6a0a9c10` 立的:声明系统要求打包对有
  placement;复制基 → 打包对同复制(可组合),colwise/rowwise 基 → 显式拒绝,
  因为 packed-TP 前向(本地反量化+本地矩乘+声明蕴含的集合)是独立单元
  (老树 `_forward_packed_tp` 的移植面)。未声明会在后续 placement 检查炸,
  或更糟——在合同期望分片处算出全量输出。
- 守卫之前的原始失败(`LoRALinear.base_qdata has no placement declared`)
  保留在矩阵日志里。

## 三、QAT(`kimi_k3_debugmodel_mx_qat`,`a1dd321de`)

老树 mx_qat 三件移植为新树转换器 `MXFP4QATConverter`
(`components/quantization/`,与 float8/mxfp8/nvfp4 同家族):

- **官方 scope 即 isinstance**:释出 quantization_config 的 ignore 列表剔到
  只剩 routed experts;新树里那恰是 `GroupedExperts.Config`,一个类型检查
  顶替了老树 quant_scope 的 100 行正则(老教训:名字表量化的恰是 K3 保持
  高精度的那套)。
- forward 窗口 `__dict__` 遮蔽(非 property——FSDP2 `reset_sharded_param`
  在 forward 外要真参数;非改名——state-dict 合同与 EP/TP 布局按原名索引)。
- 逐分片量化对 w2_EDF 的 scope 收窄(块尺度=块内 max-abs)照实警告。
- CPU 测试 4(STE 恒等梯度、不可分块直通、换类、窗口复原+梯度到主参数);
  dp2 实训 12.450→10.059,全参斜率远陡于 LoRA,形态正确。
- 与 QLoRA 的关系一句话:QAT=bf16 主参数+假量化(RFC 要的),QLoRA=真打包
  冻结基+适配器(我们的省显存路),同权重上不组合。

## 四、RFC(上游 K3 MXFP4/MXFP8 QAT issue)对照更新

| 里程碑 | 状态 |
|---|---|
| ② 训练原语(假量化 STE) | **本次落地**(torchao MX 之上,titan 侧完整) |
| ③ QAT 进 recipe | **本次落地**(debug flavor;48B flavor 是配置级) |
| ① 冻结 checkpoint 格式 / ④ 打包 I/O | DCP 转换器可复用大半,HF packed 导入未做 |
| ⑤ 分布式验证 | 方法论现成;QAT 尚缺矩阵(只有 dp2 冒烟) |

可 raise 的形态:②+③ 为主体的 QAT 转换器 PR。**未经用户确认不提交**;
正文按 FILING.md 规则另起草。

## 五、遗留

- packed TP(mxfp4 基 × colwise/rowwise):守卫已立,移植面=老树
  `apply_packed_mxfp4_tp`/`_forward_packed_tp`,待需求行使。
- QAT 并行矩阵(含 EP——w2 逐分片 scope 收窄该有一行记录)。
- NF4 grouped experts:被 mxfp4 取代,不做。
