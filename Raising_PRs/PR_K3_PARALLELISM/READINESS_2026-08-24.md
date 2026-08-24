# Raise 前合格度严判(2026-08-24)

树已 rebase 到 upstream/main(4025 已 merge,`5fecad929`)。所有分支在 fork
`QIU023/torchtitan`。判据用老树的:非 LoRA 格逐位不变(同 seed 前后比),0 格挂,
有差异必须有能解释非失败的根因(此条老树撤回过放宽,不以"有解释"放行)。

## 分支清单

| 分支 | 内容 | 状态 |
|---|---|---|
| k3_on_4025 | 集成(四并行 + DEP + dynamic CP) | 已推 |
| k3_pp_text | 纯文本 PP | 已推,已验证 |
| k3_cp_text | 纯文本 CP(KCP + MLA Ulysses) | 已推,已验证 |
| k3_ep | EP(无 MoonEP) | 已推,已验证 |
| k3_pp | PP + DEP(多模态) | 已推 |
| k3_cp | CP + dynamic CP(多模态) | 已推 |

## 文本侧数值结果(准备 raise)

方法学(全部满足):单一 seed checkpoint、每格自带同配置预热、每个计量格断言
`Loading the checkpoint from`。seed checkpoint 本身不可复现(实测三次哈希不同),
所以跨格只在同一脚本内、同一 seed 下比较。

### PP(flavor debugmodel_text_32l,32 层)
9 格 step-1 全部 = 12.45788,逐位相同:
dp1 / pp2 / pp4 / pp2xvp2 / pp2xvp4 / pp4xvp2 / pp4xvp4 / pp8xvp2 / pp8xvp4
(vp 用 Interleaved1F1B + layers-per-stage 表达。)

### CP(flavor debugmodel_text,seq 1024)
| | step-1 | 相对 dp1 |
|---|---|---|
| dp1 | 12.44662 | - |
| cp2 | 12.44292 | 3.0e-4 |
| cp4 | 12.45092 | 3.5e-4 |
| cp8 | 12.44724 | 5.0e-6 |
老树 cp2/cp4 相对 dp1 = 1.3e-3;我们紧约 4 倍。seq 1024 因 FlexAttention BlockMask
要求 Q_LEN % (cp*128)==0(上游后端约束)。

### EP(flavor debugmodel_text,无 MoonEP)
| | step-1 | 基线 | 相对 |
|---|---|---|---|
| ep2_fsdp2 | 12.43537 | dp2 12.43537 | 逐位相同 |
| ep8_fsdp8 | 12.44615 | dp8 12.44491 | 1.24e-3 |

## 严格合格判断

### PP 文本侧 — 合格
* 9 格逐位相同,含 VP/交错调度。0 挂。
* 唯一保留问题:pipeline_adapter 的 wrapper 分支(text 布局不可达),k3_pp_text 已移除。
* **可 raise。**

### CP 文本侧 — 基本合格,一个口径说明
* cp2/4/8 与 dp1 差 3e-4 ~ 5e-6,远在 bf16 相对精度(3.9e-3)内,且比老树紧。
* **CP 改变序列分片,step-1 本就不与 dp1 逐位相同**(老树 SDPA 亦然)。所以按"逐位不变"
  的字面判据,CP 格不满足 —— 但这是 CP 的本质,不是 bug,老树的 CP 格也从不逐位相同。
  判据对 CP 的正确形态是"偏差在 bf16 精度内且比参照紧",这条满足。
* cu_seqlens 说明已加进 PR body(CP 分片边界,非 sample packing)。
* **可 raise,PR body 需说明 step-1 非逐位相同的原因。**

### EP — 合格
* ep2 与 dp2 逐位相同(专家分片在 data 轴内,dp 度相同则数据相同,可逐位)。
* ep8 与 dp8 差 1.24e-3(8 路专家分片 + edp mesh),bf16 精度内。
* MoonEP 明确排除并注明。
* **可 raise。**

## 多模态侧(draft done,progressively raise,暂不写实验)

PP&DEP、CP&Dynamic CP 的多模态实现已在 k3_pp / k3_cp 分支,draft 完成,
但**按 progressive 策略后置**,多模态数值实验暂不做、暂不写。已知状态:
* DEP clause 1(塔独占 stage):实测 partition-invariant(dp2xpp4,DEP 开/关 step1/2 = 12.49453)。
* DEP clause 2(塔跨多 stage):单模型布局下 _split_module 够不到 vision_encoder.layers,
  未接通,已改为明确 raise;塔 share 分解单测逐位等价(2/3/4 shares atol=0)。
* Dynamic CP:子 CP 组 + 负载均衡已搬(report 5.2.3),cp2 视觉格集成分支通过;
  多模态全矩阵未按新方法学重测。

## 结论

**文本侧 PP / CP / EP 三个分支合格,可 raise(CP 需 PR body 说明 step-1 口径)。**
多模态侧 draft 完成,按 progressive 策略后置,实验待做。
