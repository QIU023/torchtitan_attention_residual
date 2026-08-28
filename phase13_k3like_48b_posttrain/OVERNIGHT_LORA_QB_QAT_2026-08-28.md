# Overnight 收口:LoRA 全矩阵、QB/QAT 证据、三 PR 分支(2026-08-28 夜)

判据 `EVIDENCE_METHOD_2026-08-25.md`。集成树 `k3_on_4025` @ `04f73f44c`。
三 PR 分支(上游 base 30eb5e502 + 增量):`k3_qb`(d5393c240)、
`k3_qat`(6d79d0d20)、`k3_lora_extras`(fb5ee6ae9,含两笔 EP 修复)。
三份 body:`PR_BODY_QB/QAT/LORA.md`,证据已全数回填。

## 一、LoRA 集成十格(mx3_lora_full_0828_065851)

dp1/dp2/dp4/tp2/tp4/fsdp2_tp2/cp2/fsdp2_cp2/dp2_ep2/dp4_ep4 全通,
数值见 PR_BODY_LORA。**LoRA×CP、LoRA×EP 均为本树首验即过**。
- lora×pp 双格 = 设计限制:LoRA 冻结下塔段零可训参数,优化器空匹配
  (recipe docstring 的既有预言);vp 形态亦然(vit 段吃满段预算)。

## 二、QLoRA(mx3_qlora_mm / mx3_qlora_ext / mx3_qlora_lin)

dp1/dp2/dp4 + dp2_ep2/dp4_ep4 通过;EP 行与 dp 孪生行 s1 逐位同值
(12.45138 / 12.47301)——同 seed 重造确定性 + EP 前向透明双证。
探针链修了三站:专家反量化 full_tensor→to_local(98fc734c4)、
view 前导维 -1 自导出(61e07c151)、此前的打包对 placement 声明。
- 全量 flavor × TP = 设计守卫(专家内维分片,2-D 展平不可表达);
  linears-only flavor tp2 矩阵行 12.54449/11.97425/10.69844 通过。
- 磁盘事故记录:matrix 链 stdout 全镜像入任务输出文件曾把 /tmp 灌满,
  ENOSPC 波及 harness 自身;证据文件无损,seed 由确定性重造恢复。

## 三、QB / QAT 集成四格表(mx3_qb_int2 / mx3_qat_int)

均 dp1/dp2/dp2_ep2/dp4_ep4 全通;QAT 的 dp2 与 dp2_ep2 s1 同值
(12.50530),QB 的近同(12.49584/12.49474)。数值见各 body。

## 四、分支表与结构性注记

分支基于上游 base,K3 的 TP/EP 门尚未由未合的 TP/EP PR 拆除——分支格
按构造仅 dp 族;TP/EP 行由集成表供给,body 中来源分开标注,并预答
"分支上为何带惰性 TP 代码"(适配器分片从 sharding_config 推导,上游
现为 None → 行为不变)。

## 五、PR 依赖结论

QB/QAT/LoRA 三 PR 与 EP PR 相互独立(组件层 + registry 级重叠),
可即提;顺序只受评审注意力约束。未经用户确认不提交。
