# attn-gym FLA 后端(2026-08-29,本机 5060Ti 全程验证)

背景:PR-4313 reviewer(drisspg,attn-gym owner)自提"unsupported arch 可
dispatch 到 FLA";用户定调走此路(lane A),三后端愿景 = fused(SM100)/
**fla(triton,任意卡)**/ reference(纯 torch)。fla 作可选依赖有 titan
自身先例(qwen3_5/gdn.py)。

## 实现(本地分支 kda-fla-backend @4000b0b,+81 行,待 fork 后推)

- `Impl` 枚举加 `FLA`;api 分派加支路;`fla_backend.py` 惰性导入适配器;
- 约定咬合:fla 的三个 `use_*_in_kernel` 全关,吃 attn_gym 合同下的已
  预处理张量(l2norm/bound_gate/sigmoid 在外);gate 同为自然对数,无换底;
- 唯一语义差 = 递归态末两轴互为转置,适配器双向持有(进 initial_state
  转、出 final_state 转)。

## 配平证据(FLA vs REFERENCE,bf16 输入)

| 测项 | maxdiff |
|---|---|
| 前向输出 | 2.4e-4 |
| 终态(转置适配后;适配前 0.61,即转置证据) | 2.9e-3 |
| FLA 分半回环(半程终态续算 vs 整跑) | **0.0(位同)** |
| 跨 impl 交接:reference 终态 → FLA 后半程 | 1.2e-4 |
| 梯度 dq/dk/dv/dgate/dbeta | ≤7.5e-9 |
| varlen(cu_seqlens 双文档) | 2.4e-4 |

跨 impl 状态交接绿意味着:drisspg 的 CP 编排(状态复合)可以在 per-rank
用 FLA 而复合层不感知——eager 两段式数学(#41 慢车道)的必要性进一步下降。

## 待办

fork QIU023/attention-gym(用户网页操作)→ 推分支 → 注册 logbook 子模块
→ 按 attn-gym 测试惯例补正式单测 → PR(body 引本表)。#4374 维持:titan
层旋钮与库内分派组合,后续 attn_gym 若吸收 auto 分派则瘦身。
