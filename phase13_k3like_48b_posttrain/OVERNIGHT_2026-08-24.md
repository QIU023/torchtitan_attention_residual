# Overnight 2026-08-24：审计修复 + 判据 + 不埋雷

用户要求:四个分支 PR 蕴含的代码必须干净正确、通用,不埋雷(TP 本身除外)。
按序执行,一次一个 GPU 作业,每个数值判定对照老树。

## 任务

1. [进行中] CP 两个 TP-only bug(CP PR 根本正确性)
   - MLA Ulysses 用全局 n_heads,应从投影宽度推 TP-local(老树 model.py:725)
   - k_rope 丢了 to_local_partial_grad(老树 model.py:847;dtensor_ops.py:52 记录 1-6% 梯度误差)
2. DEP R3:单模型布局下 KimiK3ViTStage 全不可达,clause 2(n_vit>1)未真正接通 —— 通用化或删死代码+标注
3. DEP 判决:dep_off vs nvit1 是缓存/噪声还是真 bug
   - 中间信号:dep_off 同格两跑 12.59604/12.59680(差 7.6e-4)-> 8卡本身有 ~1e-3 噪声
4. 10 步基线两遍暖跑,归档 gate_logs/
5. lora_pp8(pp8 某 stage 无 LoRA 参数)
6. 收尾:审计定稿、推送、更新 PR 文件夹

## 进度日志

## 进度(截至 09:50)

### 已完成并推送

* **零初始化修复** `d54d327a9`:AttnRes 三处 `res_proj` 用了 `trunc_normal`,
  而 `attn_res.py` 契约要求 "MUST be zero-initialized"。老树 `init_weights` 显式零初始化。
  已修并验证(init_weights 后 48 个 res_proj 全零)。
* **CP 两个 TP-only 修复** `685ae1110`:
  - MLA 从投影宽度推 TP-local head 数(老树 `model.py:725`),不再用全局 `n_heads`
  - Ulysses 的 k_rope 加回 `to_local_partial_grad`(老树 `model.py:847`;丢了在 tp2×cp2 有 1-6% 梯度误差)
  - 两者在 tp=1(唯一可达路径)是构造性 no-op;cp2 实跑 12.38074 无回归

### 确认的真问题

**DEP clause-1 数值差(0.17),已排除缓存**:

| | step1 (r1 / r2) |
|---|---|
| dep_off | 12.59604 / 12.59680 |
| n_vit=1 | 12.42496 / 12.42380 |

每格自身一致到 ~1e-3(8 卡噪声),两格差 **0.17**,大两个数量级,且 r1/r2 都在。
每格独立 cache、r2 为暖跑。**PP/DEP 审计 agent 的"partition-invariant 静态证明"被实测推翻。**
未定位。

**DEP R3(死代码)**:单模型布局下 `_unwrap_multimodal_for_pp` 因无 `language_model`
子节点提前返回(`pipeline_adapter.py:1100-1106`),所以搬来的 `KimiK3ViTStage` /
`promote` / `set_dep_role` / `_install_vision_stage_wiring` **全部不可达**。
n_vit=1 走的是普通 `KimiK3Model.forward`(FQN 注入把塔放独立 stage);
**n_vit>1(clause 2)在这个布局下没有真正接通**。
所以"share 分解测试 atol=0 通过"只证明塔的方法本身正确,pipeline_adapter 从不调用它们。

### 两条被更正的判断(都是我的脚手架问题,不是代码 bug)

1. **磁盘看门狗**:`disk_watchdog.sh` 在剩余空间低于 40G 时清理并杀进程。
   今晚多次"莫名 SIGTERM"(`mm_dp1pp4_off`、`fsdp2_pp2`)源于此。已清理到 96G 空闲。
2. **mm dp1×pp4 "死锁"是我自己造成的**:bisect 脚本给每格全新 inductor cache,
   4 卡冷编译超过 300 秒 NCCL watchdog(`OpType=COALESCED, NumelIn=0`,ran for 387890 ms),
   rank 间失步。**与今晚早些时候统一冷方案里 `text_pp4` 的失败签名完全一致** ——
   我重新引入了自己已诊断并否决的模式。三次二分因此全部无效,mm dp1×pp4 无真 bug
   (门里共享暖 cache 时通过,12.57409)。
   **纪律:多卡格永远不要用冷 cache 测量;要么预热,要么调大 NCCL 超时。**

### 待办(按序)

1. DEP clause-1 0.17 定位(需暖 cache 的干净对照)
2. DEP R3:接通 clause 2 或删死代码并标注
3. 10 步基线归档
4. lora_pp8
5. 审计定稿

### 更正:DEP clause-1 没有 bug(我错了三次)

用门的方法学(**共享**暖 cache、每格跑两次取第二次)重测:

| | step1 | step2 | step3 |
|---|---|---|---|
| off_w(冷) | 12.49852 | 12.49852 | 9.93466 |
| **off**(暖) | **12.49453** | **12.49453** | 9.83826 |
| on_w(暖) | 12.49453 | 12.49453 | 9.82314 |
| **on**(暖) | **12.49453** | **12.49453** | 9.82314 |

**`off` 与 `on` 在 step1/step2 完全相同。DEP clause-1 是 partition-invariant 的。**
PP/DEP 审计 agent 的静态证明是对的,我先前"实测推翻它"的结论作废。

我的错误推理:先前用**每格独立** cache 得到 0.17,并以"r1 和 r2 都有这个差"论证它真实。
但 r1/r2 共享该格自己的 cache,当然一致 —— **每格独立 cache 不能跨格拉平**,
这正是我今晚早些时候自己写进本文档的"混合状态"问题。同一个坑我踩了三次:
统一冷方案、bisect 脚本、DEP 判决。

step3 仍差 ~1.5e-2,但同配置内 off_w vs off 在 step3 就差 1e-1 —— 8 卡 step3 散布本身很大,
有意义的信号是 step1/2 的严格相同。

**方法学定论(写死):跨格数值比较只有一种合法方式 —— 单一共享 cache,
全矩阵跑两遍,比较第二遍。任何"每格独立 cache"都不构成跨格可比性。**

### 结论更新

* DEP clause-1(塔独占 stage):**通过**,partition-invariant
* DEP clause-2(n_vit>1):**未接通**(R3 死代码),仍待处理

### lora_pp8:结构性限制,非回归(已归因)

`ValueError: Optimizer param_groups pattern '.*' matched no parameters`。

K3 的注意力是 3 KDA : 1 MLA 交替,debugmodel 24 层的 MLA 在 `{3,7,11,15,19,23}`。
LoRA 目标 `wq_b / wkv_b / wo` 是 **MLA 专有**(KDA 没有这些投影)。pp8 把 24 层切成
约 3 层一段后:

| stage | layers | MLA |
|---|---|---|
| 0 | 0,1,2 | **无** |
| 1 | 3,4,5,6 | 3 |
| 2 | 7,8,9 | 7 |
| 3 | 10,11,12 | 11 |
| 4 | 13,14,15 | 15 |
| 5 | 16,17,18 | **无** |
| 6 | 19,20,21 | 19 |
| 7 | 22,23 | 23 |

stage 0 与 stage 5 整段没有 MLA 层 -> 零 LoRA 可训练参数 -> 优化器空参数组。

**与老树同一类限制**:老树 `LORA_DEP_2026-08-11.md` 记录 LoRA+DEP 下八个 PP 格全挂在
同一个错(DEP 的视觉 stage 只有塔+embedding,都不是 LoRA 目标),用对照法归因
(pp2 DEP 开=挂,关=训练正常),**并未修复**,作为已知限制留存。

差别:新树这个**不需要 DEP** 就会发生,取决于 pp 度与 3:1 混合模式的对齐 ——
pp2/pp4 每段都含 MLA 所以通过,pp8 段太薄就会出现纯 KDA 段。

**判定:不是迁移回归,是 LoRA x 高 PP 度 x 3:1 混合注意力的结构性约束。**
不在新树自行"修复"(会偏离老树);记录为已知限制,LoRA 臂按老树的配对规则判定。
