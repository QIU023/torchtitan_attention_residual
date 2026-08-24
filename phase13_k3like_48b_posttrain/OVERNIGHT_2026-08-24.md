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
