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

### 更正:lora_pp8 是迁移 bug,不是结构性约束

上一节判定"结构性约束、老树同类限制、不修"是**错的**。真正原因:

| | LoRA 目标 |
|---|---|
| 老树 `lora.py:37` `DEFAULT_LORA_TARGETS` | 15 个:MLA(`q_proj`/`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/`o_proj`)、`attn_gate_proj`、**dense FFN 与 shared experts(`gate_proj`/`up_proj`/`down_proj`)**、**latent MoE(`latent.down`/`latent.up`)** |
| 新树 `config_registry.py:174` | **3 个**:`["wq_b", "wkv_b", "wo"]`,全是 MLA 专有 |

老树把 FFN / MoE 投影也作为 LoRA 目标,而**每一层都有 FFN/MoE**,所以任何 PP 切分下
每个 stage 都有可训练参数 —— 这就是老树 pp8 能过的原因,与层数或 MLA 分布无关。

新树只挂 MLA 专有的三个,于是纯 KDA 段(pp8 的 stage 0 与 stage 5)零可训练参数。

而且新树 flavor 的 docstring 自称 "wo covers MLA and **output_proj covers KDA**",
但 `output_proj` **不在** target 列表里 —— **代码与自己的注释矛盾**,是我搬 LoRA flavor 时
漏搬了目标集。

**判定改为:迁移 bug(漏搬 LoRA 目标集),需按老树补齐。** 影响不止 pp8 ——
目标集缺失意味着 LoRA 臂**所有格**训练的参数子集都与老树不同,整个 LoRA 臂的数值
都不可与老树比对。

## 10 步基线结果:判据未达标,PP 跨格一致性退化

跑法:两遍共享暖 cache、10 步、取第二遍。冻结副本在 13:30-14:37 期间未被改动
(核实过 mtime),不含 LoRA 目标集修复(那次提交在基线启动之后),所以 `lora_pp8`
本轮失败不计入。归档 `gate_logs/gate_10step_2026-08-24_pass2.txt`。

**与 3 步那轮(修复前)相比,跨格一致性明显变差:**

| | 3 步(修复前) | 10 步(修复后) |
|---|---|---|
| text pp2 / pp4 / pp8 vs dp1 | **全部逐位相同** | pp2 相同;pp4 +2.0e-2;pp8 -4.2e-2 |
| mm pp2 / pp4 / pp8 vs dp1 | **全部逐位相同** | pp2 -1.8e-1;pp4 -8.9e-2;pp8 -1.4e-1 |
| mm cp2 vs dp1 | 3.0e-2 | -5.3e-2 |

text: dp1 12.44529,pp2 12.44529,pp4 12.46575,pp8 12.40336,cp2 12.38074
mm:   dp1 12.56760,pp2 12.39021,pp4 12.47896,pp8 12.43254,cp2 12.51413

**PP 从"三臂全部逐位相同"退化为差 1e-2 ~ 1e-1。** 期间唯一介入模型数值的改动是
**AttnRes 零初始化**(CP 的两处修复在 tp=1 下是构造性 no-op,已验证 cp2 无回归)。

一个待验证的机制假设:零初始化让块 softmax 权重初始均匀,AttnRes 退化为标准残差,
此时块残差各列数值等价,跨 stage 传递的舍入差异不再被非均匀权重压制。
**这是假设,未验证。** 也可能零初始化本身正确而暴露了 PP 块残差路径里原有的问题
—— 修复前非均匀权重可能掩盖了它。

另注:`mm warm_discard` rc=1 却打印了 loss(12.56760,与 mm dp1 同值),需要查。

**判定:判据不达标。** 不能以"有解释"放行 —— 老树把"有差异必须有根因"这条放宽
明确撤回过。PP 分支在这一项澄清前不应 raise。

**下一步(优先级最高)**:对照实验 —— 在零初始化修复前后各跑一次 text pp4,
其余条件全同,确认 PP 退化是否由它引起。

## 地基问题:seed checkpoint 不可复现(2026-08-24 15:4x)

同参数、同 `--debug.seed 42 --debug.deterministic`、`--checkpoint.create_seed_checkpoint`
连建三次,权重内容哈希互不相同:

    s1: 3147ff57e1edcfea
    s2: 11e74dbd6016a582
    s3: 57f1ccdff1a86894

**推翻今晚所有跨脚本比较。** 每个实验脚本各建自己的 seed,所以同一配置在不同脚本里
给出不同 loss(`text_dp1` 12.52573 vs 12.44529)是正常的,不是 bug。我此前把这类差异
当作证据用过,全部作废。

**脚本内部仍然有效** —— 同一脚本的所有格共享一个 seed。据此重读:

### PP:无精度问题(有效证据)

`zab` 脚本内,同 seed,每格自带预热:

| | dp1 | pp4 |
|---|---|---|
| 零初始化 ON | 12.44529 | **12.44529** |
| 零初始化 OFF | 12.58783 | **12.58783** |

两侧都逐位相同。10 步基线里 `pp4` 差 2.0e-2 是测量方法学问题(计量时该格图形状未预热),
不是代码。

### CP:有真实精度问题(有效证据)

`cp_precision` 脚本内,同 seed,每格自带预热:

    text_dp1  12.52573
    text_cp2  12.38074      差 1.4e-1

而 3 步那轮(零初始化与 CP 修复之前)text cp2 与 dp1 只差 9.3e-3。
**偏差从 9.3e-3 增至 1.4e-1**,期间介入的改动是 AttnRes 零初始化与 CP 的两处
TP-only 修复。需要 A/B 定位是哪一个,以及是"引入"还是"暴露"。

### 结论

* **PP 文本侧可以进 draft PR。**
* **CP 不可以** —— 不能把带 1.4e-1 偏差的实现写进 PR 说它对齐。

### 方法学(第五次栽在同一类问题上,写死)

1. 跨格比较必须**同一个 seed checkpoint**,即同一脚本内;跨脚本不可比。
2. 每格必须**自带同配置的预热格**;"整个矩阵跑两遍"不等价。
3. 每格独立 cache 不构成跨格可比性。
