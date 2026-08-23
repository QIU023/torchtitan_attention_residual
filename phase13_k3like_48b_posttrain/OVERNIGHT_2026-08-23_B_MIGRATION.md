# Overnight:把四种并行迁到 4025 的树上(2026-08-23 起)

## 先回答"是不是开了新仓库"

**没有。**没有第二个 git 仓库,logbook 也没动。

新树是**同一个 submodule 仓库(`QIU023/torchtitan`)的一个 `git worktree`**:

| | |
|---|---|
| 路径 | `/workspace/tt_4025/torchtitan` |
| 分支 | `k3_on_4025`(新建) |
| 起点 | `dee45e357` |
| `.git` | 与 `/workspace/torchtitan_attention_residual/torchtitan` **同一个**,同 remote |
| 参照树 | `k3_tp_declarative` 原地不动,随时可对拷 |

`dee45e357` 来自 remote `pr4025`(PR 作者的 fork),**不在我们 origin 上**。
把 `k3_on_4025` 推到我们 fork 时会连带推上他们的 commit —— 是我们自己的 fork,
公开 commit,无副作用,但先说明。

资产目录做了软链 `/workspace/tt_4025/phase13_k3like_48b_posttrain`,
让测试的 `parents[5]` 能解析(497k 键覆盖测试依赖它)。

---

## 完成判据(用户定的,写死在这里)

**rebase 到 upstream/main 之后,已迁移的分支必须让 58 格里对应的并行轴全部通过。**

58 格按轴的分布(实测,数的是格名含该轴的):

| 轴 | 格数 |
|---|---|
| pp | 26 |
| tp | 21 |
| cp | 21 |
| ep | 15 |
| 总 | **58** |

(格子会叠加,所以合计大于 58。`dep2` 和 `pp8vp4` 两组在 mm_full 臂里,
按目录名而不是格名区分。)

四个迁移做完 + 我认为他们树上缺的部分补齐 = **新树的 58/58**。

判据本身不变:0 格挂掉;非 LoRA 格逐位不变(与新树自己的基线比,不是与旧树比);
有差异必须给出能解释没出问题那些格的根因;LoRA 臂按配对规则。

**新母树基线**(`kimi_k3_debugmodel`,dp2,3 步,seed 42 deterministic):

    step 1  loss 12.51502  grad_norm 24.0000
    step 2  loss 11.35441  grad_norm 30.7500
    step 3  loss  9.89706  grad_norm 18.6250

---

## rebase 的时机与形态

现在 `k3_on_4025` 只是 running,不 rebase。4025 merge 进 main 之后:

1. `git rebase upstream/main`(那时他们的 commit 已在 main 里,冲突面应该很小);
2. 跑新树的 58 格;
3. 按分支切出四个 PR。

`dee45e357` 目前距 upstream/main **8 个 commit**(我们旧树距 main 23 个)——
起点比旧树新。

---

## 顺序

用户定的:**先文本侧,再 vit,最后 LoRA**;分支按多模态划:

| 分支 | 内容 |
|---|---|
| **CP** | MLA Ulysses + KDA KCP + **vit dynamic CP** |
| **PP** | pipeline_adapter + **DEP** |
| **EP** | expert parallel,**不含 MoonEP** |
| **TP** | 声明式 TP,最后 |

---

## 起点上的两个实测发现

### 一、他们的树已经有 Block Attention Residuals

`model.py` 里有 `_apply_attention_residual`、`output_res_proj`、`output_res_norm`、
逐层累积的 `block_residual_TND`。我们那 997 行的 `attn_res_model.py`
(耦合度测量里最高的一个,39 处)**很大一部分是重复的**,不该照搬。

### 二、折叠 token 布局对 CP 是简化,不是障碍

他们整条链路是 `[T, D]`,**没有 batch 轴**;MLA 是 `[T, H, K]`,KDA 同。

* 我们的 `_forward_kcp` 之所以按 B 循环,是因为 fla 的 `causal_conv1d_cp`
  断言 `[1, T, D]`。折叠布局里 B 恒为 1,**那个循环整段消失**。
* Ulysses 的契约维度从 `(1, 2)` 变成 `(0, 1)`。

交接文档把 folded layout 列为"没读懂就没法估"的最大风险项。读懂之后,
它对 CP 是**减法**。对 PP/EP/TP 的影响还没测。

---

## 已完成

| 项 | 状态 |
|---|---|
| worktree + 资产软链 + 基线 smoke | ✅ |
| `sharding.py` / `dtensor_ops.py` / `kcp.py` 三个叶子文件 | ✅ 350 行,**零改动落地** |
| CP 契约改为折叠布局(`SEQ_DIM=0`, `HEAD_DIM=1`) | ✅ |
| Ulysses all-to-all 数值验证 | ✅ **cp2 与 cp4** 各自 forward 逐元素精确、round-trip 逐位、backward 梯度正确 |

cp4 是必须跑的:cp2 上很多错误置换是对称的,测不出来。

---

## 待办(按顺序)

1. **CP 文本侧**:给他们的 KDA 加 KCP、给 MLA 加 Ulysses、`apply_cp_kimi_k3`,
   从 `unsupported_parallelisms` 删掉一行
2. **CP vit 侧**:`vit_cp_plan.py` + dynamic CP
3. **PP**:`pipeline_adapter.py`(唯一改动是 `vision_tower` -> `vision_encoder`)+ DEP 三个文件
4. **EP**:`apply_ep_kimi_k3`(不含 MoonEP)
5. **TP**:声明式,最后
6. LoRA 放在每一轴之后

---

## 两条纪律

**gate 从冻结副本跑,不从活树跑。**08-23 那轮 58 格我在它运行期间提交了类名迁移、
还回滚过一次工作区,后半程格子的源码和前半程不同。结果是 58/58 零失败、
类名不改数值,但证据链严格说是断的。不再这样做。

**上游接口的改动单独记。**如果 PP 的 `pipelining_fn` 或 CP 的契约挂不上他们
97 行的 `parallelize.py`,就改上游接口 —— 每一处这样的改动单列一节,
它们是 PR 里最需要 maintainer 点头的部分。
