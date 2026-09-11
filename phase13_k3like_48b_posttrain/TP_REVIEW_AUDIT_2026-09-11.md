# TP/SP #4499 返工分支审核（CPU 盒子，2026-09-11）

对象：`tp_sp_on_main` = `d4d6e774c`（4 个提交，基 `da2f82670`，落后 upstream main `91abd2301` 5 个），
`k27_vision_tables_tp` = `d8120354e`（基 `ac10ca48f`，落后 22），PR 分支 `k3_tp_sp` = `9a62f5229` 未动。
逐行读了 `git diff da2f82670 d4d6e774c`（8 个文件，+437/−55）。

## 代码：结构正确，两处要改

按 diff-audit 规则扫过：**无 logbook 路径、无 docstring 里的实验数字、无新增 pyrefly/type ignore、
无新增 TODO**；`sharding.py` docstring 74/463 行（16%）。落后的 5 个上游提交不碰本分支任何文件，
rebase 干净。

核过的三个结构风险，都没问题：

1. `set_kimi_k3_sharding_config` 里 `if not enable_tp:` 会跳过 `_set_vision_buffer_sharding` 和
   `_set_inner_kda_sharding`——**不是丢失**：TP 路径下 `_set_kda_sharding` 重新声明了 kernel 边界
   （头轴在 tp 上分片），`_set_vision_encoder_sharding` 重新声明了 `inv_freq` 和 `pos_embed`
   （`{DP: R, TP: I}`）。两条路径各自完整。
2. 塔确实复用了 K2.5 的 MoonViT 方案（`set_vision_transformer_block_sharding_config`、
   `vision_colwise_config`、`vision_scaled_bias_rowwise_config`、`invariant_norm_config`），K3 只
   定制 `projector.post_norm`。这正是 Shuhua `sharding.py:390` 的要求，也是 `common/attention.py`
   与 `common/linear.py` 能回退的原因。
3. `Decoder.Config.update_from_config` 先跑（她的 #9），顺带让 core 自己的 `n_heads % tp` 检查
   （`decoder.py:181-188`）先于我们的塔检查执行——**解码器的头数归 core 管，塔的头数归我们管**，
   分工干净。塔检查是 K2.5 `model.py:81` 那条的移植，只是 K3 的属性路径更深
   （`vision_encoder.block.attn.num_heads`）。

### 要改的两处

**(a) `parallelize.py` 里有一条死逻辑。**

```python
if parallel_dims.tp_enabled and parallelism.spmd_backend != "spmd_types":
    raise NotImplementedError(...)
...
if (parallelism.spmd_backend == "spmd_types"
        or parallel_dims.ep_enabled
        or parallel_dims.tp_enabled):        # <- 永远轮不到它决定
    model.parallelize(parallel_dims)
```

上面那个 raise 已经保证 `tp_enabled ⟹ spmd_backend == "spmd_types"`，而那正是第一个析取项。
第三项永远不可能是决定性的。删掉，否则 reviewer 会问"这是不是说明 TP 可以不在 spmd_types 上跑"。

**(b) 顺手删掉的空行**：`model.parallelize(...)` 与 `if ac_config is not None:` 之间的空行在 diff
里被删了，和这次改动无关。补回去，保持 diff 最小。

### 作者那条已经解决

`tp_sp_on_main` 的栈底 `ddb306332` 作者已经是 `QIU023 <yiqiaoqiu@hotmail.com>`；
`Shuhua Yu` 只留在旧 `k3_tp_sp` 的 `c0e1584df` 上，强推同步时会被整条替换。
`k27_vision_tables_tp` 的作者也是 QIU023。**不需要额外动作。**

## 决定一：tp4 —— 建议从 body 拿掉，改成写明约束

数字查清楚了：**debug 塔 6 个头，真实 K3 塔 12 个头**（`_kimi_k3` 的
`vision_encoder=_vision_encoder_config(..., num_heads=12)`）。所以：

- debug flavor：tp=2 可以，**tp=4 被塔的头数检查拒绝**；
- released 模型：tp ∈ {2,3,4,6,12} 都能整除 12，**tp=4 没问题**；tp=8 会被拒（12 % 8 ≠ 0），
  尽管解码器的 96 个头能整除 8。

也就是说 **tp4 跑不了是 debug flavor 的产物，不是模型的限制**。这比"待补"好讲得多。

body 现在两处要改：

- Results 末句"the tp=4 rows move to a **text-only** or larger-tower configuration"——**没有
  text-only flavor**（评审里被拒过），这是在承诺一个不存在的东西。
- Limitations 的"EP x TP is measured at dp2 x ep2 with **tp=2 and tp=4**"——那些 tp4 数字是上一个
  head 上塔还被 replicate 时测的，**在这个栈上不成立**，照发就是错的主张。

建议改成一句陈述：the debug tower has 6 heads, so tp=4 is refused by the same check Kimi K2.5
carries; the released tower has 12, where tp=4 divides (and tp=8 would not). 然后表里只留 tp=1/tp=2。

## 决定二：K2.5 tables —— 建议单独开 PR

支持单独提的理由：

- 它是**一个文件 +3/−6**，修的是 K2.5 自己的问题：main `da2f82670` 上 K2.5 debug 模型
  dp2 x tp2 + 类型检查在 `_compute_learned_pos_embeds` 报
  `mutate_type: expected current type R on axis mesh_tp, got I`；K2.5 没有 TP 的 CI 格，所以没人撞见。
- Shuhua 明确问过"do we need changes in this file"——把它挪出 4499 的 diff 正面回答了这个问题。
- 4499 的 diff 少一个跨模型文件，审起来更快。

必须在那个 PR 里写明的一句：**它只是让 K2.5 过了这两张表，之后 K2.5 会撞上自己的
`QK clip scales do not match the MLA weight shape`**——不要让它看起来像"打开了 K2.5 的 TP"。

操作：`k27_vision_tables_tp`（`d8120354e`）落后 main 22 个提交，先 rebase 再提；在它合并之前
4499 保留栈底那个提交，合并后 rebase 时 drop 掉。

## body 的问题（`PR_BODY_TP_SP_v2_DRAFT.md`）

**Results 现在是一份待办，不是结果。** 开头就是"To re-measure on the new stack…"，然后给了一张
3 步表，再用散文给出两组**不同批量、不同初始化**的数字（类型检查那组 4096 tokens + 每 cell 独立
seed-42 初始化，没有共享检查点）。一个 Results 里三套协议，违反数值表规则（一张表只配对读同样
样本的格，一条噪声底行）。

重测时建议钉死一套：tp=1（main）作参照、同一个 seed 检查点、同一个 inductor 缓存、100 步、
步 1/10/20，cell 为 tp=1 main / tp=1 本分支 / tp=2 SP on / tp=2 SP off / dp2×tp2 / dp2×ep2×tp2，
外加一条噪声底行（同一个格两次，或同样数据换一个归约顺序）。tp=1 本分支对 main 必须逐位——
这是这个 PR 的验收线，比后面几步的百分比重要得多。

Tests 段里"the CPU suite's failure set is main's own"太含糊，给个数或者删掉；
"pinned pyrefly 13 errors, none in the touched files"很好，保留。
