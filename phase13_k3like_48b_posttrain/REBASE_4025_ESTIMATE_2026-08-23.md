# 4025 merge 后的迁移顺序与四个 branch 的划分(2026-08-23)

前提:4025 已被 approve(@shuhuayu),merge 后我们拉上游树,把四种并行分四个 PR 推上去。
这份文档回答的是**顺序**和**每个 branch 的边界**,不是工期。

---

## 一、最重要的一条实测:参照树四种并行全是 NotImplementedError

`torchtitan/models/kimi_k3_up/`(vendor 自 `dee45e35`,与 4025 head 零漂移)的
`parallelize.py` 只有 **97 行**,正文是:

```python
unsupported_parallelisms = [
    name for name, enabled in (
        ("tensor parallel",   parallel_dims.tp_enabled),
        ("pipeline parallel", parallel_dims.pp_enabled),
        ("context parallel",  parallel_dims.cp_enabled),
        ("expert parallel",   parallel_dims.ep_enabled),
    ) if enabled
]
if unsupported_parallelisms:
    raise NotImplementedError(
        "Kimi K3 currently supports FSDP2 data parallelism only; disable ...")
```

model compilation 同样 `NotImplementedError`。

整棵参照树的 `sharding_config` / `ShardingConfig` / `SpmdLayout` 出现次数 **0**:

| 树 | 声明出现次数 |
|---|---|
| 参照树全部 10 个文件 | **0** |
| 我们 `sharding.py` / `model.py` / `vision_encoder.py` / `attn_res_model.py` / `parallelize.py` | 22 / 18 / 18 / 13 / 12 |

**被 approve 的是模型定义 + FSDP2,不是任何并行实现,也不是声明式。**

这一条推翻了 08-23 早些时候我给出的"TP 该最先,因为方向就是他们 approve 的方向"。
那句话的前提是错的。

---

## 二、按两个维度重排

### 维度 1:声明式 gap —— 全部残留都在 TP 侧

`parallelize.py` 里残留的命令式 `plan[...]` / `parallelize_module(...)` 条目:

| 函数 | 条数 |
|---|---|
| `apply_tp_kimi_k3` | 6 |
| `_apply_tp_moonvit_mlp` | 2 |
| `_register_lora_tp` | 1 |
| `apply_cp_kimi_k3` | **0** |
| EP / PP | **0** |

且 PP/CP 对 TP 声明式结构**零依赖**(grep `ShardingConfig|SpmdLayout|tp_replicate|tp_shard|colwise_config|rowwise_config`):

| 文件 | 命中 |
|---|---|
| `apply_cp_kimi_k3`(155 行) | 0 |
| `pipeline_adapter.py`(1701 行) | 0 |
| `kcp.py` / `vit_cp_plan.py` / `moon_ep_dispatcher.py` | 0 / 0 / 0 |
| `apply_ep_kimi_k3`(60 行) | 1(`sharding_config`) |

**PP 和 CP 不等 TP 迁移。**

### 维度 2:护城河 —— 要按"别人能不能复制"分,不是"上游有没有"

上游对 K3 四种并行都没有。区别在于有没有可套的通用基建:

| | 上游通用机制 | 别人能否机械复制 |
|---|---|---|
| **PP**(Block-AttnRes 跨 stage adapter) | **无**,且通用化被 Tianyu 约 2026-04 拒过 | **不能** |
| **CP**(KDA state-passing / KCP + MLA Ulysses) | **无**;qwen3_5 这个混合线性注意力先例的 CP 也空着 | **不能** |
| EP | 有 `token_dispatcher`、`moe_sharding` | 能 |
| TP | 有声明式 sharding 全套 | 能 |

TP/EP 是"拿上游基建套上去就有",**先发价值最低**;PP/CP 没有可套的东西。

---

## 三、顺序与四个 branch 的边界

### 第 0 层:不占这条队列,现在就能提

模型文件夹外的全部改动,17 个文件约 1500 行,与 4025 无关:

| 文件 | 行数 | 状态 |
|---|---|---|
| `components/quantile_balance.py` + 测试 | 374 + 273 | PR body 已写好(`Raising_PRs/PR_QUANTILE_BALANCING/`) |
| `components/mx_qat.py` + 测试 | 291 + 124 | K3 依赖已抽成参数 |
| `components/optimizer/utils.py` | 39/12 | **上游潜伏 bug**:DCP resume 失败 |
| `components/data/loader.py` | 18/4 | **上游潜伏 bug**:同上 |
| `distributed/utils.py` / `fsdp.py` | 73/5、29 | |
| `models/common/moe.py` / `moe_sharding.py` | 47/14、9/2 | |

### 第 1 批(并行):PP 与 CP

| branch | 主体 | 在 `parallelize.py` 里 |
|---|---|---|
| **PP** | `pipeline_adapter.py` 1701 行 + `dep_bubble_*.py` 704 行 | **0** —— PP 是 ModelSpec 的 `pipelining_fn`,`__init__.py` 一行注册 |
| **CP** | `apply_cp_kimi_k3` 155 + `kcp.py` 104 + `vit_cp_plan.py` 301 + `sharding.py` 的 CPContract | 155 行,自成一块 |

两者对 `parallelize.py` 的共同触点只有 `unsupported_parallelisms` 那个 list 里
**各删一行**。git 层面基本不冲突 —— 这是"真并行提交"第一次成立的原因。

### 第 2 批:EP(不含 MoonEP)

`apply_ep_kimi_k3` 60 行 + `moon_ep_dispatcher.py` 137 行。MoonEP 单独留后。

### 第 3 批:TP

唯一还有 9 条命令式残留的;要吸收上游声明式全套;vit 侧(`wqkv` 融合、
三个 AttnRes norm)本来就没迁完 —— 放最后不影响前三批。

---

## 四、母树对齐(所有 branch 的共同前提)

四个 branch 都从对齐后的母树切出去,所以母树的结构对齐必须先做完。
剩余步骤见 `HANDOFF_2026-08-23.md` 第一节,状态:

| 步 | 内容 | 改数值? |
|---|---|---|
| 2 | KDA -> `kda.py` | ✅ 已完成(`0531a8a3b`) |
| 3 | 类名对齐(`MoonViT*` -> `KimiK3Vision*`)+ config 字段名 | 否 |
| 4 | KDA dtype fp32 -> bf16 | **是,单独 gate** |
| 5 | folded token layout | **是**,且是唯一"没读懂就没法估"的项 |

**步 5 在关键路径上,而且纯读代码不占卡** —— 应排在下一个 GPU session 之前。

---

## 五、工期

不给天数。本地 Claude 那份评估的落点我核过(见第二节),**工期依据没核**,
而它自己把最大不确定项(folded layout)标成"没读懂"。
在读懂步 5 之前给出的任何总量都是猜的。

可以说的两条:
* 第 0 层不依赖 merge,可以立刻并行开;
* 第 1 批的两个 branch 主体已经写完并被 58 格覆盖过,不是新开发,是**搬运 + 适配**。

---

## 六、方法记录

这一轮我给出的第一版顺序(TP 最先)建立在"4025 是声明式 TP 树"这个**没测过的**前提上。
一条 `grep -c ShardingConfig` 就推翻了它 —— 参照树的答案是 0。

和交接文档第七节记的是同一个形态:**读到一段能解释的东西就下结论,
没先跑那个一锤定音的观测**。参照树就在本地 `models/kimi_k3_up/`,
97 行的 `parallelize.py` 打开就看得见。
