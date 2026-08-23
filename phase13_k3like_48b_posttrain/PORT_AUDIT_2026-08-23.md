# 移植审计:新树 vs 原树(2026-08-23)

判据是用户定的:**凡是原树已经写好的,不允许在新树上再写一遍。**

## 机械对比:13 个搬过去的文件

**11 个逐字节相同**:`dtensor_ops` / `kcp` / `attn_res` / `layout` / `knobs` /
`dep_bubble_{plan,backward,runtime}` / `vit_cp_plan` / `vit_prefetch` / `vision_preprocess`。

| 文件 | 差异 | 判定 |
|---|---|---|
| `sharding.py` | 裁掉 TP helper(`tp_replicate`/`tp_shard`/`declare_*`);`SEQ_DIM/HEAD_DIM` 从 `(1,2)` 改 `(0,1)` | **没有发明新函数**;折叠布局没有 batch 轴 |
| `pipeline_adapter.py` | 5 处:`vision_tower→vision_encoder`、层数从 Config 树取、FQN 拼写回映、塔的 stage 归属、DEP 明确报错 | 都是新树结构差异所必需 |

## 我重写了原树已有的东西 —— 四处,全部换掉

| 我原本写的 | 原树已有 | 换掉后 |
|---|---|---|
| 内联 `+ vision_embeds.sum() * 0.0` | `add_zero_valued_dependency`(`distributed/fsdp.py`) | **表达式一字不差**。helper 已搬到新树,改用它 |
| all-gather 整条 token 序列重建全局位置 | `_exchange_sentinel_counts` + `_select_cp_shard` | 通信降到**每 rank 一个整数**;多了"计数和 == 编码行数"一致性检查;数值逐位相同 |
| `isinstance(p, DTensor)` | `isinstance(p, DTensor) and any(pl.is_shard())` | **Replicate 的 DTensor 不发 all-gather**,不需要占位 |
| 前向里 `n_heads % cp` | 接线时 `_check_head_divisibility(..., tp*cp)` | **TP 已切过一次头,除数是 `tp*cp`**;cp-only 判据会拒掉能跑的配置 |

顺带补上原树有、我完全没有的两样:

* **kcp 模式检查 fla CP 算子可用**,报错里写清需要 fla-core >= 0.5.1。否则是层内第一步的 ImportError。
* **接线时打印接了多少层**(实测 6 MLA + 18 KDA)。这是原树防"静默没接上"的手段 ——
  而 TP 那次静默空转已经咬过一次。

## 核过、确认不用改的

| 项 | 结论 |
|---|---|
| KDA 的 `_forward_kcp` / `_forward_ulysses` | 与原树差异**全是属性改名**。唯一实质点 `output_final_state`:他们的 kernel 不传,而 `chunk_kda` 的默认就是 `False`,与原树显式传 `False` 等价 |
| MLA Ulysses | 已按原树结构重写(rope 排除在 all-to-all 外)。**数字一位没变** —— 两种写法数学等价,消除的是分叉风险不是 bug |
| `_full_sequence_causal_mask` | 原树用 SDPA、不传 mask;新 core 禁了 SDPA。新树独有 |
| 块残差作 stage 载荷 | 原树用 `CrossStageCacheAdapter`,依赖新树没有的 wrapper 结构 |

## 两处上游规则

* **`to_local` 的 `grad_placements` 现在显式**(`rules/distributed.md` 硬要求)。
  原树也没显式 —— 是两棵树共同的规则不符,行为不变。
* **`cp_via_sharding_config`**:试过改成从 config 推导以避免往 core 加字段,**实测否掉**——
  llama3 也是在调用基类之后才设声明,推导会把守卫对所有模型静默关掉。
  已退回显式开关,并把这次否定写在字段旁边。

## 补上的一个真死锁

4025 的 `parallelize.py` TODO 写着"一个 DP rank 有图、另一个没图会死锁,需要通用解法"。
原树有解法(`_keep_tower_alive` + `_tower_placeholder`),新树没有。已补,并**证明它是真的**:

| | 结果 |
|---|---|
| 关掉修复 | **240s 超时挂住,零个 rank 完成** |
| 打开修复 | 两个 rank 都完成 |

探针:`mixed_batch_deadlock.py`。
