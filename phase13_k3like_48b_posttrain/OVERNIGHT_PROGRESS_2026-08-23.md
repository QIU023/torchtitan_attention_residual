# Overnight 进度(2026-08-23,滚动追加)

树:`/workspace/tt_4025/torchtitan`,分支 `k3_on_4025`,起点 `dee45e357`。
计划见 `OVERNIGHT_2026-08-23_B_MIGRATION.md`。判据见同文件。

新母树基线(`kimi_k3_debugmodel`,dp2,3 步,seed 42 deterministic):
loss 12.51502 / 11.35441 / 9.89706。
文本基线(`kimi_k3_debugmodel_text`,dp2):loss 12.40265 / 9.97898。

## CP 分支 — 完成

| commit | 内容 |
|---|---|
| `03e1fdb1d` | KDA 的两条 CP 路径(kcp / ulysses) |
| `dfc04cbb9` | 接进 parallelize;core 加 `cp_via_sharding_config`;文本 flavor |
| `79abe69b2` | CP x 视觉塔 |

实测:
* all-to-all(折叠布局 3D):放置逐元素精确、round-trip 逐位、backward 正确,**cp2 + cp4**
* KDA CP vs 非 CP:**cp2/cp4 双模式 max_abs ~1e-6**(bf16 ulp 量级)
* 文本 cp2 训练:loss 12.45262 -> 10.25607
* 多模态 cp2 训练:loss 12.45567 -> 11.37354

### 上游接口改动(PR 里要单独说明)

1. **`Decoder.Config.cp_via_sharding_config`(新,默认 True)**。
   `validate_cp_backend` 自己的 docstring 写着它是给"在 ShardingConfig 里声明 CP
   的模型"调的,但 `Decoder.Config.update_from_config` 无条件调。KDA 跑 fla triton
   kernel,不走 DTensor 派发,**任何 ShardingConfig 都够不到它**,所以 K3 不可能是
   声明式 CP。声明式模型行为不变。

### K3 自己新增的前提条件

2. **CP 下禁止 load balancer**。默认 `headtail` 会跨 rank 置换 token;Ulysses 的
   all-to-all 按 rank 顺序重组序列,KDA 的递归按 rank r -> r+1 传状态,**两者都要求
   rank r 持有第 r 段连续序列**。置换后形状仍然对得上,数值静默错。

### 两个必须记下的机制

3. **mask**:core 切 CP mask 是按 **ring attention** 的形状(q 本地、kv 全局),
   而 Ulysses 每个 rank 重建完整序列 -> 尺寸不匹配。K3 禁止 sample packing,
   所以完整序列上就是纯因果 mask,在 Ulysses 路径里重建。**packed 序列需要把全局
   边界传下来,现在没有。**
4. **视觉塔必须留在每个 rank 的图里**。局部分片没有图像 token 的 rank 不消费任何
   embedding,塔就拿不到梯度,FSDP 少发一次 reduce_scatter -> 死锁(实测 300s
   watchdog,一个 rank 卡在 reduce_scatter,另一个已领先两个 collective)。
   加一个精确的零解决,embeddings 不变。

## 上游 vs 我们(已验证,不是读文档)

* PyTorch 的 `_ContextParallel` 实现的是 **ring attention**
  (`_templated_ring_attention` / `_RotateMethod`),**不是 Ulysses**。
  torchtitan 只在 docstring 里提它。
* 所以 MLA 的 Ulysses 和 KDA 的 KCP **上游都没有**。
