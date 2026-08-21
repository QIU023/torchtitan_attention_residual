# TP 让 grad_norm 放大 5.6 倍(2026-08-21)

## 结论先说

**开启 TP 会把 `grad_norm` 放大约 5.6 倍,而 TP 在数学上不该改变梯度范数。**

对照实验说明这是**我们的 bug,不是上游的**:

| 模型 | 无 TP | 有 TP | 变化 |
|---|---|---|---|
| `llama3_debugmodel`(上游) | 1.5110 | 1.5250 | **+0.9%** |
| `kimi_k3_mini_block_attn_res` | 3.2618 | 18.2391 | **+459%** |

两边都是 `partial_dtensor`、同 dp、只加 tp=2。上游模型的表现是应有的(TP 透明),我们的不是。

## 影响面

`grad_norm` 直接决定梯度裁剪的强度。放大 5.6 倍意味着 **TP 格子上的裁剪一直比应有的激进得多**,
训练动力学因此不同 —— 而 loss 曲线看不出这一点(step 1 的 loss 7.71255 vs 7.71445 只差在第四位)。

**所以 58 格矩阵里每一个带 TP 的格子,其数值都建立在这个缺陷之上。** 它们"通过"过,因为
gate 的判据是跑满步数,不比数值;而它们彼此一致,因为缺陷是稳定的。

## 是怎么撞见的

不是专门找它。做 `spmd_types` 迁移时,TP+CP 在新后端下跑通了但 grad_norm 是 **2.8975**,
与基线的 18.2391 差六倍。第一反应是"新路径错了" —— 直到注意到基线里的模式:

| 基线格子 | grad_norm |
|---|---|
| cp2 / fsdp2_pp2_cp2 / ep2_fsdp2_pp2_cp2(无 TP) | 3.3554 / 3.2983 / 3.2437 |
| fsdp2_tp2_cp2 / tp2_pp2_cp2 / ep2_fsdp2_tp2_cp2(有 TP) | 18.2391 / 17.5364 / 17.8860 |

**所有带 TP 的格子都在 ~18,不带的都在 ~3.3。** 而 spmd_types 给的 2.8975 落在无 TP 那一族。

于是问题从"新路径为什么不对"变成"**哪一边才是对的**"。同 dp 同 cp 只加 TP 的实验,
和 llama3 的对照,都指向基线。

## 定位:97% 来自词表侧

在 `clip_grad_norm_` 入口按参数形状分组统计梯度平方和(没有先挑假设去改,而是先测):

| 参数组 | 无 TP | TP + partial_dtensor | TP + spmd_types |
|---|---|---|---|
| **`(2016, 512)` x2** | **3.1199** | **592.6188** | **2.0295** |
| `(512, 512)` x87 | 0.9723 | 14.9619 | 0.6519 |
| `(896, 512)` x2 | 0.2136 | 0.3273 | - |
| **总计 sqrt** | **2.2586** | **24.6775** | **1.8975** |

`(2016, 512)` 是 **`embed_tokens.weight` 与 `lm_head.weight`** —— 词表侧那两个。
它们单独膨胀 **190 倍**,占总膨胀的 97%。

这与 TP 计划里词表侧的特殊处理吻合:`lm_head` 是 `ColwiseParallel(output_layouts=Shard(-1))`
走词表并行的交叉熵,而 `embed_tokens` **没有条目** —— 靠 torchtitan `Embedding` 模块自己的
vocab-parallel 前向。梯度归约的责任在那条路径上,而它显然没有被正确处理。

**新路径(spmd_types)在同一分组上是 2.03,落在无 TP 的 3.12 量级。** 这把
"新路径数值不一致所以可疑"翻转成"**旧路径独自暴涨,新路径更可能是对的**"。

### 仍未证的一步

"词表并行的梯度归约不对"是**从数据推出的定位,不是机制**。具体是缺一次 all-reduce、
还是 `_LossParallelCrossEntropy` 的缩放、还是 `Embedding` 模块内 vocab-parallel 前向的
反向路径 —— 三者都未验证。定位到两个参数已经足够窄,但**别把它当成已知原因**。

## 与 spmd_types 迁移的关系

spmd_types 下的 TP 数值(2.8975)看起来更接近正确,但**不能据此宣布新路径是对的** ——
它也可能因为别的原因恰好落在那个量级。两条路都要独立验证,而不是互相背书。

在这条查清之前,**任何"TP 格子逐位一致"的比对都只是在比对同一个缺陷的两次复现**。


> **已修复(同日)。** 根因是 embedding 声明缺了输入侧那两件,不是下面一度写的
> "`_redistribute_outputs` 不支持 plain"。TP 现在对 grad_norm 透明:3.2444(有 TP)对
> 3.2618(无 TP),三步全程贴合,修复前是 18.24。提交 `d315cdd64`。
> 下面保留完整过程,因为归因错了一次,而**发现它错的是"llama3 为什么没事"这一问**。

## 追因过程(2026-08-21 续)

### 先修好的是另一个 bug

`Embedding.forward` 的 vocab-parallel 分支按 `chunk = ceil(vocab / tp)` 索引权重,而我们的 TP 计划
在 08-14 删掉 `embed_tokens` 条目后**没有任何东西再切权重** —— 实测 `local_rows=2016` 对
`chunk=1008`,**rank 1 一直在读错误的行**。加上 `tp=S(0)` 声明后变 OK(提交 `a75d93f83`)。

这是真实的前向错误,窗口 **08-14 至今**。但它**不解释 grad_norm**:修完之后 `embed_tokens`
的梯度平方和 592.5 -> 594.8,几乎没动。**两个是独立的 bug。**

### grad_norm 的根因

补上上游那份完整声明(输出 `tp=P`,再重分布到 `R`,那次重分布就是跨 rank 求和)之后,
**数值一点没变**。逐层查下去:

| 观测 | 结果 |
|---|---|
| 驱动器进了 Embedding | 是(`entered ... {'Embedding': 1}`) |
| 声明装上了、forward 被包装 | 是(`cfg=True forward_wrapped=True`) |
| 包装后的 forward 被调用 | 是 |
| **包装后的返回值** | **plain tensor,不是 Partial DTensor** |

根因在 `Module._redistribute_outputs` 的 `partial_dtensor` 分支:**它的两处重分布都写在
`if isinstance(outputs, DTensor)` 里**。我们的 `Embedding.forward` 在 vocab-parallel 分支里
显式 `to_local()` 后返回 plain,于是两个分支都不进,**原样返回**。

上游 llama3 不受影响,是因为它的 embedding 输出本来就是 DTensor。

**所以声明写着 `P -> R`,而没有任何代码把 plain 提升成 Partial 去执行那次求和。**
各 rank 的置零结果从未相加 —— 这就是缺失的跨 rank 归约。

### 一条方法上的教训

中途我用探针钩 `Module._redistribute_outputs` 想看它是否被调用,**没有输出**,当时读成了
"包装没生效"。但另一个探针证明包装后的 forward 确实被调用了。两个观测矛盾时,**先怀疑探针**:
那次钩子没生效,而"没有输出"看起来和"代码没跑"一模一样。改成直接看返回值才拿到真相。


## 修复:声明要五件套,不是两件

上游 `tok_embeddings` 的声明是:

    state_shardings   {"weight": tp=S(0)}          # 权重按词表切
    in_src / in_dst   {"input": tp=R}              # <- 我最初漏掉的
    out_src / out_dst  tp=P -> tp=R                # 各 rank 的部分和求和
    local_map          in_grad_placements=None

我分三次补,每次都实测:

| 补了什么 | grad_norm | 说明 |
|---|---|---|
| `state`(权重切分) | 18.24 -> 25.12 | 修好了读错行,**没解决膨胀** |
| `+ out_src/out_dst` | 25.12(不变) | 完全没效果 |
| **`+ in_src/in_dst`** | **25.12 -> 3.2444** | **解决** |

为什么是输入侧:`Module._redistribute_outputs` 在 partial_dtensor 下**只对 DTensor 做重分布**,
而 `Embedding.forward` 的 vocab-parallel 分支 `to_local()` 后返回 plain。声明输入会把它提升成
DTensor,于是 `F.embedding` 的输出**自然是 DTensor**,那条 `P -> R` 的路径才进得去。

实测确认:补齐后 embedding 输出从 `plain` 变成 `DTensor(Replicate())`。

## 归因错过一次,以及是什么纠正了它

中间版本写的根因是"`_redistribute_outputs` 不支持 plain,所以要改 `models/common` 的共享模块"。
那个说法**命名了一个真实行为却归错了因**,而且它有一个自己解释不了的事实:**llama3 用同一个
`Embedding` 类、同样 `to_local()` 返回 plain,为什么不受影响?**

实测两边的 embedding 输出:

| | 输出 |
|---|---|
| llama3 | `DTensor(Shard(dim=1))` |
| kimi_k3(修复前) | `plain` |

同类同代码不同结果 -> 差别只能在声明。查上游声明,发现是五件套,我抄了两件。

**沿着错误归因走下去会去改共享模块**,而缺陷完全是本地的。拦住它的不是更仔细的代码阅读,
是"为什么别人没事"这个问题 —— **一个正确的根因必须能解释所有的观测,包括没出问题的那些。**

## 三个 bug 同源

都来自 08-14 那次"删掉 `embed_tokens: RowwiseParallel`,改用模块自带的 vocab-parallel 前向":

1. 权重没人再切 -> rank 1 读错行(前向错误);
2. 输出 P 声明缺失(被 3 掩盖,单独补无效果);
3. **输入声明缺失 -> 输出不是 DTensor -> 跨 rank 求和从未执行 -> grad_norm 膨胀 5.6 倍**。

删 style 的理由本身成立(MaskPartial 无法与 P(sum) 重分布),漏的是**删掉 style 之后要补等价的
声明**,而"等价"意味着五件套,不是其中一件。
