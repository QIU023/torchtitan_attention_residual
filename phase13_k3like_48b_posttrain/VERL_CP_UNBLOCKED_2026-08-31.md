# veRL×titan:CP 在 packed 无填充流上跑通(2026-08-31,QIU023/verl@3263714f)

Stage-0 章记的"CP 三连阻断"里,第三条(flex BlockMask 整除)当时判为硬墙,
留作 pad-to-multiple 工程。今天做完,并且在做的过程中发现**真正的首因是第
四条、且更靠前**:分片切错了轴。

## 一、四道关与解法

| # | 症状 | 真因 | 解 |
|---|---|---|---|
| 1 | KeyError: positions | 引擎把 positions 放 extra_inputs,titan 要 extra_kwargs | 调用前后交接 |
| 2 | headtail 要求 seq % (2·cp) == 0 | 与 NO_PADDING 变长流冲突 | load_balancer 传 None,连续切 |
| 3 | flex BlockMask 要求整除 cp·2·128 | packed 流长度任意 | **pad 到倍数 + logits 侧 unpad** |
| 4 | **rank1 拿到 `(0, 1950)` 空批** | **`cp_shard` 切 dim 0,而 rmpad 流是 `[1, T]`——dim 0 是折叠轴不是序列轴** | **切前折成 `[T]`,切后展回** |

第 4 条此前从未被看见:它的症状是下游一个形状解包错误
(`too many values to unpack`),看上去像模型 bug。`cp_shard` 其实有
`input_seq_dim` 参数,但外层 `prepare_context_parallel_input` 没有暴露它——
所以在引擎侧折流,而不是改 titan(折叠形态本来就是 K3 模型自己的契约)。

pad 的两个细节:**position 续写而非归零**(填 0 会被任何按 position 推文档
边界的 mask 读成新文档开头,静默错误);**mask 在 pad 后重建**。labels 保持
全长,loss 侧只见真 token。

## 二、实证(4×5060Ti,K3 debug SFT)

| 臂 | 网格 | step 1 loss |
|---|---|---|
| cp1 基线 | fsdp2 | 12.494515 |
| **cp2** | fsdp1×cp2,两 rank 各 (1, 975) | **12.495392** |

差 9e-4,CP 归约序量级;分片形状对称,无空批。

## 三、范围

- 本次只验 SFT 前向/反向路径;GRPO 环下的 CP 未测(Stage-2 是 FSDP4)。
- 倍数按模型选:模块内建 CP 的模型(kimi_k3,KCP 自己切序列)用 `cp`,
  走 flex-CP BlockMask 的模型用 `cp·2·128`。
- 这四条 + 解法是 verl 侧提 issue/PR 的现成材料(K3 线在 verl 的第二块阵地,
  第一块是 Stage-1/2 的 titan 引擎接线)。
