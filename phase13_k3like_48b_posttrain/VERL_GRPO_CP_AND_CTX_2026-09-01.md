# GRPO×CP 通过,以及本地能模拟多长 ctx(2026-09-01,4×5060Ti)

树:k3_on_4025 @ 03b685196(verl 的 torchtitan 已改指集成树)。
verl:QIU023/verl@3263714f。

## 一、GRPO 在新树上闭环,并且 CP 进了 RL 环

| 跑 | 网格 | 步 | rollout↔actor pearson | probs_diff_max |
|---|---|---|---|---|
| run 31(新树基线) | FSDP4 | 4/4 | 0.9999 | ~1e-5 |
| **run 32(GRPO×CP2)** | FSDP2×CP2 | 4/4 | **0.9995~0.9996** | 2.4e-5 |

CP 使归约序改变,corr 从 0.9999 降到 0.9995 属预期。至此 Stage-2 的并行维度
补齐:FSDP、EP(Stage-1 背书)、CP。

### 顺带还清一条真树债

GRPO×CP 首跑抛
`NotImplementedError: context parallel folds 3152 tokens into one stream but
the context window is 2048`——集成树的 CP 仍停在"causal-only mask + 上下文
窗口守卫",doc-mask 修复只做在 cp_review1 上、从未回灌。rollout 的变长
packed 流天然多文档,必撞;SFT 那条路只是恰好没超窗。已 cherry-pick
(98e6a7cc2 → 03b685196),两处冲突取并集:保留树上后来的 `tp*cp` 头整除
检查,丢掉随修复作废的 `_cp_max_context_length` 管线。52 项 + 186 子测绿。

## 二、ctx 阶梯:8k 稳跑,16k OOM

单序列微批(只让 ctx 变动),FSDP4,vLLM 份额 0.30:

| ctx | fused kernels | 峰值 reserved | 结果 |
|---|---|---|---|
| 2048 | off | 4.97 GiB | 通过 |
| 4096 | off | 7.51 GiB | 通过 |
| **8192** | off | **12.43 GiB** | 通过(15.5 GiB 卡) |
| 8192 | **on** | 12.49 GiB | 通过,**与 off 无差别** |
| 16384 | on | — | **OOM** |

斜率 **1.23 GiB / 1k token**,与按 logits 主导项算出的 1.25 GiB/1k 几乎重合
(vocab 163840 × bf16 = 320 KiB/token)。16k 需要 ~22 GiB,单卡装不下。

### per-token 成本表(K3 debug:12 层、hidden 1024、9×KDA + 3×MLA)

| 项 | 每 token | 备注 |
|---|---|---|
| vLLM KV cache | **3.4 KiB** | 9 层 KDA 是常数状态、不随长度增长;3 层 MLA 压到 512+64 |
| 训练侧激活 | ~144 KiB | selective AC |
| **训练侧 logits** | **320 KiB** | **瓶颈** |

KV 便宜到几乎不构成约束(5.4 GiB ≈ 150 万 token),这是混合架构的直接好处。

### 一个反直觉实测:`use_fused_kernels=True` 在这条路径上无效

开关确实生效(配置里 True),但显存与关闭时一致。原因:verl 的融合
linear-CE 走它自己的 HF-model 路径,而 **titan 引擎的 `model_forward_step`
返回的是已物化的 logits**,融合核接不进来。这是 titan 引擎在 verl 里的真实
缺口,不是配置错误。

### 拉长的三条路(按代价)

1. **分片 loss**:去掉引擎里把分片 logits all-gather 回全长的那步(它是为
   迁就 verl 假设全长的 loss 路径),让 loss 在 CP 分片上算 → CP2 峰值减半,
   8k→16k;CP4→32k。
2. **把融合核接进 titan 引擎**:引擎返回 hidden states 而非 logits →
   320 KiB/token 那项消失,单卡 30k+。顺带修好上面那个缺口。
3. 压 vLLM 份额:只能挤 1-2 GiB,治标。

真实形状下比例会变:K3 生产模型 vocab 同为 163840 但 hidden 7168、层数远多,
激活项会反超 logits 成为主导。这张表按项分解,可直接换参数重算。

## 三、复现

`scratchpad/verl_stage2_cp.sh`(GRPO×CP2)、`scratchpad/verl_ctx_ladder.sh`
(阶梯,含 fused 开关);logs 在 `/workspace/verl_stage2_cp.log`、
`/workspace/verl_ctx_ladder/`。
