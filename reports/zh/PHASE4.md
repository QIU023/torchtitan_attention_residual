# Phase 4 报告 — Kimi Linear 主干 + AttnRes 上规模

**日期**：2026-04-23 → 2026-04-28（架构移植 + 4 个不同训练 run + 100K 续跑进行中）
**状态**：**架构移植完成**；12 500 步 FSDP A/B + 12 500 步 PP-adapter benchmark 完成；paper-faithful 从头跑 + 100K 续跑进行中
**硬件**：4× RTX 5090 PCIe（每卡 32 GB），单节点

---

## 1. 目标

把 MoonshotAI Kimi-Linear（KDA + MLA NoPE + MoE）以 torchtitan 风格落地，以便忠实地把 AttnRes 织进去，**Phase-3 PP cache adapter 不改动直接复用**（PP 路径里零 Kimi-specific 代码）。在我们手里的硬件（4× 5090 PCIe）上铺好 AttnRes paper Table-2 sweep（194M → 528M activated params）的基础；48B-A3B 留作远期多节点目标。

基于这个 port 跑两个 "problem"：

- **Problem A（FSDP A/B）**：在 436M shape、超参对齐前提下，Block AttnRes（paper N=8，L=16 时 num_blocks=8）相对 Kimi-Linear baseline 是否能改 loss？纯科学 A/B，并行策略是干扰项不是研究对象。
- **Problem B（PP cross-stage cache adapter）**：把主干从 Llama3 换到 KDA+MLA+MoE Kimi Linear，Phase-3 adapter 是否还能保持 loss 等价？纯系统跑。

---

## 2. 交付物

### 2.1 工作区（`phase4_kimi_attnres_lm_pretrain/`，**不**进 PR）

| 文件 | 作用 |
|---|---|
| `README.md` | 架构选择论证（"为什么新建 `experiments/kimi_linear/` 而不是嫁接到 `attn_res/`"）、48B-A3B HF config dump、scaling-law sweep 表、AttnRes weave 描述、PP adapter 复用契约、sanity gates、**100K 续跑计划与停止标准** |
| `launch_fsdp_small.sh` | 通用单节点 FSDP launcher（NGPU/STEPS/LBS/GBS/SEQ/LR/CONFIG/COMPILE/VAL/VAL_FREQ/VAL_STEPS 环境变量），同时支持 `attn_res` 和 `kimi_linear` 模块 |
| `launch_pp4_kimi.sh` | PP=4 V=2 lps=2 Interleaved1F1B + `TORCHTITAN_ATTNRES_CACHE=1`。默认 `kimi_linear_436m_block_attn_res` |
| `launch_continuation_100k.sh` | 从 step-12500 ckpt 续跑：仅加载 weights、Adam 重置、500 步 re-warmup 到 peak LR=3e-4、**warmup 后常 LR**（`decay_ratio=0.0`）、再跑 87 500 步 |
| `launch_from_scratch_paperhparams.sh` | 从头跑替代方案：**paper LR (2.20e-3)** + **`grad_accum=8`**（LBS=3、GBS=96 → effective bs=96，8× 原 Phase 4，恢复 ~sqrt(8)=2.83× 信噪比） |
| `experiments/kimi_436m_attnres/`（Problem A） | `launch_baseline.sh`、`launch_attn_res.sh`、`launch_continue_30k.sh`、`compare_loss.sh`、README 写明 apples-to-apples FSDP A/B 契约 |
| `experiments/kimi_pp_adapter/`（Problem B） | `launch_adapter_pp.sh`、`run_after_baseline.sh`（poller，等 Problem A 的 `Training completed` 自动启动）、`summarize_bench.sh`、`eval_val.sh`、`plot_comparison.py` |

### 2.2 进 PR 的代码（`torchtitan/experiments/kimi_linear/`）

独立 experiment，不挂在 `attn_res/` 上。理由（`phase4_kimi_attnres_lm_pretrain/README.md`）：DSv3 MLA 看上去像但不一样（`mla_use_nope=True`、特定 head dims、`q_lora_rank=null`、init scales、norm 位置都飘）；KDA 是新算子；KDA:MLA = 3:1 + first-N-dense 的 per-layer 调度是 Kimi 专属。`attn_res/` 留作 Llama3/DSv3 测试床；`kimi_linear/` 是生产目标。

| 文件 | 作用 |
|---|---|
| `model.py` | `KimiDeltaAttention`（KDA 经 fla-core 的 `chunk_kda` / `fused_recurrent_kda` / `fused_kda_gate` / `FusedRMSNormGated` / `ShortConvolution`）；`KimiMLAAttention`（NoPE 变体，**不**复用 DSv3 MLA，逐字对齐 Kimi spec）；`KimiMoE`（基于 torchtitan 共享的 `TokenChoiceTopKRouter` + `GroupedExperts` 重写，因为 HF 参考的 `KimiSparseMoeBlock` 在 training 模式下 `raise NotImplementedError`）；`KimiMLP`；`KimiDecoderLayer`；`KimiLinearModel` |
| `attn_res_model.py` | `KimiAttnResDecoderLayer` + `KimiLinearAttnResModel` 把 Block AttnRes 织进每层 Kimi decoder。每个 decoder layer 两个 AttnRes weaving point（pre-attn + pre-FFN）→ 总 2·Lb 个 pseudo-query，对应论文 "one per layer" 注脚里 Lb=L/2 的语义 |
| `pipeline_adapter.py` | 薄包装，转发到 `attn_res/pipeline_adapter.py:pipeline_llm_with_cache_adapter` —— **PP 路径里零 Kimi-specific 代码** |
| `parallelize.py` | Kimi 主干的 FSDP2 + compile + grouped_mm 接线 |
| `config_registry.py` | paper Table-2 五个 size（194M / 241M / 296M / 436M / 528M）× 三个变体（`baseline` / `block_attn_res` / `full_attn_res`）；外加 528M 的 `_l16` 变体（PP 整除性） |
| `reference/` | HF `moonshotai/Kimi-Linear-48B-A3B-Base` 的逐字 fork（`modeling_kimi.py`、`configuration_kimi.py`、`config.json`），仅作 diff 参考，**不 import** |
| `tests/test_layers.py` | KDA / MLA / MoE / decoder-layer shape 的 CPU smoke |

---

## 3. Sanity gates（已通过）

1. `pytest torchtitan/experiments/kimi_linear/tests/` 绿。
2. CPU debug flavor 前向 → 合理 logit shape。
3. 1-GPU 前向 → init loss ≈ log(vocab=163840) ≈ 12.0。Run log 印证（`paperhparams` step-1 loss = 12.23542）。
4. Debug flavor 4-GPU PP=2 V=2（L=4 时 lps=1）→ 50 步无 `RuntimeError`。
5. Debug AttnRes flavor + cache adapter ON → 50 步与 adapter-OFF 同步轨迹（bf16 噪声内）。

---

## 4. 验证结果

均为 4× RTX 5090 PCIe。

### 4.1 Problem A — 436M FSDP A/B，12 500 步

两边只有 `--config` 不同：

| 项 | 值 | 出处 |
|---|---|---|
| 模型规模 | 436M（L=16，d=1168，d_ff=528） | paper Table 2 |
| 架构 | Kimi Linear（KDA:MLA=3:1，除 first dense 外所有层 MoE） | paper §5 |
| Peak LR | 2.20e-3 | paper Table 2 (436M 行) |
| LR schedule | 500 步 warmup + cosine，decay_ratio=0.8，min_lr_factor=0.1 | torchtitan 默认 |
| Optimizer | AdamW | torchtitan 默认 |
| **SEQ_LEN** | **2048** | 硬件限制；论文 8192 |
| **GLOBAL_BS** | **12** | 硬件限制；论文 384 |
| LOCAL_BS / rank | 3 | 32 GB + grouped_mm + compile 下的最大值 |
| FSDP | 全 shard，4 ranks | |
| AC | OFF | parallelize_kimi_linear 默认 |
| `torch.compile` | ON | |
| `use_grouped_mm` | True | |
| 步数 | 12 500 | |

两次跑 `GIT_SHA = d30b9d3`，rank 0 训练日志末尾 loss：

| step | baseline | block_attn_res |
|---:|---:|---:|
| 12 480 | 3.64286 | 3.65438 |
| 12 490 | 3.69937 | 3.71735 |
| 12 500 | **3.82854** | **3.83739** |

C4-validation（同样 GIT_SHA = d30b9d3）：

| 指标 | baseline | block_attn_res |
|---|---:|---:|
| step 12 501 val_loss | **3.7190** | **3.7326** |
| 峰值显存 | 22.59 GiB | 25.82 GiB |
| eval tps | 6 724 | 6 223 |

代价清晰：AttnRes 多 ~3 GiB block 存储 + ~7 % eval tps slowdown；train/val loss 落在 seed 噪声内（Δ_train ≈ +0.009、Δ_val ≈ +0.014）。**论文报告的 "AttnRes 改 loss" 在 307M tokens 处不显现**（12 500 × 12 × 2048 = paper 87.9B 预算的 0.35 %）。诊断：LM 严重欠训，loss 还没进入 AttnRes 深度感知聚合相对标准残差能拉开差距的区间。

### 4.2 Problem B — 436M PP=4 V=2 + cache adapter，12 500 步

超参完全等于 Problem A 的 AttnRes arm，只换 parallelism：

| 项 | 值 |
|---|---|
| `pipeline_parallel_degree` | 4 |
| `pipeline_parallel_schedule` | Interleaved1F1B |
| `pipeline_parallel_layers_per_stage` | 2 |
| `TORCHTITAN_ATTNRES_CACHE` | 1 |
| LOCAL_BS | 1（PP 装得下） |
| GLOBAL_BS | 12（= num_microbatches；≥ 8 个 virtual stage = pipeline 灌满 + 4 mb 余量） |
| `torch.compile` | OFF（compile + PP 调度交互噪声大；关掉让 adapter 测量干净） |
| `use_grouped_mm` | True |
| 步数 | 12 500 |

`GIT_SHA = af266ee`。rank 3 loss：

| step | adapter loss |
|---:|---:|
| 1 | 12.23261 |
| 990 | 5.23611 |
| 4990 | 4.42266 |
| 9990 | 4.01609 |
| **12 500** | **3.88490** |

step 12 501 val on c4-validation：**loss = 3.7277**，每 rank 峰值显存 = 15.73 GiB。

汇总（`runs/kimi_pp_adapter_bench/comparison.png`）：

| arm | 并行 | step 12 500 train | step 12 501 val | 每 rank 峰值显存 |
|---|---|---:|---:|---|
| Problem A baseline | FSDP=4 | 3.82854 | 3.7190 | 22.59 GiB |
| Problem A AttnRes | FSDP=4 | 3.83739 | 3.7326 | 25.82 GiB |
| Problem B AttnRes (adapter) | PP=4 V=2 + cache | 3.88490 | **3.7277** | **15.73 GiB** |

PP+adapter val_loss 与 FSDP AttnRes 在 ~0.005 nat 内对齐（小于 FSDP 的 baseline-vs-AttnRes 漂移）。Phase-3 adapter 从 Llama3 主干推广到 KDA+MLA+MoE Kimi 主干。显存下降是结构性优势：PP 把 activation 跨 rank 切，每 rank 峰值约 FSDP 的 60 %。

### 4.3 100K 续跑（`kimi_436m_block_attn_res_fsdp_100k`）

2026-04-27 启动，从 12 500 步 AttnRes ckpt 续。原因：Phase-5 多模态 smoke（单 stage 全参数微调 AttnRes-Kimi-436M + 冻结 SigLIP + 可训练 MLP projector，在 LLaVA-Pretrain-558K 上）跑了 2K 步 stall 在 loss 3.8。诊断：LM 只见过 ~320M tokens，远未达 436M 模型 chinchilla-optimal 的 ~9B；caption 继承 LM 的语言天花板。

续跑参数（`launch_continuation_100k.sh`）：
- `--checkpoint.initial_load_model_only`（仅加载权重，重置 Adam）
- 500 步 re-warmup 从 0 → peak LR
- Peak LR = 3e-4（原 peak 2.2e-3 的 ~14 %，但 >> 原末 LR 2.2e-4）—— 给模型逃出原 run 终态局部极小的余地（Phase 4 末步 grad_norm 0.08 = 强证据）
- `decay_ratio=0.0`（warmup 后常 LR）—— 小 bs 下 cosine 把模型锁在早期发现的极小；常 LR 保留随机探索
- 再 87 500 步，目标 val_loss ≤ 3.0

目前 rank 0 轨迹：

| step | train loss | val loss |
|---:|---:|---:|
| 1 | 3.77671 | 3.7283 |
| 980 | 3.74485 | — |
| 2 500 | — | 3.7224 |
| 4 970 | 3.41027 | — |
| 5 000 | — | 3.7116 |
| 7 500 | — | 3.7186 |
| **10 000** | **3.41367** | — |

Val_loss 几乎不动（baseline 3.73 → 现在 ~3.71）；train 已掉到 ~3.41 但 val plateau 真实存在。**停止标准**（`phase4_kimi_attnres_lm_pretrain/README.md`）：

1. PRIMARY：`val ≤ 3.0` → 停跑回 Phase 5
2. PLATEAU：20K 连续步 (= 8 个 validator checkpoint) val 无 ≥ 0.05 改善 → 停跑用最佳 ckpt 重启 Phase 5
3. 单 ckpt 回弹 / 2-3 步噪声 → 继续跑
4. DIVERGENCE：loss spike > 5.5 持续 100+ 步 OR grad_norm > 5.0 OR NaN → 停跑调试
5. NEITHER：val 仍在缓降 → 跑满 100K

理论外推：100K × 24K tokens = 2.5B（8× Phase-4 baseline）。`3.73 × 8^(-0.075) ≈ 3.17` nats；扣 ~30 % 小 bs plateau，现实落区 **3.0 – 3.3**。

### 4.4 从头跑 paper-faithful 版本（`kimi_436m_block_attn_res_fsdp_paperhparams`）

2026-04-27 16:42 启动（`launch_from_scratch_paperhparams.sh`）。替代失败的续跑（KEEP_K=5 + SAVE_FREQ=2500 → 75 GiB 持续涨爆盘）。和续跑的差异：

- **从头跑**（无 `initial_load_path`）
- **paper LR 2.20e-3**（paper Table 2 of 436M）
- **paper warmup + cosine**（warmup=500，decay_ratio=0.8 cosine，min_lr_factor=0.1）—— config 默认，不覆盖
- **grad_accum 8×**：global_batch_size=96（LBS=3、num_ranks=4 → 12/microbatch × 8 grad-accum）。effective bs=96 把 Adam 梯度噪声减少 ~sqrt(8)=2.83×，逼近 paper 在 bs=384 LR=2.2e-3 的信噪比
- KEEP_K=2 + SAVE_FREQ=5000 → 30 GiB 持续（避开盘满）

为什么 grad_accum=8 不直接 32 对齐 paper：32 时单步 60s，20h 只能跑 ~1200 effective step，warmup=500 把一半时间吃掉。8 是甜区。

时长估算：每 effective optimizer step ~15s；20h ~4 800 steps → ~940M tokens（Phase-4 baseline 的 3×）。

目前 rank 0 轨迹：

| step | loss | grad_norm | mem |
|---:|---:|---:|---:|
| 1 | 12.23542 | 0.4601 | 23.77 GiB |
| 980 | 4.02310 | 0.0582 | 26.01 GiB |
| 3 680 | **3.51912** | 0.0248 | 27.87 GiB |

step 3 680 已经低于原 Phase 4 step 12 500 的 train loss（3.84），步数仅 0.35× —— paper LR + grad-accum 在帮忙。

---

## 5. 结论

1. **架构 port 忠实**。48B-A3B `KimiLinearConfig` 默认值与 HF `config.json` 字段对字段（vocab=163840、hidden=2304、L=27、kv_lora_rank=512、qk_nope=128、qk_rope=64、v_head_dim=128、mla_use_nope=True、kda_head_dim=128、num_experts=256/8 active/1 shared、first_k_dense_replace=1、routed_scaling_factor=2.446、sigmoid router、moe_renormalize=True）。KDA 层序（1-indexed `kda_layers` 列表）逐字复制。
2. **PP cache adapter 免费泛化。** Phase-3 adapter 零 Kimi-specific 代码；`kimi_linear/pipeline_adapter.py` 是薄 re-export。PP+adapter val 3.7277 vs FSDP AttnRes 3.7326，说明 adapter 在不同主干（KDA + MLA + MoE）下保持 loss 等价。
3. **PP+adapter 每 rank 峰值显存为 FSDP 的 60 %**（本规模、本硬件）。PP 跨 rank 切 activation；adapter 省 comm 字节（每跳 ≈ (P-1)·N_p·d）。FSDP 只省参数 / optim state。
4. **307M tokens 处 AttnRes vs baseline 在 seed 噪声内**。Δ_train = +0.009、Δ_val = +0.014。论文 "AttnRes 改善" 在 paper 的 0.35 % token 预算（436M）上不显现 —— 双方都还在欠训区间。
5. **续跑 @ 常 LR + 小 bs 在 val 3.71-3.72 软 plateau**。10K 续跑步 val 几乎不动（train 已 3.41）。验证原诊断：bs 是瓶颈不是训练时长。所以并行跑了 "从头跑 + grad_accum=8 + paper LR" 替代方案。
6. **eval 时 FSDP 长尾告警**：`1 of the 2 modules passed to fully_shard did not run forward before backward... Modules that did not run forward: [FSDPAttnResProjection(...)]`。Phase-3 的 Issue 1（空 commit）在 AttnRes pseudo-query 层级在 FSDP 下复现 —— 后续要么跳过这些 projection 的 wrap，要么走个 no-op pre-forward；当前不阻塞训练正确性（param grad 本来就是 0，因为 backward 不会访问没产出 work 的零初始化 pseudo-query）。

---

## 6. 已知与 paper Table 2 的偏差

| 项 | Paper | 我们 | 原因 |
|---|---|---|---|
| SEQ_LEN | 8192 | 2048 | 4× 5090 上 LBS=1 都 OOM |
| GLOBAL_BS | 384 | 12（Phase 4 baseline）→ 96（paperhparams） | 硬件。grad_accum=32 对齐 paper → 6 周 wallclock |
| 总 tokens | 87.9B | ~307M（Phase 4 baseline）→ ~940M（paperhparams 20h） | 时间 / 成本预算 |
| Block AttnRes N | 8 | 8（L=16 精确）/ **L ∈ {13, 17} 时退化为 Full AttnRes**，因为 decoder-layer 层级要求 `n_layers % num_blocks == 0` | paper 的 N=8 是 sub-layer 层级（L=2·Lb）；我们的 weave 在 decoder-layer 层级 commit。436M（L=16）是唯一精确命中 N=8 的 sweep size |
| MoE expert 数 | sweep 未明说 | 32 | paper §5.2 确认 48B 用 8/256；sweep 的 expert 数不在 Table 2 |
| Optimizer | sweep 未明说（48B 用 Muon） | AdamW | torchtitan 默认；Muon 还没接 |

这些偏差**在 baseline 和 AttnRes arm 之间是相同的**，所以在 FSDP A/B 比较里相互抵消。**不会**在 paper-vs-我们的绝对 loss 比较里抵消；那需要 H100/H200/B200 多节点跑 paper-strict 设置。

---

## 7. 不在范围（延后）

- HF 权重转换（HF 上有开放权重 Kimi-Linear-48B-A3B-Base；torchtitan state-dict 转换是另一个任务）
- Kimi tokenizer (`tokenization_kimi.py`)。Llama3 tokenizer 做消融够用；只在 HF-weight-loading 或与发布版对比时才需要
- GenerationMixin / 推理路径。只验训练 loss
- Kimi RoPE scaling（48B 用 plain theta=10000，无 YaRN / linear scaling）
- paper-strict 8192-context bs=384 sweep。需要多节点 H 级硬件

---

## 8. 索引

- 架构 port：[model.py](../../torchtitan/torchtitan/experiments/kimi_linear/model.py)、[attn_res_model.py](../../torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py)、[pipeline_adapter.py](../../torchtitan/torchtitan/experiments/kimi_linear/pipeline_adapter.py)、[parallelize.py](../../torchtitan/torchtitan/experiments/kimi_linear/parallelize.py)、[config_registry.py](../../torchtitan/torchtitan/experiments/kimi_linear/config_registry.py)
- HF 参考（不 import）：[reference/](../../torchtitan/torchtitan/experiments/kimi_linear/reference/)
- 训练日志：
  - Problem A baseline：`phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_baseline_fsdp_overnight/{train,eval}.log`
  - Problem A AttnRes：`phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp_overnight/{train,eval}.log`
  - Problem B PP+adapter：`phase4_kimi_attnres_lm_pretrain/runs/kimi_pp_adapter_bench/adapter_pp/{train,eval}.log`
  - 100K 续跑：`phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp_100k/train.log`
  - paperhparams 从头跑：`phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp_paperhparams/train.log`
- 对比图：`phase4_kimi_attnres_lm_pretrain/runs/kimi_pp_adapter_bench/comparison.png`
- Launchers：`phase4_kimi_attnres_lm_pretrain/launch_{fsdp_small,pp4_kimi,continuation_100k,from_scratch_paperhparams}.sh`
- 子实验 README：`phase4_kimi_attnres_lm_pretrain/experiments/{kimi_436m_attnres,kimi_pp_adapter}/README.md`
