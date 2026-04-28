# Phase 2 报告 — 单卡 Block AttnRes 损失曲线对齐

**日期**：2026-04-19（主跑）/ 2026-04-20（N-消融）/ 报告 2026-04-28
**状态**：**已完成** —— 主 A/B 已采证，N-sweep 消融已闭环
**硬件**：1× RTX 5090（32 GB），单卡

---

## 1. 目标

在完全相同的超参下分别训练 175M Llama3 baseline 和 175M Block AttnRes 变体，证明 AttnRes 训练损失曲线肉眼可见地低于 baseline。这是必须先于 Phase 3 PP 工作和 RFC PR 落地的正确性门槛。

附加：扫 `num_blocks` ∈ {3, 6, 12}，让 RFC PR 描述里有一份小消融，证明在该规模下 N≈8（论文宣传值）并不是 "AttnRes < baseline" 的关键变量。

---

## 2. 交付物

**工作区脚本**（`phase2/`，**不**进 torchtitan PR）：

| 文件 | 作用 |
|---|---|
| `setup_env.sh` | 创建 conda env `attnres`，装 torch nightly + torchtitan editable，拉 Llama-3.1 tokenizer（默认走 `NousResearch/Meta-Llama-3.1-8B` 非门控镜像），跑独立 smoke + torchtitan 单测 |
| `smoke_test_attn_res.py` | 纯 torch 内联实现 `block_attn_res()`。4 项自检：identity（单 partial）、零初始化 proj 时 softmax 均匀、梯度流、多层 block commit、零初始化 pseudo-query 下输出/loss/梯度有限 |
| `launch.sh` | tmux 4 窗启动器：`baseline` → `guardian`（守 baseline log 里的 `Training completed`，触 `DONE`）→ `attn_res`（等 `DONE`）→ `monitor`。`DATA_FRAC` 环境变量按比例缩 steps（用于 `runs_1_8th/`） |
| `launch_ablation.sh` | 用 `;` 不用 `&&` 串联，单变体崩了不会阻塞下一个；每个 variant 写一份 `STATUS` 记录 torchrun rc。默认变体 `n3` + `n12`（两端，主 `n6` 已在 `runs/attn_res/` 跑过） |
| `compare_losses.py` | 读两个 run 的 TB events，画 3 联图（全曲线 / 暖机后放大 / 同步 step 差），打印关键 step 的 delta |
| `plot_ablation.py` | N-sweep 叠加图 |
| `README.md` | runbook（环境、dry-run 验证、smoke、主跑、监控、对比、排错） |

**可提交代码**（`torchtitan/experiments/attn_res/`）：

| 文件 | 作用 |
|---|---|
| `attn_res.py` | `block_attn_res()` 原语（论文 Figure 2）、`AttnResProjection`（D→1，pseudo-query 零初始化）、`stack_blocks` / `unstack_blocks` |
| `model.py` | `AttnResLlama3TransformerBlock` 与 `AttnResLlama3Model` 子类——核心 `decoder.py` / `model.py` 不改 |
| `__init__.py` | model flavors：`debugmodel_attn_res`、`175M_attn_res`（默认 N=6）、`175M_attn_res_n{3,4,12}`、`175M_attn_res_L16_n8` |
| `config_registry.py` | trainer config：`llama3_175m_baseline`、`llama3_175m_attn_res` |
| `tests/test_attn_res.py` | 单测（原语、projection、stack/unstack、dense model、decoder 集成、init 契约） |

---

## 3. 配置（脚本里 commit 死的）

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | Llama3-175M（自定义 flavor） | `torchtitan/experiments/attn_res/__init__.py` |
| Tokenizer | Llama-3.1（NousResearch 非门控镜像） | `setup_env.sh` |
| 数据 | C4（HF stream） | torchtitan 默认 |
| `local_batch_size` | 8 | `launch.sh`（默认 16 在 5090 上 OOM，因为 logits 是 `[B*T, V=128256]` fp32；用 grad accum 保等效 bs） |
| `global_batch_size` | 16 | `launch.sh`（grad accum = 2） |
| 序列长度 | 2048 | torchtitan 默认 |
| 步数 | 20 000 | `launch.sh` 默认 |
| 种子 | torchtitan 默认 | baseline / AttnRes 之间故意不固定（当噪声底） |
| `dtype` | bf16 | torchtitan 默认 |
| 优化器 | AdamW | torchtitan 默认 |
| LR schedule | cosine，200 step warmup，默认 decay | torchtitan 默认 |

---

## 4. 验证结果

### 4.1 主 A/B（`runs/baseline` vs `runs/attn_res`）

各 20 000 步，约 650 M tokens 每跑。

| step | baseline loss | attn_res loss | delta |
|---:|---:|---:|---:|
| 1 | 12.26907 | 12.26127 | −0.008 |
| 990 | 5.26878 | 5.17480 | −0.094 |
| 4990 | 4.24073 | 4.16045 | −0.080 |
| 9990 | 4.09605 | 4.03300 | −0.063 |
| 19990 | 3.88939 | 3.84742 | −0.042 |
| 20000 | 3.68482 | 3.61859 | **−0.066** |

两边都打印了 `Training completed`，delta 全程为负，对比图保存在 `runs/comparison.png`。

时间和吞吐：
- Baseline：5090 上 ~2h42m，~71.2 K tps，MFU ~15.5 %，峰值显存 29.1 GiB。
- AttnRes：~3h41m，~50.1 K tps，MFU ~10.9 %，峰值显存 30.05 GiB。

每步 ~30 % 的 tps 损失符合预期（论文 §3.2）：每个 sub-layer 现在多做一次 (N+1) 个 block 上的 stack + RMSNorm + einsum + softmax + 加权和。这部分开销在 PP（cross-stage cache）和 `torch.compile` 下会大幅回血——本期不在范围内。

### 4.2 全路径快速验证（`runs_1_8th/`）

`DATA_FRAC=0.125` → 各跑 2 500 步：
- baseline 4.82641
- attn_res 4.71197（同步 step 差 −0.114）

方向一致、量更大（早期曲线特性），证明启动路径完整再开 overnight 跑。

### 4.3 N-sweep 消融（`runs/ablation/`）

各 20 000 步，超参完全一致，只改 `num_blocks`。

| 变体 | num_blocks | layers_per_block | step 20000 loss | tps | MFU | mem |
|---|---:|---:|---:|---:|---:|---:|
| `attn_res_n3` | 3 | 8 | **3.65491** | 52 664 | 11.5 % | 29.88 GiB |
| `attn_res`（n6） | 6 | 4 | **3.61859** | 49 412 | 10.8 % | 30.05 GiB |
| `attn_res_n12` | 12 | 2 | **3.62343** | 26 437 | 5.8 % | 29.90 GiB |
| `baseline` | — | — | 3.68482 | 70 660 | 15.4 % | 29.11 GiB |

三个 N 都打过 baseline。N=6（在我们 L=24 setup 下相当于论文 "≈8" 甜区）以微弱优势胜出；N=3 只差 0.04，N=12 居中。N=12 吞吐塌方（每个 sub-layer 的 stack 与 N 线性，12 块缓存把 layer 成本拉爆）。

第一次 `n12` 试跑（`llama3_175m_attn_res_n12_crashed_20260419/`）中途吃到 HF httpx 短暂错误（`Cannot send a request, as the client has been closed`）；消融串联脚本继续推进，重跑（`llama3_175m_attn_res_n12/`）干净跑完，替代了那次 crash 的位置。

### 4.4 独立 smoke + 单测

`smoke_test_attn_res.py` 和 `torchtitan/experiments/attn_res/tests/test_attn_res.py` 都在 `setup_env.sh` 流程中通过。覆盖：

- 原语 identity（单 partial + 零初始化 proj → identity-like）
- 零初始化 pseudo-query 时 softmax 均匀 → step-0 数值上等价于标准残差
- 通过 stack/cat 的梯度流
- 多层 block commit（3 个已 commit 块 + 1 partial → 4 源聚合）
- 每层参数的输出/loss/梯度有限

---

## 5. 结论

1. **原语层面 Block AttnRes 实现正确。** 末步 −0.066 train loss delta（650 M tokens）是 RFC PR 的正确性证据。方向定性对得上论文 "AttnRes ≈ baseline × 1.25 compute at matched size"；绝对量级不可直接比，论文用的 scaling-law setup 不一样。
2. **Pseudo-query 必须零初始化。** 单测 `test_pseudo_queries_are_zero_after_init` 显式守这一点。step-0 非零 pseudo-query → softmax 非均匀 → init 时残差被打散 → bf16 下训练抖动 / NaN。
3. **在 175 M / 650 M tokens 这个规模上 N 不是关键变量。** N ∈ {3, 6, 12} 都打过 baseline；N=6 比 N=3 好 ~0.04，比 N=12 好 ~0.005。论文 "N≈8" 的工程价值在 PP 规模下（Phase 3）才显现，单卡这步只是定性。
4. **单卡每步开销 ~30 %。** 正确性实验可接受；RFC PR #2 头号宣称要靠 PP 把这部分回血。

## 6. 解锁的下游工作

- Phase 3（PP cache adapter）：单卡正确性基线已立。
- RFC PR 描述：`comparison.png` + 训练日志末尾 + 参数计数三件套。
- `attn_res/__init__.py` 里的 Block AttnRes flavors 在 Phase 4（Kimi Linear backbone）可不修改直接复用。

## 7. 索引

- 训练日志：`phase2/runs/{baseline,attn_res}/train.log`、`runs/ablation/*/train.log`
- 图：`phase2/runs/comparison.png`、`runs/ablation/comparison.png`、`runs_1_8th/comparison.png`
- Smoke：`phase2/smoke_test_attn_res.py`
- Runbook：`phase2/README.md`
- 代码：[torchtitan/experiments/attn_res/](../../torchtitan/torchtitan/experiments/attn_res/)
