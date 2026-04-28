# Phase 5 报告 — 多模态 AttnRes-Kimi-VL（双 arm 验证）

**日期**：2026-04-27（脚手架）→ 2026-04-28（handoff 文档 + Arm 1 初次 smoke 结果）
**状态**：**Arm 1 脚手架完成，2K 步 smoke stall（LM 瓶颈）→ 触发 Phase 4 100K 续跑。Arm 2 已写完自包含 handoff，等另一台租来的机器执行**
**硬件**：原 4× RTX 5090 PCIe（Phase 4 续跑在跑）+ 另一台租来的 4 卡机（每卡 ≥16 GB 即可）跑 Arm 2

废弃的 KD/MiniPLM phase 移到 `phase5_distillation_deprecated/`（保留作为负面结果记录）。这里说的是**双 arm 多模态** phase。

---

## 1. 目标

把 **Phase-4 AttnRes-Kimi-436M ckpt** 当作 LLaVA 风格多模态模型的 LM 主干（SigLIP 冻结 + 2 层 MLP projector + AttnRes-Kimi-436M LM，单 stage 全参数微调，LLaVA-Pretrain-558K 数据）。沿用 Phases 3/4 在单模态阶段用过的**双 arm 模式**：

```
                同数据：LLaVA-Pretrain-558K
                同模型：SigLIP 冻结 + MLP projector + AttnRes-Kimi-436M
                同 ckpt 初始化
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
       Arm 1（主，质量）              Arm 2（系统，新颖）
       FSDP2=4，PP=1                  FSDP2=1，PP=4 V=2 + cache adapter
       3 epoch，~5h overnight          5k-10k 步 smoke
       多模态模型收敛                  跨模态 cache 不变性验证
```

**Arm 2 是头条新结果** —— 据项目作者的文献检索，这是开源界**第一次验证 AttnRes 跨 stage caching adapter 在混合视觉+文本变长 padding 序列下保持 loss 不变性**。Kimi 内部团队大概率早做完了；公开没有写 up。

非目标：在 LLaVA-1.5 / 1.6 上打过 benchmark（436M LM 比 Vicuna-7B 小 10-20×，VQA 不可能竞争 —— 重点是集成不是分数）；多模态阶段再做一次 AttnRes-vs-vanilla A/B（已经在 Phase 4 单模态阶段做完）。

---

## 2. 交付物

### 2.1 工作区（`phase5/`，**不**进 PR）

| 文件 | 作用 |
|---|---|
| `README.md` | 双 arm 规格、为什么不是 re-A/B、架构图、Arm 1 + Arm 2 配置、双机并行排期、数据 + tokenizer 细节、文件分工 |
| `HANDOFF_arm2_pp_adapter.md` | **38 KB 自包含 handoff**，给租来的机器上的另一个 Claude session 用。覆盖：项目背景、硬件预算、环境搭建、4 个工程 gap、init 策略菜单、代码地图、**12 个高概率 bug 的 debug 清单（10.1 → 10.12）** |
| `data_prep.py` | LLaVA-Pretrain-558K 通过 `huggingface_hub.snapshot_download` 下载（`HF_HUB_DISABLE_XET=1` 绕过已知的 xet client 线程死锁），解压图片（~28 GB），bucket 落位检查 |
| `multimodal_dataset.py` | `LlavaPretrainDataset` `IterableDataset`：每条样本 `[<img> × N_vision] [BOS] [caption tokens] [EOS]`，`IMAGE_TOKEN_ID=32000`（Llama-3.1 保留 special token），`N_VISION_TOKENS=196`（SigLIP-Base @ 224×224 patch16 → 14×14），`IGNORE_INDEX=-100` 在 image + BOS；按 (dp_rank, world_size) 分片，无限循环；`collate_with_pad` 拼 batch |
| `multimodal_model.py` | `Projector`（2 层 MLP `vision_dim → lm_dim → lm_dim`，GELU，trunc-normal init）；`multimodal_loss`（vision_tower `no_grad` 前向、projector 可训练、LM 单 FSDP-root 调用，视觉 token scatter **在 `lm.forward` 内部**）；强制 "每行恰好 N_vision 个 image token" 的不变量 |
| `train_mm.py` | `MultimodalTrainer` 继承 torchtitan `Trainer`。`__init__` 里加载冻结的 vision_tower + tokenizer + image_processor + projector AdamW；替换 dataloader；override `forward_backward_step`。**当前在 PP 时 raise NotImplementedError**（line 176-179）—— Arm 2 第一件事就是删它 |
| `launch_train.sh` | Arm 1 启动器（FSDP=4，PP=1） |
| `launch_pp_adapter.sh` | Arm 2 启动器 —— **暂未存在**；HANDOFF 文档 § 11 给了模板 |
| `eval_caption.sh` | caption loss + 简单 VQA 准确率（小 held-out 集） |
| `tests/__init__.py` | 占位；还没写测 |

### 2.2 LM 主干复用

`torchtitan/` 里没有新生产代码。Arm 1 复用：
- `torchtitan/experiments/kimi_linear/attn_res_model.py:KimiLinearAttnResModel` —— 已经接收 `vision_embeds + image_mask` kwargs；line 263-267 在 `embed_tokens` 之后做 scatter
- `torchtitan/experiments/kimi_linear/parallelize.py` 做 FSDP 包装
- `torchtitan/experiments/attn_res/pipeline_adapter.py:pipeline_llm_with_cache_adapter` —— Arm 2 需要扩展它的 kwargs 转发，把 `vision_embeds + image_mask` 只送到 stage 0

### 2.3 废弃的 KD phase

`phase5_distillation_deprecated/`（工作区根下的 `phase5/` 兄弟目录）：MiniPLM 风格知识蒸馏实验，已废弃。保留作为负面结果记录；不属于当前双 arm phase。

---

## 3. 架构（双 arm 共用）

```
SigLIP-Base-Patch16-224（冻结，~92M params）
   image (224×224)
   └─ 196 vision tokens × 768 dim
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ Projector：2 层 MLP             │   从头训练
   │   768 → 1168 → 1168 (Kimi dim)  │   ~3M params
   └─────────────────────────────────┘
                   │
                   ▼
   [196 个图像对齐 token，LM space]
                   │
                   │ 插入 <image> sentinel 位置
                   ▼
   AttnRes-Kimi-436M（Phase 4 ckpt）
                                          全参数可训练
                   │
                   ▼
   caption tokens 自回归（CE loss 仅算 text token，
   image-token 位置 labels = -100）
```

- **冻结**：SigLIP（`google/siglip-base-patch16-224`）
- **可训练**：projector（随机 init）+ LM（Phase 4 ckpt init），全参数
- **Loss**：CE 仅算 caption text token；image 位置 `labels = IGNORE_INDEX = -100`

---

## 4. Arm 1 — FSDP2=4，PP=1（主，质量）

### 4.1 配置

单 stage 端到端（不分 LLaVA stage-1 projector 预训练）。LLaVA-1.5 两阶段配方假设 LM 已经很强（Vicuna-7B）+ projector 完全随机；我们 436M 部分预训练 backbone，直接单 stage。

| 项 | 值 |
|---|---|
| 硬件 | 4× RTX 5090，FSDP2 4 ranks（LM），PP=1 |
| SigLIP | 冻结，~0.3 GB / rank，复制 |
| Projector | ~0.01 GB / rank，复制，可训练 |
| LM | Kimi 436M FSDP-shard，~0.5 GB 权重 / rank + activation + AdamW state |
| 单图 | 196 vision token + ~30 caption token ≈ 226 token seq |
| `local_batch_size` | 8 |
| `seq_len` | 256（短 caption） |
| `TORCHTITAN_ATTNRES_CACHE` | 不设（adapter 关） |
| 吞吐目标 | 4× 5090 上 ~5-10K 图文对 / 分钟 |
| 单 epoch 时间 | ~93 分钟（558K / 6K/min） |
| 总时长 | 3 epoch ≈ 5h overnight |

### 4.2 状态

- 初次 Arm-1 smoke 跑了 2K 步，**caption loss stall 在 ~3.8**。
- 诊断：LM 是瓶颈。Phase 4 之后 LM 只见过 ~320M tokens（12 500 × 12 × 2048），远未达 436M 模型 chinchilla-optimal 的 ~9B；caption 继承 LM 的语言天花板，多模态实验在 LM 没强起来之前根本验证不了 AttnRes。
- **这就是 Phase 4 100K 续跑（`launch_continuation_100k.sh`）被启动的诊断起点**，目标 C4 val_loss ≤ 3.0。Arm 1 卡在续跑结果上。
- Phase 4 续跑当前在原机器上跑；Phase 4 平行的 "从头跑 + grad_accum=8 + paper LR" 替代方案也在跑。

### 4.3 复现（等 Phase 4 ckpt 出炉后）

```bash
# Step 1: 数据 + vision tower（~12 GB，普通带宽 ~30 分钟）
python phase5/data_prep.py

# Step 2: smoke（5 步，单卡）
STEPS=5 LOCAL_BS=2 bash phase5/launch_train.sh

# Step 3: 全过夜（~5h）
bash phase5/launch_train.sh

# Step 4: eval ckpt
bash phase5/eval_caption.sh
```

---

## 5. Arm 2 — FSDP2=1，PP=4 V=2 + cache adapter（系统，新颖）

### 5.1 配置

| 项 | 值 |
|---|---|
| 硬件 | 4× GPUs（每卡 ≥16 GB 够 436M cache-adapter；32 GB 留 SEQ=2048 余地） |
| `parallelism.pipeline_parallel_degree` | 4 |
| `parallelism.pipeline_parallel_schedule` | Interleaved1F1B（cache-adapter 前置） |
| `parallelism.pipeline_parallel_layers_per_stage` | 2（V=2 × lps=2 = 8 个 virtual stage，每个 block 边界与 stage 边界对齐） |
| `TORCHTITAN_ATTNRES_CACHE` | 1（adapter 开） |
| `local_batch_size` | 1（PP 装得下） |
| `global_batch_size` | 12（= num_microbatches；≥ V·PP = 8 满足 Interleaved1F1B 的 lookahead） |
| `seq_len` | 258（196 vision + 60 caption + bos + eos） |
| LR | 1e-5（从 ckpt 全参数微调；很小） |
| `data_parallel_shard_degree` | 1（无 FSDP 分片，复制） |

### 5.2 每 rank 显存预算（PP=4 V=2，LBS=1，SEQ=258，FSDP=1 复制）

| 组件 | rank 0 | rank 1-2 | rank 3 |
|---|---|---|---|
| LM 4 层（bf16） | ~216 MB | ~216 MB | ~216 MB |
| LM AdamW state | ~1.5 GB | ~1.5 GB | ~1.5 GB |
| `embed_tokens`（vocab × hidden） | ~300 MB | — | — |
| `lm_head` + AdamW | — | — | ~2.1 GB |
| `final_attn_res_*` + AdamW | — | — | ~30 MB |
| `vision_tower`（冻结） | ~184 MB | — | — |
| `projector` + AdamW | ~50 MB | — | — |
| PP cache（最坏 8 块 × 12 mb） | ~70 MB | ~250-400 MB | ~700 MB |
| Activations（SEQ=258） | ~300 MB | ~300 MB | ~500 MB |
| PyTorch CUDA reserved | ~1-2 GB | ~1-2 GB | ~1-2 GB |
| **rank 总计** | **~3.7 GB** | **~3.5 GB** | **~6.5 GB** |

每块大小：`B × T × D × 2 (bf16) = 1 × 258 × 1168 × 2 = 0.6 MB`。rank-3 cache 总：96 × 0.6 MB = **58 MB** —— 轻松装下。

### 5.3 与 Phase 4 续跑独立 — 三种 init 策略

Arm 2 测的是 PP-vs-FSDP loss **delta**（相同步数），绝对 loss 无关。所以三个 init 策略都行（推荐顺序 **A → B → C**）：

- **Strategy A（随机 init，推荐先做）**：不加载 ckpt。loss 起步 ≈ log(vocab) ≈ 11.7，梯度动力学大 → PP-adapter 与 FSDP 任何数值偏差立刻显形。跑 1-2K 步；通过标准 `|Δ| ≤` Phase 3 测得的 FSDP seed-vs-seed 噪声带（~0.13 nats）。
- **Strategy B（弱 Phase 4 ckpt）**：把 `phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500`（~15 GB）拷过去。loss 起步已接近底部（~3.7 train），梯度小。测 adapter 在小梯度 regime 下行为。3-5K 步。
- **Strategy C（续跑后 ckpt）**：等 Phase 4 续跑跑完（val ~3.0 目标）。最贴近 Arm 1 hand-off 现实，但阻塞。

### 5.4 4 个工程 gap（Arm 2 的具体工作）

`HANDOFF_arm2_pp_adapter.md` §§ 6-9 详细写了。

1. **Gap 1 — 删 PP 禁用 guard**（5 分钟）：删 `train_mm.py:176-179` 的 `NotImplementedError`。必要不充分 —— 删了之后 gap 2/3/4 浮上来。
2. **Gap 2 — PP 下视觉 scatter 时机**（中等，2-3 天）：FSDP=1 PP=1 下 scatter 在 `lm.forward` 内部，FSDP root 看到一次 forward 调用。PP=4 下 `lm.forward` 被切成 4 stage，scatter 必须在 stage 0 把 hidden state 发给 stage 1 之前完成。两个修法：**A（推荐）** 让 `vision_embeds + image_mask` kwargs 通过 `pipeline_llm` 的调度路由到 stage 0 的 submodule（HANDOFF 文档给了具体扩展点）；**B** 在 trainer 里预 scatter，传 `inputs_embeds=h`（带来 FSDP/PP 接线复杂度）。
3. **Gap 3 — PP 下变长序列**（中等，1-3 天）：`PipelineSchedule` P2P 发的是定形 tensor，recv buffer 按第一个 microbatch 形状预分配。caption token 长度 5-60 抖动 → 后面 mb 形状不一致 crash（典型 NCCL 信息 "Tensors must have the same shape"）。**修法**：把 `collate_with_pad` 改成 pad 到 `GLOBAL_MAX_LEN = 196 + 60 + 2 = 258`（跨 microbatch 确定）。caption > 60 直接砍（`max_caption_tokens` 已经在做）。padding 浪费一些算力；动态形状 PP 重构成本太高。
4. **Gap 4 — Cache adapter 跨模态 smoke**（关键 milestone，3-5 天 + debug）：开 `TORCHTITAN_ATTNRES_CACHE=1`；同时跑 PP=4 V=2 + FSDP=4 baseline（同 seed）。验证：`|loss_pp_adapter[step] − loss_fsdp_baseline[step]| ≤ noise_band`。对齐通过 → 出货。对齐失败 → 根因分析 + 把失败模式当结果出。

总计预估 **2-3 周** 真实工程量（租来的 4 卡机上）。

### 5.5 高概率 bug 清单（debug 检查表 10.1-10.12）

前一个 agent 的分析列了 12 个候选失败模式。每个都有 **症状 → 检测 → 修法**。要点：

- **10.1 `pipeline_llm` 没把多模态 kwargs 转到 stage 0** —— stage 0 处 vision_embeds=None；loss 用 `embed_tokens(IMAGE_TOKEN_ID)`（随机初始化的那一行）算。
- **10.3 vision_embeds 没跟着 microbatch split** —— `(B_global, 196, 1168)` 到 stage 0 还是 B_global，input_ids 已经被切成 B_micro → 形状不匹配 crash。**最可能的失败模式。** 修法：把 `pixel_values` 放进 PP 知道怎么切的 input dict，在 stage 0 forward 内部跑 vision_tower + projector。
- **10.4 projector 不是 stage 0 的子模块 → 梯度不累计** —— projector 不在任何 stage 的 module tree 里 → forward 图与 stage 0 backward 脱钩 → 梯度 0。修法：把 projector 包进 stage 0 的 submodule。
- **10.6 attn_res_proj 零初始化在多模态 step 0** —— projector 起手随机 → vision 位置 hidden norm 与 text 位置差很多 → 1 步梯度后 attn_res_proj 偏离零，权重不再均匀，梯度尖刺。**大概率不需要修**（零初始化保证 step 0 softmax 均匀，与大小无关），但留心早期 grad_norm 突变。
- **10.7 cache 跨 batch 漏内存** —— `_install_step_drop_patch` 应该照常触发；多模态下要复测。
- **10.8 第一个变长 mb 时 recv buffer 形状不匹配** —— 见 Gap 3。
- **10.10 视觉位置残差梯度流不一致** —— cache adapter 的 `_LocalCacheCapture` 当初是为纯文本设计的；要验证它在视觉位置的梯度累加正确。检测：相同步数下比较 FSDP 与 PP-adapter 的 `projector.fc1.weight.grad.norm()`。

清单的处理原则：每一项最后要么**已修**要么**确认不存在 + 为什么**。"smoke 跑通了但说不清每个 bug 触发了没"**不是结果**。

---

## 6. 双机并行排期

| 周 | 原机（Phase 4 续跑） | 租机（Phase 5 Arm 2） |
|---|---|---|
| W0（现在） | Phase 4 续跑进行中 | 租机、环境搭建、gap 1+2 + 1-microbatch PP=2 smoke |
| W1 | Phase 4 续跑 | gap 3+4 + 2K 步 fresh-init 对齐测 |
| W2 | Phase 4 续跑结束 | 3-5K 步 weak-ckpt-init 对齐测 + 出图 |
| W3 | Arm 1 FSDP overnight（加载新 ckpt） | Arm 2 写 up + commit |
| W4 | Arm 1 结束，写 phase 5 终稿 | — |

**合计端到端 ~4 周**，串行做要 ~6-8 周。

---

## 7. 阶段性结论（截至 2026-04-28）

1. **多模态脚手架在 FSDP=4 单 stage 558K 图 LLaVA 设置下端到端跑得起来**。CE loss 忽略 image + BOS 位置正确接通。
2. **Arm-1 初次 smoke stall 在 caption loss 3.8** 因 LM 欠训。诊断顺利追到 LM val_loss 3.73 → 触发 Phase 4 100K 续跑。**没有 Phase 4 续跑 Arm 1 没用。**
3. **Arm 2 已完整规格化但未执行**。总工作：4 个工程 gap + 12 bug debug 清单；2-3 周租机时间。
4. **PP-adapter 跨模态对齐潜在是开源界第一**，按项目作者文献检索 —— Megatron 开源多模态配方**没解决**这个（它们复制 vision tower，PP 只在 LM 上跑、全形状 send/recv pad-to-global-max）。在多模态上叠加 cache adapter 是真正新内容。

---

## 8. 索引

- 工作区：[phase5/](../../phase5/)（data_prep、multimodal_dataset、multimodal_model、train_mm、launch_train、eval_caption、README）
- Arm 2 自包含 handoff：[phase5/HANDOFF_arm2_pp_adapter.md](../../phase5/HANDOFF_arm2_pp_adapter.md)
- 复用的 LM 主干：[torchtitan/experiments/kimi_linear/attn_res_model.py](../../torchtitan/torchtitan/experiments/kimi_linear/attn_res_model.py)、[torchtitan/experiments/attn_res/pipeline_adapter.py](../../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py)
- 废弃的 KD：[phase5_distillation_deprecated/](../../phase5_distillation_deprecated/) —— 保留作为负面结果，不属于当前双 arm phase
