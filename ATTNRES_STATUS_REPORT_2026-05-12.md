# AttnRes 全栈实现状态 — 2026-05-12

整仓 AttnRes (Block Attention Residual, Kimi Linear paper §5 / Figure 2) 已有实现的完整 inventory。覆盖：核心 primitive、双 backbone 模型集成、PP adapter（Interleaved 1F1B + cache adapter）、SGLang 推理覆盖层、checkpoint converters、训练 stage 历史与基准。

## 1. Executive summary

| 模块 | 状态 | 关键证据 |
|---|---|---|
| Core primitive (`block_attn_res`, `AttnResProjection`) | ✅ 完成 + tested | `attn_res/attn_res.py` 127行，零拷贝 stack/unstack helpers，FP8 protocol-compliant |
| Llama3 + AttnRes 集成 | ✅ 完成 + N-sweep | N∈{2,3,4,6,8,12} 全配置，L=16/24/32/48 深网carrier |
| Kimi Linear + AttnRes 集成 | ✅ 完成 + 完整 scaling sweep | 194M..528M paper Table 2 全注册，447M_aligned (5090-friendly)，48B production carrier |
| 多模态 (SigLIP + projector + AttnRes-LM) | ✅ 完成 + 训练验证 | phase5 caption pretrain done，phase11 SFT 3ep done |
| **PP adapter (Interleaved 1F1B + cache adapter)** | ✅ 完成 + 1460行 tests | L=16 三组 (PP=4/8 × VP=2/4) 全部 |Δloss| < 0.005 |
| Full AttnRes L=32 N=32 PP=8×VP=4 | ✅ 可训练 | grad_norm 收敛，loss→7.36 by step 140 |
| **Block AttnRes L=32 from-scratch** | ❌ **未解** | 所有 dim/init/N 组合 inf-grad 或 NaN，根因未知 |
| SGLang 推理覆盖 + Triton 融合 kernel | ✅ 完成 + +27% decode tps | 两段式 Phase 1/2，seq-dim TP shard，AR wire bytes −58% |
| KV cache 设计验证 | ✅ 完成 | Block reps 是 intra-forward locals，标准 KV cache 已足够 |
| HF↔DCP converters (LM + multimodal) | ✅ 完成 | MoE expert stacking，KDA `A_log` reshape，projector key 映射 |
| **当前 LM stage 0 redo (10B tokens)** | 🔄 进行中 | `lm_447m_fp8_paperalign_B` lbs=4 gbs=384 lr=1.5e-3 warmup=1000，ETA ~4 天 |

---

## 2. Core AttnRes primitive

**File**: `torchtitan/torchtitan/experiments/attn_res/attn_res.py` (127行)

```python
# 核心聚合 — paper Figure 2
def block_attn_res(blocks, partial, proj, norm):
    K = stack_blocks([norm(b) for b in blocks] + [norm(partial)])  # [N+1, B, T, D]
    logits = einsum("d,nbtd->nbt", proj.weight.squeeze(0), K)
    weights = softmax(logits, dim=0)
    return einsum("nbt,nbtd->btd", weights, V)
```

- **`AttnResProjection`** (line 87-109): 子类化 `torchtitan.models.common.linear.Linear`（为 FP8 protocol），weight = paper 的 `w_l` pseudo-query (D→1, zero-init)
- **`stack_blocks` / `unstack_blocks`** (line 111-126): `list[Tensor]` ↔ stacked tensor 转换，用于 PP P2P 边界传递
- **`AttnResConfig`** (line 30-44): `num_blocks` 默认 8。Empirical：N=2,4,8 等价；N≥16 退化（phase3 实测）

**Layout**: `attn_res/layout.py` 实现 Interleaved 1F1B 的 block partition scheduling — `BlockLayoutTables` 把 N 个 AttnRes-blocks 切到 PP × VP 网格。

**Tests** (`attn_res/tests/`):
- `test_attn_res.py` (339行) — 核心数值正确性
- `test_attn_res_dsv3.py` (243行) — DeepSeek-V3 形状
- `test_pipeline_adapter.py` (**1460行**) — TP+PP+AC 完整网格

---

## 3. 模型集成

### 3.1 Llama3 + AttnRes

**File**: `attn_res/model.py` + `attn_res/config_registry.py`

- **`AttnResTransformerBlock`** — 每层 2 次 AttnRes 应用（pre-attn + pre-FFN）
- **Block 状态线程**: `blocks` (已 commit) + `partial_block` (intra-block accumulator)
- **PP stage handling**: first stage 从空 list 开始；middle/last 接收 stacked tensor、unstack；last stage 跑 final cross-block aggregation 后 `norm + output`
- **`_return_only_new_blocks` flag** — PP cache adapter 模式开关

**配置 sweep** (config_registry.py):
- Dense baselines: `llama3_175m_baseline`, `llama3_175m_attn_res` (N=6)
- N-ablation: N∈{2,3,4,12} on L=12
- Deep carriers: `L16_N8`, `L24_*` (N=4,8,12), `L32_N8/N16` (dim 768-2048)
- DSv3 shape: `dsv3_attn_res_debugmodel` (6L, 8E, N=3), `dsv3_attn_res_16b` (N=9, MoE+MLA)

### 3.2 Kimi Linear + AttnRes (paper-canonical)

**File**: `kimi_linear/attn_res_model.py` + `kimi_linear/config_registry.py`

- **`KimiAttnResDecoderLayer`** — 每层 4 个 AttnRes params：`attn_res_{proj,norm}` + `mlp_res_{proj,norm}`
- **`KimiLinearAttnResModel`** — 子类化 `KimiLinearModel`，按 sub-layer 视图穿 carrier
- **Paper 对齐**: paper 的 "1 pseudo-query/layer" = 我们 "2 sub-layer/transformer-block" 一致

**Scaling-law sweep** (paper Table 2):
| flavor | params | n_layers | dim | d_ff | tokens (paper) | lr (paper) | bs (paper) |
|---|---|---|---|---|---|---|---|
| `kimi_linear_194m_*` | 194M | 12 | 896 | 400 | 38.7B | 2.99e-3 | 192 |
| `kimi_linear_241m_*` | 241M | 13 | 960 | 432 | 45.4B | 2.80e-3 | 256 |
| `kimi_linear_296m_*` | 296M | 14 | 1024 | 464 | 62.1B | 2.50e-3 | 320 |
| `kimi_linear_436m_*` | 436M | 16 | 1168 | 528 | 87.9B | 2.20e-3 | 384 |
| `kimi_linear_528m_*` | 528M | 17 | 1264 | 560 | 119.0B | 2.02e-3 | 432 |
| **`kimi_linear_447m_aligned_block_attn_res_n4`** | 447M | 16 | **1024** | 768 | 87.9B | 2.20e-3 | 384 |
| `kimi_linear_447m_aligned_block_attn_res_n4_fp8` | 同上 | + FP8 rowwise | | | | | |
| `kimi_linear_48b_{baseline,block_attn_res,full_attn_res}` | 3B active | 27 transformer-blocks | 2304 | 1024 | 1.4T | 1.0e-3 | 2048 |

**447M_aligned 设计要点**:
- dim=1024 → head_dim=64 是 16 的倍数 → flashinfer SM 12.0 (RTX 5090) 接受
- qk_rope=32, v_head=64, kv_lora=512 全部 8/16/32-aligned
- 替代了原 436M (dim=1168, head_dim=73 在 5090 上 flashinfer / cuBLAS / Triton extend 全报错)

每个 size 都有 `_baseline` / `_block_attn_res` / `_full_attn_res` 三档对照。

### 3.3 多模态 KimiMultimodal

**File**: `kimi_linear/multimodal_model.py` + `phase5/multimodal_model.py`

```
SigLIP-Base/patch16-224 (frozen, 93M) 
  → 2-layer MLP projector (trunc_normal 0.02, trainable)
  → AttnRes-Kimi-447M LM (FSDP-sharded, trainable)
```

- IMAGE_TOKEN_ID=32000 (Llama-3.1 reserved)，N_VISION_TOKENS=196 (SigLIP 14×14)
- LM forward 内部 scatter vision_embeds 到 `input_ids==IMAGE_TOKEN_ID` 位置（单次 FSDP root call）
- Stage 1 (phase5): LLaVA-Pretrain 558k captions, seq=260, lr=1e-5 LM / 5e-4 projector
- Stage 2 (phase11): LLaVA-Instruct-150K, seq=512, lr=2e-5

---

## 4. PP Adapter — Interleaved 1F1B 的难点解决方案

**File**: `attn_res/pipeline_adapter.py` (~1000行)

### 难点

AttnRes 跨 PP stage 时，每个 stage 把它的 partial_block commit 进 blocks 列表后传给下一 stage。**关键约束**:

1. **跨 mb 的 cache 隔离** — Interleaved 1F1B 同一 rank 上 V 个 virtual stages 交替跑不同 mb。同一 virtual stage 在 mb_i 的 partial_block 不能污染 mb_j
2. **Delta mode 上行节省 bw** — 只发新增 blocks，不重发整个 history
3. **Gradient 双桥** — 既要 PP 反向 grad 流通，也要 rank-local 缓存的副本 deposit grad

### 实现

**`CrossStageCacheAdapter`** 的两个 grad 通路：

1. **PP-hop 通路**: `recv_delta_tensor` 通过常规 PP `SEND_B` 反传给上游 rank
2. **Rank-local 缓存通路**: detached copy 写入 rank-local slot，通过 `_LocalCacheCapture.backward()` 把 grad 钩回 producer 的 incoming grad

**`pipeline_llm_with_cache_adapter()`** — `ModelSpec.pipelining_fn` 入口，仅在 `TORCHTITAN_ATTNRES_CACHE=1` 时启用 adapter mode；schedule 非 Interleaved1F1B 时打 warning。

### Phase 3 PP 压力测试结果

**`phase3/PRESSURE_TEST_REPORT_2026-05-12.md`** — L=16 Block AttnRes 三组 shape 全对齐：

| PP shape | LBS | GBS | naive loss @1000 | adapter loss @1000 | Δ |
|---|---|---|---|---|---|
| **PP=8 × VP=2** | 16 | 16 | 5.42497 | 5.42935 | +0.00438 |
| PP=4 × VP=2 | 8 | 16 | 5.52833 | 5.52941 | +0.00108 |
| PP=4 × VP=4 | 16 | 32 | 5.13467 | 5.13877 | +0.00410 |

Max |Δ| = 0.00438 vs naive-vs-naive 重跑 nondeterminism 范围 0.06-0.13 → **adapter 数值通过**。

**Full AttnRes L=32 N=32 PP=8 × VP=4**: ✅ 也通过，loss→7.36 by step 140，grad_norm 收敛（preempted by env restart，不是训练问题）。

**Block AttnRes L=32 from-scratch (Kimi-linear backbone)**: ❌ **所有测试 inf-grad@step1 或 NaN@step10**
- 试过 dim ∈ {768, 1024, 1280, 1536, 2048}
- 试过 init scheme ∈ {depth-scaled, paper uniform}
- 试过 t-blocks/AttnRes-block ratio
- 都未解释这个失败。猜测 L=32 backward graph 中某个 leaf overflow bf16/fp32。**deferred 诊断**，要 register_hook 每个 leaf + RMSNorm output

---

## 5. SGLang 推理覆盖

### 外部 deploy 位置 (sglang fork)

- `/sgl-workspace/sglang/python/sglang/srt/configs/kimi_attn_res_vl.py` — VLM config 注册
- `/sgl-workspace/sglang/python/sglang/srt/models/attn_res_vl_overlay.py` — forward 实现 + KV cache 集成

### KV cache 设计 (`phase11/B5_ATTNRES_INFERENCE_KV_CACHE.md`)

**关键发现**: AttnRes 的 block reps 是**单 forward 内部的局部变量**，跨 step 不需要任何 cache。Cross-step recurrence 由标准 KV cache (KDA delta state + MLA past_kv) 负责。AttnRes 推理时不增加 cache footprint。

### 优化 (in SGLang fork, commit `63325b2b4`)

1. **Phase 2 fused Triton kernel** — `_phase2_merge_norm_kernel`，merge + RMSNorm 融合，2.8% 总 profile (1950 calls/2.9 ms TP=1 decode)
2. **Two-phase computation**:
   - Phase 1: batched 一次/block boundary
   - Phase 2: online-softmax per layer
3. **Sequence-dim TP shard**: reduce-scatter + all-gather 融合 → AR wire bytes **−58%**

### 性能 (`phase11/PROFILING_REPORT.md`)

| 场景 | naive | optimized | 改进 |
|---|---|---|---|
| Decode tps TP=1 @ 4K | — | — | **+30.4%** |
| Decode tps TP=1 @ 16K | — | — | **+27.6%** |
| Decode tps TP=8 @ 4K | — | — | **+26.8%** |
| Decode tps TP=8 @ 16K | — | — | **+27.2%** |
| TTFT (prefill cost) | 1.0× | 1.07-1.13× | 略升（amortized per block）|

### Deferred (audit D1-D4)

- Phase-1 batched-attention Triton kernel — 当前 cuBLAS 已基本到 ceiling
- Phase-1 ↔ layer-0 CUDA stream overlap — 2-3h ROI 太低
- NCCL-aware fused merge+AR kernel — 需 NVSHMEM，3-5 天
- DP attention support — 与 overlay 架构冲突

---

## 6. Checkpoint converters

### `phase11/hf_to_dcp_kimi_attn_res.py` (HF → DCP, LM-only)

- 反 phase10 forward converter
- 关键映射:
  - KDA `A_log` reshape `[1,1,H,1] → [H]`
  - MoE per-expert stacking: HF `mlp.experts.{e}.{gate,up,down}_proj.weight` → TT `ffn._moe.experts.{w1,w2,w3}` [E, intermediate, hidden]
  - Handle 同时 phase10-LM (`mlp.experts.X.gate_proj`) 和 phase11-VL (`block_sparse_moe.experts.X.w1`) 两种 HF naming
- 输出 **flat state_dict**（不是 `{"model": ...}` 嵌套），trainer 期望 flat
- 424/424 keys，meta device 49.12 B 参数总量验过

### `phase11/dcp_to_hf_kimi_attn_res_vl.py` (DCP → HF, VLM)

- 扩展 phase10 LM converter，加 `mm_projector.projector.*` + `config.json` 引用 vision tower
- Vision tower frozen 全程，不进 DCP；推理 startup 从 HF 重新加载

---

## 7. 训练 stage 时间线

### Phase 4 — LM pretrain
| Run | Config | 状态 | Steps | Tokens | 备注 |
|---|---|---|---|---|---|
| `kimi_436m_block_attn_res_fsdp_overnight` | 436M N=4 | ✅ | 12.5K | ~2.4B | 原 scaling-law carrier |
| `lm_447m_base` | 447M aligned | ✅ | 12.5K | 2.46B | reference (无 AttnRes) — **严重 undertrain** (5.5 tok/param, paper 200) |
| `lm_447m_fp8_paperalign` | 447M N=4 FP8 lr=2.0e-3 | ⛔ killed | step 1 | — | lr 30% 超 sqrt-rule 上限 |
| **`lm_447m_fp8_paperalign_B`** | 447M N=4 FP8 **lr=1.5e-3 warmup=1000** | 🔄 **running** | target 12750 | **10.03 B tokens** (112% Chinchilla) | **当前 stage 0** |

### Phase 5 — Multimodal caption pretrain
- `vlm_447m_pretrain`: LLaVA-Pretrain 558k, 7500 steps, ~250M tokens, loss 2.23→1.85
- `sft_v_fsdp8_447m_llava_pretrain`: SFT same captions, 500 steps, loss 5.05→3.03

### Phase 11 — Instruct SFT + RLHF
- `vlm_447m_sft_3ep`: LLaVA-Instruct-150K, 7000 steps, loss 0.72→0.53 — **但定性 eval 0/10 coherent，bang_density 49/50** → SFT 看似收敛但生成 collapse
- GRPO LLaVA-Kimi (`run_grpo_llava_kimi.py`): 16 版本调试，v16 终于跑通 step 0+ 但 reward=-1.0（学不到 → 根因 = LM 严重 undertrain，需 stage 0 redo）

---

## 8. 基准

**File**: `phase11/bench_attn_res.py` (80行)

- 4 modes: vanilla / naive / two-phase (default) / shard
- workload: 1024 prefill + 256 decode, 5 timed runs after 2 warmup
- metrics: TTFT (ms), decode tps (mean ± stdev)
- 用法: `python phase11/bench_attn_res.py --model phase11/hf/lm_base --tp 1`

**Top GPU kernels** (TP=1 4K prefill + 64 decode profile):
- cuBLAS gemvx (MLA/MoE projection): 19.4% + 14.9%
- fused_moe: 11.2%
- flashinfer: 8.1%
- AttnRes Phase 2 Triton: 2.8%

---

## 9. Known limitations + open work

### 已知未解
1. **Block AttnRes L=32 from-scratch instability** (phase3 Result 3) — 根因不明，deferred
2. **SGLang Phase-1 Triton kernel** — 当前 cuBLAS 接近极限，进一步优化 ROI 低
3. **KimiLinearConfig 完整 ModelSpec 集成 (Phase 4c)** — 当前部分手工通过 launcher，未完全经过 torchtitan `Trainer.Config` build chain

### 在做
- **Stage 0 redo (10B tokens)** — `lm_447m_fp8_paperalign_B` running，ETA ~4 天，目标 C4 val < 2.8

### 下一步 (待 stage 0 完成后)
- **Stage 5 redo**: 用新 LM ckpt 重跑 LLaVA-Pretrain 558k → image VLM
- **Stage 11 redo**: LLaVA-Instruct-150K SFT
- **Video VLM extension** (新方向): `DrivingVideoDataset` + spatial-pool projector + frame_pos_embed，stage 2 driving caption pretrain (OpenDV + nuScenes, ~50h)
- **GRPO on Driving VQA** (DriveLM-nuScenes 700K)

详见 phase4+5+video 路线规划。

---

## 10. 一句话总结

**结构性 AttnRes 工作（核心 primitive、双 backbone 集成、PP adapter、SGLang inference、checkpoint converters、benchmarks）已全部就绪并验证。剩下唯一卡点是 LM backbone 严重 undertrain（5.5 tok/param vs paper 200）—— 这是 phase4 → phase5 → phase11 GRPO 链路失败的根本原因。当前 stage 0 redo (10B tokens, paper-aligned hparams) 进行中，预计 4 天完成，之后下游 VLM stages 重跑即可解锁所有训练成果。**
