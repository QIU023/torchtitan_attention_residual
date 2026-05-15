# AttnRes 推理优化 — Infra 计划 + Profiling 深度分析

2026-05-13。基于 phase11 全部 SGLang inference docs 整合：`PHASE11_SGLANG_REPORT.md`、`SGLANG_ATTNRES_AUDIT.md`、`PROFILING_REPORT.md`、`SGLANG_PR_PROPOSALS.md`、`B5_ATTNRES_INFERENCE_KV_CACHE.md`、`SFT_QUANTITY_AND_INFERENCE_BUG.md`、4 套 `bench_results*/`。

## 0. 一句话状态

> **5/5 性能优化全部实现并验证**（+27% decode tps），**0 算法 bug**，但有**一个 production blocker** — flashinfer_mla 在 RTX 5090 (SM 12.0) 上对高 magnitude bf16 输入产 NaN，阻断 GRPO/PPO inference rollout。这个 bug 的修复方案 (B: 4-8h overlay fp32 fallback) 是 stage 0 完成后**第一个要做的 inference 任务**。

---

## 1. 已实现的优化清单（与对应增益）

来源：`SGLANG_ATTNRES_AUDIT.md` 表 C + `PROFILING_REPORT.md`。

| ID | 优化 | 实现位置 | 状态 | 增益 |
|---|---|---|---|---|
| **C1** | Two-phase computation (Phase 1 IO amortise) | `/sgl-workspace/sglang/.../layers/attn_res.py:123, 242` | ✅ | TTFT 0.92-0.95× naive |
| **C2** | Sequence-dim TP shard | overlay 内 `shard mode` | ✅ | AR wire bytes **−58%** (60.0 → 25.4 GB on 16K TP=8) |
| **C3** | Chunked-prefill 兼容 | 自动 | ✅ | 8K prompt × 2K chunk 验证 |
| **C4** | **Phase-2 fused Triton kernel** `_phase2_merge_norm_kernel` (RMSNorm + merge 融合) | `attn_res.py:509` commit `63325b2b4` | ✅ | **+27-30% decode tps** ⭐ |
| **C5** | RS-merge-RMSNorm-AG fusion (algorithm-level) | NCCL trace 观测到 | ✅ | 蕴含在 C2 + C4 |

算法层：A1-A6 全部通过（`fp32 ≤ 1e-4`、`bf16 ≤ 5e-2` 等价性 vs naive oracle）。
分布式层：B1-B4（TP=1, TP=8, TP=2×PP=2×EP=2 + 真实 trained ckpt）全部 boot + gen。
两 carrier 通用性：Kimi + Qwen3 都有 overlay。

---

## 2. 推理性能 — 关键 profiling 数字

来源：`PROFILING_REPORT.md`，torch.profiler kineto trace TP=1 prefill=4096 + decode=64。

### 2.1 Decode throughput summary（kernel ON / OFF 对照）

| TP | ctx | mode | tps OFF | **tps ON** | Δ |
|---|---|---|---:|---:|---:|
| 1 | 4K | two-phase | 535 | **698** | **+30.4%** |
| 1 | 16K | two-phase | 497 | **634** | **+27.6%** |
| 8 | 4K | two-phase | 441 | **559** | **+26.8%** |
| 8 | 16K | two-phase | 379 | **482** | **+27.2%** |
| 8 | 16K | shard | 389 | **481** | **+23.6%** |
| 1 | 24K | two-phase | — | 623 (-12.9% vs vanilla) | **稳定 overhead** |
| 8 | 24K | two-phase | — | 466 (-11.5% vs vanilla) | **稳定 overhead** |

### 2.2 TTFT (prefill) — 注意 long-ctx 反转

| TP | ctx | mode | TTFT (ms) | vs naive |
|---|---|---|---:|---:|
| 1 | 4K | two-phase | 14.5 | 0.92× |
| 1 | 16K | two-phase | 15.9 | 0.95× |
| 8 | 16K | shard | 18.8 | — |
| 1 | **24K** | two-phase | 17.9 | **1.04×** ⚠️ |
| 8 | **24K** | two-phase | 21.9 | **1.07×** ⚠️ |

→ Two-phase 在 ≤16K **打败** naive，到 24K **被反超**。Phase 1 在长 prefill 时被 amortise 但 fixed overhead 变 visible。**不是 bug 是 design**。

### 2.3 Kernel time breakdown — 谁吃了 GPU 时间

TP=1 两段式 mode total kernel time = 105.0 ms。Top-12 占 86%：

| % | time (ms) | calls | kernel | 性质 |
|---:|---:|---:|---|---|
| **19.4%** | 20.1 | 1857 | cuBLAS gemvx (MLA 投影) | 已是 cuBLAS 上限 |
| **14.9%** | 15.5 | 4612 | cuBLAS gemvx (MoE 投影) | 已是 cuBLAS 上限 |
| 11.2% | 11.6 | 1950 | fused_moe_kernel | sglang 已优化 |
| 9.1% | 9.4 | 975 | moe_fused_gate | sglang 已优化 |
| 8.1% | 8.5 | 256 | **flashinfer MLA paged attention** | ⚠️ NaN-prone |
| 3.5% | 3.7 | 975 | moe_align_block_size | |
| 3.5% | 3.6 | 2015 | flashinfer silu | |
| **2.8%** | **2.9** | **2015** | **`_phase2_merge_norm_kernel`** ← 我们 | **本身已小，无优化空间** |
| 2.6% | 2.7 | 2176 | flashinfer RMSNormKernel | |
| 2.6% | 2.7 | 3380 | vectorized_elementwise (add) | |
| 2.4% | 2.4 | 1495 | direct_copy | |
| 2.0% | 2.1 | 768 | KDA fused_sigmoid_gating_delta_rule | |

**关键洞察**：
1. **AttnRes 的 Phase-2 fused kernel 只占 2.8%** — 我们已经把它榨到几乎极限了。继续优化 AttnRes 自身的 kernel ROI = 0。
2. **真正吃时间的是 cuBLAS gemvx (MLA + MoE 投影) = 34.3%**。这是 sglang 已经在用的最快路径，没有 AttnRes 特定优化空间。
3. **MoE 相关 kernels 累计 ~24%** (fused_moe + gate + align)。可以通过 EP/decode-time MoE 减少专家激活进一步省，但这是 sglang 通用问题，非 AttnRes 任务。

**结论**：从 AttnRes 角度看推理几乎到达性能上限。**继续推 decode tps 的杠杆已经从 AttnRes 转移到 MLA/MoE/decode-time loop overhead**。

### 2.4 NCCL fabric

| | shard=0 (AR replicated) | shard=1 (RS+AG) | Δ |
|---|---:|---:|---:|
| AllReduce bytes | 60.0 GB | 25.4 GB | **−58%** |
| Reduce-scatter bytes | 0 | 537 MB | 新增（小）|

3D mesh：AR −60%。

### 2.5 Kernel call count 健康

2015 actual / 2048 expected (64 decode × 16 layers × 2 queries) = **98% 命中率**。33 个 miss 是 first-block-empty short-circuit case，符合 design。

---

## 3. ⚠️ 唯一 production blocker：flashinfer_mla bf16 NaN

来源：`SGLANG_PR_PROPOSALS.md`、`SFT_QUANTITY_AND_INFERENCE_BUG.md`、`VISION_INJECTION_BUG_RCA.md`。

### 现象

- **同一 SFT ckpt 在 torchtitan eager 跑**：clean logits, max=10.69 ✓
- **同一 ckpt 在 SGLang flashinfer_mla 跑**：layer 16 attention 出 NaN ❌
- AttnRes residual stream 在两条路径 **magnitude 增长完全一致**（chunk 12 达 max=77）→ **不是算法问题**

### 根因

RTX 5090 SM 12.0 (Blackwell) 上 MLA 路径**只有 flashinfer_mla 一个可用 backend**：

| Backend | 状态 |
|---|---|
| `flashinfer_mla` | ❌ NaN @ high magnitude bf16 |
| `torch_native` | shape mismatch (不支持 MLA q_a/kv_compressed layout) |
| `triton` | OOM (need 131072 shared mem, have 101376) |
| `fa3` | requires SM 80-90，5090 是 SM 12.0 |
| `flashinfer`(bare) | 同 path 同 NaN |

bf16 在 attention QK 内积里精度不够当 input |x|>~32 时，flashinfer fast path 会 overflow。production Kimi 用 fp32 scoring 路径，blog 没明说。

### 影响

GRPO/PPO 整条 rollout 链路 = NaN → reward=-1.0 → 学不到东西。这是 phase11 GRPO 16 版本调试**最终 surface 的 blocker**，不是 reward function、不是 torchstore、不是 monarch，是底层 attention 数值。

### 三个修方案（按 ROI 排序）

#### 方案 B（推荐先做）— Overlay 内 fp32 MLA fallback (4-8h)

**自包含**，不依赖 SGLang upstream 改动：

```python
# 在 KimiAttnResDecoderLayer._run_attn 加 magnitude guard:
def _run_attn(self, attn_input, positions, forward_batch):
    if (
        self.layer_idx in self.config.full_attn_layers  # MLA 层
        and attn_input.abs().max() > FALLBACK_THRESHOLD  # e.g. 32
    ):
        return self._run_attn_fp32_fallback(attn_input, positions, forward_batch)
    return self.self_attn(...)  # 默认 flashinfer_mla 路径

def _run_attn_fp32_fallback(self, attn_input, positions, forward_batch):
    # fp32 RMSNorm + 投影 + 手动 SDPA, V matmul 还可以 bf16, 末态 cast 回 bf16
    # 完全 mirror DeepseekMLAForwardMixin.forward_absorb_core 的 fp32 path
    ...
```

- **代价**: 受影响 layer 延迟 ~3-5×。decode tps 整体 −10%~−20%（只 MLA 层走 fallback，KDA 层照旧）
- **可接受**：research / VLM PPO 用例 OK；production serving 不 OK
- **正确性验证**：跟 torchtitan eager 输出 **token-by-token compare**

#### 方案 A — SGLang upstream issue (1h)

文档化 + 最小 repro，让 SGLang 团队决定要不要加 `--mla-fp32-scoring` flag。**不阻塞 B，并行做**。

#### 方案 C — 算法 fix: `block_attn_res` 后加 RMSNorm (paper-track)

在 `block_attn_res` 出口加一个 RMSNorm，让 residual stream **layer-to-layer 有界**：

```python
def block_attn_res(blocks, partial, proj, norm, final_norm):
    K = stack_blocks([norm(b) for b in blocks] + [norm(partial)])
    logits = einsum("d,nbtd->nbt", proj.weight.squeeze(0), K)
    weights = softmax(logits, dim=0)
    h = einsum("nbt,nbtd->btd", weights, V)
    return final_norm(h)  # ← 新增
```

- **代价**: 算法变更，**需重训** (~2-4h SFT)
- **价值**: paper-worthy ablation。如果证明同等 loss 但 inference 更稳，是论文贡献
- **依赖**: stage 0 完成后再做

### Recommended order

1. **A 立刻做**(1h)：file SGLang issue，把 minimal repro 发出去给上游
2. **B 在 stage 0 期间并行做** (4-8h)：unblock GRPO，不等上游
3. **C 在 stage 0 + B 都完成后**：作为 paper track follow-up

---

## 4. Deferred 优化（不推荐花时间，但要解释为什么）

来源：`SGLANG_ATTNRES_AUDIT.md` 表 D + F6。

| ID | 项目 | 决定 | 不做的理由 |
|---|---|---|---|
| **D1** | Phase-1 batched-attention Triton kernel | Skip | Phase 1 在 d=1024 上跑 cuBLAS einsum 已是上限，重写 Triton 不会赢 |
| **D2** | Phase 1 ↔ layer-0 CUDA stream overlap | Defer | Phase 1 ~1ms vs decode ~25ms，省的 marginal。2-3h 工，ROI 太低 |
| **D3** | NCCL-aware fused merge+AR kernel | Defer | 需 NVSHMEM / NCCL2-aware Triton，3-5 天真工程；blog "和 AR 融合" 暗示这是它们的下一步 |
| **D4** | DP attention support | **Block** | 与 overlay 架构冲突 — overlay 绕过 `LayerCommunicator.prepare_attn`（DP scatter 住所），半实现风险大 |
| **F6** | Multi-stream NCCL for shard mode | Defer | TPParallel-async 已部分 overlap，显式 overlap 需要深度集成 |
| **F1-F5** | Python op/list overhead 等微观项 | Close | 已被 cuda-graph + torch.compile 折叠，bench 验证无 Δ |

---

## 5. 还能继续做的事 — 但不属于 AttnRes infra 优化

按 ROI 高低排：

| # | 项目 | 性质 | 估时 | 预期增益 |
|---|---|---|---|---|
| **0** | **方案 B fp32 MLA fallback** | bugfix | 4-8h | **unblock GRPO** ⭐⭐⭐ |
| 1 | A: SGLang issue file | 文档 | 1h | upstream 推进 |
| 2 | **CI smoke** for AttnRes overlay (single-shot gen on smoke ckpt) | quality | 1-2h | upstream PR-ready |
| 3 | 4-mode bench harness 加 verification (`__file__` + `_phase2_merge_norm_triton is not None`) | quality | 30 min | 防止 install path drift 再坑一次 |
| 4 | Cross-reference 到 sglang `layers/mhc.py` 上游 design doc | 文档 | 1h | upstream PR-ready |
| 5 | C: post-aggregation RMSNorm（needs retrain） | paper-track | 6-8h（含重训）| paper contribution |
| 6 | Long-ctx (32K) 跑通（当前 fail @ `max_position_embeddings=32768` overrun） | feature | 2-3h | research only, 不解锁任何下游 |

**前 4 项加起来 < 10 h** = 一天的工作即可关闭整个 inference 工作流。

---

## 6. 离 upstream PR 还差什么

来源：`SGLANG_ATTNRES_AUDIT.md §I`。

代码量（已就绪，不再改动）：
- `layers/attn_res.py` 645 LOC（核心算法 + Triton kernel）
- `models/attn_res_overlay.py` 970 LOC（Kimi carrier）
- `models/qwen3_attn_res_overlay.py` 605 LOC（Qwen3 carrier）
- `test/registered/layers/test_attn_res.py` 7 tests
- `docs/supported_models/text_generation/block_attn_res.md`

**只欠**（doc + test additions）：
1. CI smoke test
2. 4-mode bench 作为 profiling tool 集成
3. 引用 upstream `layers/mhc.py` design doc

---

## 7. 立刻行动建议

**本周内**（stage 0 跑着的时候并行）：
1. **方案 B：fp32 MLA fallback in overlay** ← 最高优先 ⭐
2. **方案 A：SGLang upstream issue 提交** ← 1h 顺手做
3. **bench 加 verification check** ← 30 min 防再翻车

**Stage 0 完成（~4 天后）**：
4. 用新 LM ckpt 重跑 phase5_vlm_multimodal_sft/phase11 SFT → 测试 fp32 fallback 在新 ckpt 下的 inference 表现
5. 决定方案 C 要不要做（依据：B 修复后 inference quality 是否仍有问题）

**永远不做**：
- D1-D4（已论证无 ROI）
- 试图把 Phase-2 Triton kernel 再加速（已 2.8% 总时间，地板了）

---

## 8. 一句话最终结论

**AttnRes 的 inference infra 层面工作已经 90% 完成**（5/5 性能优化 + 27/27 全套 audit）。**唯一阻塞 production 的是 flashinfer_mla bf16 NaN**，这是 SGLang 上游问题不是我们 AttnRes 问题，但**方案 B (4-8h overlay fp32 fallback)** 可以自包含解决，stage 0 期间并行做即可在 stage 0 完成时 unblock 整个 GRPO 链路。
