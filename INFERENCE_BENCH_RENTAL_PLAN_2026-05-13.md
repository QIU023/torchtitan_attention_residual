# Inference Bench Rental Plan + PP Pressure Test 状态

2026-05-13

本文档两部分：
1. **AttnRes 推理 bench 外租 GPU 计划** — 多少张卡、多久、跑什么
2. **PP 压测当前结果汇总 + PP=4×VP=8 是否值得跑**

---

# 第一部分：推理 bench 外租 GPU 计划

## 1.1 为什么要外租 + 在我们本地不够什么

当前 bench 数据全部 on **8× RTX 5090 PCIe (SM 12.0 / Blackwell consumer)**，已覆盖：
- TP=1 / TP=8 × ctx ∈ {4K, 16K, 24K} × 4 modes (vanilla/naive/two-phase/shard)
- Phase-2 Triton kernel 命中确认 (98% kernel call coverage)
- NCCL fabric AR bytes −58% with seq-dim TP shard

**5090 setup 上有但说不清楚的事**:
- `flashinfer_mla` 在 high-magnitude bf16 输入下 NaN — **是 Blackwell-only？还是 algorithmic？**
- 我们加了 `fp32 MLA fallback` 当 workaround，但**不知道在 H100 上还需不需要**
- AttnRes Phase-2 fused Triton kernel 的 +27% decode tps — **是 5090 PCIe 特有？还是 cross-arch？**
- 32K ctx 在 5090 跑不通 (`max_position_embeddings=32768` overruns) — **是 model config 限制还是 mem 限制？**

外租 H100 是为了**回答这些 cross-architecture 问题**，并拿到 enterprise-tier 推理基准用来 paper 对比。

## 1.2 推荐配置 — 8× H100 SXM @ 6 小时

| 选项 | 配置 | 时长 | 估价 (vast.ai) | 覆盖度 |
|---|---|---|---|---|
| **A (推荐)** | **8× H100 SXM 80GB** | **6 h** | **$100-120** | TP=1+TP=8 全 bench + ablation + trace |
| B (budget) | 1× H100 80GB PCIe | 4 h | ~$8 | 只 TP=1，省 80%，但不能验 TP shard mode |
| C (research scope) | 1× B200 + 1× H200 | 3 h | ~$25 | datacenter Blackwell + Hopper next-gen cross-validate |

**为什么不要 A100 / 4090 / 5090 rental**:
- A100 (SM 8.0)：fa3 在 SM ≥ 80 上有但 SM=80 是 Ampere 边界，过时 baseline
- 4090 / 5090 rental：跟我们本地一样，不增 information
- H100 是 paper 通用 baseline，最 valuable comparison point

## 1.3 时间表 — 6 小时内做完的事

**Phase 0 — Setup (2h, 第一台机起来)**
- 0.0  `git clone -b attention_residual_inference git@github.com:QIU023/sglang.git` (15 min)
- 0.1  `cd sglang && pip install -e python/` + flashinfer + fla-core + torchao (30 min)
- 0.2  scp Kimi 1.4B AttnRes ckpt (~3 GB safetensors) + Qwen3 dummy ckpt (~250 MB) 到 H100 box (20-30 min @ residential bandwidth)
- 0.3  健康 boot smoke test single-shot generation TP=1 (10 min)
- 0.4  健康 boot TP=8 (10 min)
- 0.5  Verify `_phase2_merge_norm_kernel` 确实 fire（避免又翻车）— 加个 `print(sglang.srt.layers.attn_res.__file__)` (5 min)

**Phase 1 — Single-arch bench (1.5h)**
- 1.0  4-mode bench TP=1 × ctx {4K, 8K, 16K, 24K, 32K} = 20 runs × 30s = 10 min
- 1.1  4-mode bench TP=8 × ctx {4K, 8K, 16K, 24K, 32K} = 20 runs × 30s = 10 min
- 1.2  TP=4 bench at ctx 16K + 32K (paper 常用) = 5 min
- 1.3  Sanity: bench on Qwen3 dummy ckpt TP=1 = 10 min
- 1.4  JSON output + raw → `bench_results_h100/`

**Phase 2 — fp32 MLA fallback ablation (1h)** ⭐ 重要
- 2.0  baseline: H100 without fallback (`ATTNRES_MLA_FP32_FALLBACK=0`)，long-ctx prefill stress
- 2.1  baseline + clamp: `ATTNRES_INPUT_CLAMP=32`
- 2.2  full fallback: `ATTNRES_MLA_FP32_FALLBACK=1 ATTNRES_FP32_NORM=1`
- 2.3  **关键 deliverable**：H100 是否 NaN 自然消失？如果消失 → fallback **只是 Blackwell-specific**，paper 可声明 production hardware 不需要。如果仍 NaN → fallback **是 algorithmic** 永久必要。

**Phase 3 — Profiling traces (1h)**
- 3.0  torch.profiler kineto on TP=1 prefill=4K decode=64 (与 5090 PROFILING_REPORT 对照)
- 3.1  torch.profiler kineto on TP=8 prefill=16K decode=64
- 3.2  NCCL trace shard=0 vs shard=1 on TP=8 prefill=16K (验证 AR bytes -58% 在 H100 NVLink 上仍成立)
- 3.3  Save .trace.json.gz 文件 → 下载回本地

**Phase 4 — VLM bench (可选, 0.5-1h)**
- 4.0  Image-batched VLM bench: 4 images × {2K, 4K, 8K} prompt + 256 decode
- 4.1  对比 SigLIP + projector 在 H100 vs 5090 的 TTFT

**Phase 5 — Wrap (0.5h)**
- 5.0  Generate report data → JSON + markdown
- 5.1  download bench_results_h100/ + traces to local
- 5.2  shutdown rental box

## 1.4 Deliverables

| Artifact | 内容 |
|---|---|
| `phase11_rlhf_grpo_infra/bench_results_h100/` | 4-mode tps × ctx 5 × TP 3 × Kimi + Qwen3 |
| `phase11_rlhf_grpo_infra/PROFILING_REPORT_H100.md` | 跟 5090 对照表 (kernel breakdown, AR bytes, tps Δ) |
| `phase11_rlhf_grpo_infra/PR_FP32_FALLBACK_NECESSITY.md` | **关键决定**：fp32 fallback 是 Blackwell-only 还是 universal |
| Cross-arch chart for paper | Decode tps vs context (H100 / 5090) |
| Triton kernel coverage table | 验证 Phase-2 kernel 在 H100 上 also 98% hit rate |

## 1.5 风险 + Fallback

| 风险 | 概率 | Fallback |
|---|---|---|
| H100 vast.ai 8× 缺货 | 中 | 拆成 2× 4× H100 PCIe，但 TP=8 mesh 测试受限 |
| Network 慢 / ckpt 传不完 | 低 | 提前 upload ckpt 到 S3，rental box `aws s3 cp` |
| flashinfer / fla-core 在 H100 编译失败 | 低 | 用 vast.ai preconfig image (pytorch + cuda 13.0) |
| sglang import 失败 | 极低 | fork 已 sync 干净 (b3f6b543f) |
| 跑超时（>6h 还没做完） | 中 | Phase 4 VLM 可砍，Phase 2 ablation 可只跑 fallback ON/OFF 两档 |

## 1.6 总成本估算

| 项 | 金额 |
|---|---|
| 8× H100 SXM @ $15-20/h × 6h | $100-120 |
| vast.ai disk @ $0.1/GB × ~20 GB | $2 |
| network egress (download traces) | <$1 |
| **Total** | **~$120** |

如果用 1× H100 PCIe（Option B）只跑 TP=1 部分：**$8 一晚搞定**，能拿到 H100 vs 5090 的 single-rank 头条数字。

## 1.7 时间预算（含 setup overhead）

- 准备 work pre-rental（dump qwen3 ckpt 到 S3，写 launch script）：**1 天**（在本地，不占 rental）
- Rental phase 0-5：**6 小时** wall-clock 一次性跑完
- 下载 + analyze + 整理 paper figure：**1 天**

**总（夯实推理 cross-arch 故事）= ~3 天 dev + 6 小时 rental** ≈ **$120 + 3 工作日**。

---

# 第二部分：PP 压测当前结果 + PP=4×VP=8 评估

## 2.1 已 ran 的 PP 配置完整 grid

来源：`phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md` + `phase3_attnres_pp_integration/runs/`

### Llama3 175M backbone (历史 carrier)

| Carrier | PP | VP | chunks | mode | final loss | Δ adapter vs naive | 结论 |
|---|---|---|---|---|---|---|---|
| L=16 Block AttnRes N=8, dim=768 | 8 | 2 | 16 | naive / adapter | 5.42497 / 5.42935 | **+0.00438** | ✓ pass |
| L=16 Block AttnRes N=8, dim=768 | 4 | 2 | 8 | naive / adapter | 5.52833 / 5.52941 | **+0.00108** | ✓ pass |
| L=16 Block AttnRes N=8, dim=768 | 4 | 4 | 16 | naive / adapter | 5.13467 / 5.13877 | **+0.00410** | ✓ pass |
| L=32 Full AttnRes N=32 | 8 | 4 | 32 | adapter | 7.36 @ step 140 | (no naive run, preempted) | ✓ trainable |
| L=32 Block AttnRes N=8/16, multiple dims | — | — | — | — | inf / NaN | — | ❌ **untrainable from-scratch** (deferred) |

### Kimi Linear paper-architecture (KDA + MLA + MoE + Block AttnRes)

| Carrier | PP | VP | chunks | mode | final loss @ 300 | Δ | 结论 |
|---|---|---|---|---|---|---|---|
| 48B-layout d=1280 e=32 L=24 N=8 | 8 | 3 | 24 | adapter / naive | 6.226 / 6.187 | **+0.039** | ✓ pass |
| 48B-layout d=1280 e=16 L=32 N=8 | 8 | 4 | 32 | adapter / naive | 5.970 / 5.959 | **+0.011** | ✓ pass |
| 48B-layout d=1280 e=32 L=32 N=8 | 8 | 4 | 32 | adapter (smoke only step 1) | — | — | ✓ boots |

## 2.2 chunks-count vs PP-depth coverage map

```
                    chunks count
                  8    16   24   32
              ┌────────────────────────
        PP=4 │  ✓2   ✓4   .    .       (Llama3 L=16, plus PP=8×VP=4 from kimi covers 32)
              │
        PP=8 │  .    ✓2   ✓3   ✓4      (mixed L=16/24/32 carriers)
              │
   (PP×V=32) │   .    .    .   ✓×2     (PP=8×VP=4 covered twice, Llama3 Full + Kimi Block)
```

## 2.3 PP=4 × VP=8 = 32 chunks 值不值得跑

### 算法层面：**重复信息，已有 PP=8×VP=4 = 32 chunks 覆盖**

adapter 的正确性两件事：
1. **Cross-stage cache delta 数值对齐** — 已通过 PP=8×VP=4 验证 (kimi L=32 |Δ|=+0.011)
2. **多个 microbatch 跨 virtual stage 的 cache 隔离** — 已通过 PP=4×VP=4 (4 mb × 4 virtual) 验证

PP=4×VP=8 = PP×V=32 chunks = 跟 PP=8×VP=4 在 schedule chunks 总数上一样。

### 工程层面：**会暴露 ONE 新东西 — VP=8 per-rank memory 峰值**

| Shape | per-rank virtual stages active | per-rank cache | per-rank steady-state mem |
|---|---|---|---|
| PP=8 × VP=4 | 4 | 4 × layer_state | 24.76 GiB (kimi L=32) ✓ |
| PP=4 × VP=4 | 4 | 4 × layer_state | 23.65 GiB (llama3 L=16) ✓ |
| **PP=4 × VP=8** | **8** | **8 × layer_state** | **~38 GiB 预测** ⚠️ |

每个 virtual stage 保持自己的 incoming-blocks cache + scheduler 状态。VP=8 是 VP=4 的 2× per-rank memory。在 5090 32 GB 上**几乎必 OOM**，只能在 H100/A100 80GB 跑。

### 决策建议：**不要跑，除非有 specific motivation**

不跑的理由：
1. **算法 信息含量 = 0** — PP=8×VP=4 已 cover 同 chunks count
2. **5090 OOM 风险** — 38 GiB > 32 GiB
3. **如果想测 max-VP，PP=8×VP=8 (64 chunks) 比 PP=4×VP=8 更 informative** — 更 aggressive，且 PP=8 已知 OK
4. **paper 不要求** — paper §"Training recipe" 用 PP=16 VP=2 (32 chunks)，我们 PP=8 VP=4 同 chunks 已 representative

跑的理由（如果坚持）：
1. **paper-audit-completeness**：四象限 PP-depth × VP-width 表填满
2. **memory-stress test**：验证 max-per-rank-virtual schedule 不 leak
3. **CI smoke**：发现潜在 corner case bug

**我推荐**：**skip PP=4×VP=8**。如果要做 max-stress audit，**跑 PP=8×VP=8 = 64 chunks**（信息量更高，且 5090 mem 可能还 OK 因 8 ranks 各只 8 virtual = 同 VP=8 但分到更深 pipe）。

## 2.4 真正应该补的 PP 测试（如果要 paper-quality coverage）

| 优先级 | shape | 理由 | 估时 |
|---|---|---|---|
| ⭐⭐⭐ | **PP=2 × VP=2 vs PP=2 × VP=4** | 最浅 PP，验证 adapter 在 lower bound 不崩 | 30 min |
| ⭐⭐ | **Block AttnRes L=32 — 改 fp32 RMSNorm 后重测** | 当前 inf-grad 可能 RMSNorm backward 数值导致；这个 unblock 比 PP=4×VP=8 重要 100×  | 2-3 h |
| ⭐⭐ | **PP=8×VP=4 long-run 5000 steps** | 当前 only 300 步，验证 adapter 在 long-tail 不漂移 | overnight |
| ⭐ | PP=4×VP=8 | audit-completeness | 1 h (H100 上, 5090 OOM) |
| ⭐ | PP=8×VP=8 | max-stress | 2 h (paper-quality) |

## 2.5 一句话总结

- 推理 bench rental：**8× H100 SXM × 6h = ~$120**，回答 fp32 fallback 是否 cross-arch + 提供 paper-tier baseline
- PP=4×VP=8：**不跑** — PP=8×VP=4 已覆盖同 chunks count，VP=8 反而 5090 OOM 风险
- 真正缺的：**Block AttnRes L=32 from-scratch 诊断**（fp32 RMSNorm 实验）+ **PP=8×VP=4 long-run 5000 steps**
