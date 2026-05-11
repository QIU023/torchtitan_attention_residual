# PP Adapter Pressure Test — Plan v2 (2026-05-11)

## TL;DR

Stress test `CrossStageCacheAdapter` at **prod-grade aggressive configs**:
PP=8 × VP=4 single-node + (stretch) PP=16 × VP=2 multi-node. Builds a
**32-layer test carrier** because our 16-layer 447M can't satisfy
PP*VP ≤ num_layers for VP=4. Cannot start until Stage 3 GRPO finishes
(~13:45 today). Doc captures prod-config research, refined test matrix,
batch-size constraints, and the launcher to write.

## Prod-grade PP configs (researched 2026-05-11)

| System | Total params | PP | VP | TP | DP | CP | Global batch | µbs | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Llama 3.1 405B** (16K GPUs) | 405B | **16** | unspecified (likely 4-8) | 8 | varies | 2 | **16M tokens** | varies | PP=16 chosen "to fit model in memory" |
| Llama 3.1 405B (512 H100 deploy) | 405B | **9** | **7** | 8 | 4 | 2 | 2304 / 253 | **1** | Note non-power-of-2 PP/VP |
| Llama 3.1 405B (448 B200) | 405B | 8 | unspecified | 4 | 7 | 2 | -- | -- | -- |
| **DeepSeek V3 671B** (2048 H800) | 671B (37B active) | 16 (est) | **DualPipe** (bidir) | -- | -- | -- | -- | -- | Custom bidirectional schedule, fewer bubbles |
| Megatron 1T benchmark | 1T | 64 | 12 | 8 | -- | -- | -- | -- | Academic upper bound |
| GPT-3 175B (original) | 175B | 64 | 1 | -- | -- | -- | -- | -- | Pure 1F1B, big bubbles |
| Bloom 176B | 176B | 12 | 1 (no VP) | -- | -- | -- | -- | -- | -- |

**Production sweet spot**: PP ∈ {8, 16, 32}, VP ∈ {2, 4, 8}. PP=64 is
academic; PP=2 disables VP.

**Llama 3.1's interesting choice**: PP=9 VP=7 on the 512-H100 deploy.
Non-power-of-2 sizes mean their VP scheduler must handle uneven layer
distribution — exactly the "flexible interleaved 1F1B" torchtitan now
supports (relaxes `num_microbatch % pp_size == 0`).

## Constraints we must respect

| Rule | Source | Implication for our 447M (16 layers) |
|---|---|---|
| `num_microbatches % PP == 0` (strict 1F1B) | Megatron | gbs/(DP*µbs) must be PP-multiple |
| OR `num_microbatches % num_rounds == 0` (flexible) | torchtitan flex 1F1B | More flexible — accepts e.g. µbs=PP*VP=32 |
| `num_layers % (PP*VP) == 0` | Megatron | **PP=8 × VP=4 = 32 > 16 layers ❌** |
| `PP > 2` for VP | Megatron | OK, we use PP=8 |
| Bubble ratio = (PP-1)/(M*VP) | analytic | high VP shrinks bubble |

**The shape problem**: 447M has 16 layers. For PP=8 VP=4 we'd need 32+
layers. Two paths:

* **Path A — build a deeper test carrier** (recommended for stress test).
  Use `llama3_300m_attn_res_L32_n8` (32 layers, 8 attn-res blocks of 4
  layers each). SMOKE only — no real training, just step time + numerics
  vs naive. Synthesizes prod-realistic config.
* **Path B — restrict to VP=2 on 447M**. Less aggressive but real research
  weights.

User asked for "aggressive" → Path A primary, Path B secondary.

## Refined test matrix

### Tier 0 — sanity smoke (1h, low risk)

Run the EXISTING `phase3/launch_8gpu_adapter.sh` with the unchanged 175M
L16 toy carrier as a re-validation. Catches whether anything in
torchtitan trunk regressed since phase3's earlier passing runs.

| run | model | PP | VP | µbs | gbs | layers/stage |
|---|---|---|---|---|---|---|
| `pp8_vp1_175m_naive` (sanity) | 175m L16 | 8 | 1 | 4 | 32 | 2 |
| `pp8_vp1_175m_adapter` (sanity) | 175m L16 | 8 | 1 | 4 | 32 | 2 |

### Tier 1 — aggressive PP=8 × VP=4 on 32-layer carrier

| run | model | PP | VP | µbs | gbs | layers/chunk | num_microbatches |
|---|---|---|---|---|---|---|---|
| `pp8_vp4_L32_naive` | 300m L32 | 8 | 4 | 1 | 32 | 1 | 32 (PP*VP) |
| `pp8_vp4_L32_adapter` | 300m L32 | 8 | 4 | 1 | 32 | 1 | 32 |

**Why µbs=1, gbs=32**: minimum to fill the PP=8 × VP=4 pipeline
(32 microbatches in flight, one per chunk). Smaller gbs would leave
the pipe underfilled; larger gbs just makes test slower.

Real prod would use a larger gbs (Llama 3.1's PP=9 VP=7 uses gbs=2304
with µbs=1, i.e. 2304 microbatches → ~36× the pipe size for high
utilization). For SMOKE only, gbs=32 is enough to measure adapter
overhead.

### Tier 2 — real-weights PP=8 × VP=2 on 447M

| run | model | PP | VP | µbs | gbs | layers/chunk |
|---|---|---|---|---|---|---|
| `pp8_vp2_447m_naive` | kimi 447M | 8 | 2 | 4 | 64 | 1 |
| `pp8_vp2_447m_adapter` | kimi 447M | 8 | 2 | 4 | 64 | 1 |

This uses our real research weights and matches the inference-side
VLM SFT 3ep ckpt. Numeric loss must match phase 4 LM-only training
within bf16 tolerance.

### Tier 3 (stretch) — multi-node PP=16 × VP=2

Skip unless we get a second 8-GPU node from vast.ai. Mentioned in the
test plan only for completeness — the on-wire signature on PP cross-
node ethernet (vs intra-node NVLink) is where the adapter's bandwidth-
constant property is most visible.

## Exit criteria

| metric | target |
|---|---|
| Adapter step time | within **5%** of naive baseline at same config |
| Loss curve | match naive within **1e-3 relative** for first 100 steps |
| Stage→stage send-bytes (steady state, from NCCL trace) | adapter shows **constant in stage id** (O(ΔK_i = 1) per µbs); naive shows linear-in-stage (O(K_i)) |
| Memory | no OOM at any config |

NCCL traces save to `phase3/runs/{run_name}/tier_b_trace/`.

## When can we start

Pipeline state (06:30):

| Stage | Status | ETA |
|---|---|---|
| Stage 1 pretrain | ✅ DONE step-7500 | -- |
| Stage 2 SFT 3ep | step ~5500/7000, in flight | ~08:15 |
| Stage 3a HF convert | waiting | ~08:25 |
| Stage 3a-eval gate | waiting | ~08:35 (NEW — see below) |
| Stage 3b GRPO 1500 step | waiting | ~14:00 |

**Earliest PP smoke start: 14:00 today.**

## Wired into pipeline: post-SFT eval gate (Stage 3a-eval)

To address user's "no `!!!!` outputs" requirement, added a qualitative
eval between Stage 3a (DCP→HF) and Stage 3b (GRPO):

`phase11/eval_sft_3ep_qualitative.py` — runs SGLang on 10 LLaVA images
with both greedy and T=0.7 sampling. Counts samples where output starts
with a letter AND has < 30% '!' density. **Threshold 6/10 to proceed**;
otherwise the pipeline stops with a clear message recommending more SFT.

Wired in `phase11/run_stage3.sh`. Result is logged to
`phase11/hf/vlm_sft_3ep/qualitative_eval.log`.

## Launcher to write before PP smoke start

`phase3/run_pressure_test.sh` wrapping `launch_8gpu_adapter.sh`:
- Resolves carrier flavor (175m / 300m / 447m)
- Runs naive + adapter back-to-back at each (PP, VP) config
- Calls phase7/extract_collectives.py per run
- Outputs Markdown summary table

ETA to write: 30 min once Stage 3 finishes.

## Sources

- [Scaling Llama 3 Training (ISCA'25)](https://aisystemcodesign.github.io/papers/Llama3-ISCA25.pdf)
- [DeepSeek-V3 Technical Report (arXiv)](https://arxiv.org/pdf/2412.19437)
- [Megatron-LM pipeline_parallel docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/pipeline_parallel.html)
- [TorchTitan (ICLR'25)](https://proceedings.iclr.cc/paper_files/paper/2025/file/e6231c5f46598cfd09ff1970524e0436-Paper-Conference.pdf)
- [DualPipe (DeepSeek)](https://github.com/deepseek-ai/DualPipe)
