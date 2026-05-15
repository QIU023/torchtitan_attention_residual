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

## Refined test matrix (Path A — from-scratch C4 training, no real weights needed)

User correction (2026-05-11 06:35): pressure tests don't need real
weights. Build deeper Path-A carriers (L32, L48) and train each from
random init on C4 for 1000 steps. The adapter-vs-naive comparison is
about **wire bandwidth + numerics match**, not absolute loss quality.

**Carriers added** to `torchtitan/experiments/attn_res/__init__.py`:

| flavor | n_layers | n_blocks | layers/block | dim | n_heads | params |
|---|---|---|---|---|---|---|
| `175M_attn_res_L16_n8` (existing) | 16 | 8 | 2 | 768 | 12 | ~175M |
| `175M_attn_res_L32_n8` (NEW) | 32 | 8 | 4 | 768 | 12 | ~280M |
| `175M_attn_res_L48_n8` (NEW) | 48 | 8 | 6 | 768 | 12 | ~390M |

Verified registered + ModelSpec builds (06:42).

### Sweep matrix

| run | flavor | PP | VP | layers/chunk | µbs | gbs | num_µbatches | notes |
|---|---|---|---|---|---|---|---|---|
| **A1 sanity** | L16_n8 | 8 | 2 | 1 | 1 | 16 | 16 | reproduce existing phase3 baseline |
| **A2 aggressive** | L32_n8 | 8 | 4 | 1 | 1 | 32 | 32 | matches user's "PP=8 VP=4" |
| **A3 alt-shape** | L32_n8 | 4 | 8 | 1 | 1 | 32 | 32 | same 32 chunks but VP doubled, PP halved |
| **A4 prod-depth** | L48_n8 | 8 | 6 | 1 | 1 | 48 | 48 | closer to Llama-3.1 8B's 32 layers × 1.5 |
| **A5 deep-VP** | L48_n8 | 4 | 12 | 1 | 1 | 48 | 48 | extreme VP=12 (Megatron's 1T benchmark uses VP=12) |

Each row runs **naive** (adapter OFF) then **adapter** (adapter ON) →
10 runs total. At 1000 steps each on 8× 5090 with PP=8 ≈ ~1500-2500
tps × 1000 ≈ 8-15 min/run → **total sweep ~2-3h**.

### Why this design

* **From scratch on C4**: matches phase 3's original adapter-vs-naive
  protocol (which validated bit-identical loss curves on the 175M L16
  pair). Same standard, just deeper.
* **Multiple shapes per (chunks total)**: A2 vs A3 both have 32 chunks
  but different (PP, VP) splits — adapter overhead per-chunk vs per-
  rank-collective is measured separately.
* **A5 VP=12**: above-prod aggressive. If adapter overhead stays bounded
  at VP=12 it'll trivially hold at the prod sweet spot VP=4-8.
* **No real-weights tier**: dropped per user correction. Real-weights
  comparisons go in the carrier paper, not the infrastructure paper.

### Tier B (stretch) — multi-node PP=16

Defer until we get a second 8-GPU node. The intra-node-NVLink vs
inter-node-ethernet asymmetry is where the adapter's constant-bandwidth
property is most visible (NVLink can hide naive's growing send-bytes
in the noise; ethernet can't).

## Exit criteria

| metric | target |
|---|---|
| Adapter step time | within **5%** of naive baseline at same config |
| Loss curve | match naive within **1e-3 relative** for first 100 steps |
| Stage→stage send-bytes (steady state, from NCCL trace) | adapter shows **constant in stage id** (O(ΔK_i = 1) per µbs); naive shows linear-in-stage (O(K_i)) |
| Memory | no OOM at any config |

NCCL traces save to `phase3_attnres_pp_integration/runs/{run_name}/tier_b_trace/`.

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

`phase11_rlhf_grpo_infra/eval_sft_3ep_qualitative.py` — runs SGLang on 10 LLaVA images
with both greedy and T=0.7 sampling. Counts samples where output starts
with a letter AND has < 30% '!' density. **Threshold 6/10 to proceed**;
otherwise the pipeline stops with a clear message recommending more SFT.

Wired in `phase11_rlhf_grpo_infra/run_stage3.sh`. Result is logged to
`phase11_rlhf_grpo_infra/hf/vlm_sft_3ep/qualitative_eval.log`.

## Launcher to write before PP smoke start

`phase3_attnres_pp_integration/run_pressure_test.sh` wrapping `launch_8gpu_adapter.sh`:
- Resolves carrier flavor (175m / 300m / 447m)
- Runs naive + adapter back-to-back at each (PP, VP) config
- Calls phase7_nccl_traffic_catalog/extract_collectives.py per run
- Outputs Markdown summary table

ETA to write: 30 min once Stage 3 finishes.

## Sources

- [Scaling Llama 3 Training (ISCA'25)](https://aisystemcodesign.github.io/papers/Llama3-ISCA25.pdf)
- [DeepSeek-V3 Technical Report (arXiv)](https://arxiv.org/pdf/2412.19437)
- [Megatron-LM pipeline_parallel docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/pipeline_parallel.html)
- [TorchTitan (ICLR'25)](https://proceedings.iclr.cc/paper_files/paper/2025/file/e6231c5f46598cfd09ff1970524e0436-Paper-Conference.pdf)
- [DualPipe (DeepSeek)](https://github.com/deepseek-ai/DualPipe)
