# Project Status — Block AttnRes for torchtitan (multi-phase)

Snapshot date: **2026-05-06**. Machine swap pending — this doc plus the
git history is the ground truth for picking up on a fresh box.

## Project goal

Validate, scale, and profile Block Attention Residual (Block AttnRes,
from Kimi paper §5) inside the torchtitan training framework, and
deliver fabric / NCCL traces consumable by IXIA so a wire-level test
can model the production communication pattern across pretrain,
post-train, inference, and RLHF regimes.

## Phase status table

| Phase | Goal | Status | Headline artifacts |
|---|---|---|---|
| 1 | Repository scaffold + AttnRes math reference | ✓ done | `torchtitan/torchtitan/experiments/attn_res/` |
| 2 | Single-GPU loss-curve alignment vs baseline | ✓ done | `phase2/README.md` + Llama3-175M curves |
| 3 | Pipeline-parallel integration + cross-stage cache adapter | ✓ done | `phase3/adapter_design.md` + `experiments/attn_res/pipeline_adapter.py` |
| 4 | Kimi Linear (48B-A3B faithful) port + scale-up | ✓ done | `experiments/kimi_linear/`; phase4 step-8000 ckpt |
| 5 | Multimodal AttnRes-Kimi-VL dual-arm validation | ✓ done | `phase5/train_mm.py` + LLaVA-Pretrain + Arms 1/2 |
| 6 | Pre-merge infra completeness for upstream PR | ✓ done | `phase6/launch_8gpu_mm.sh` + alignment matrix + cache-adapter ablation |
| 7 | NCCL fabric pattern catalog under 3D-5D parallelism | ✓ done | `phase7/FINAL_CATALOG.md` + 8+ ixia_config.json |
| 8 | VQA evaluation (qualitative) | ✓ qual / ✗ quant | `phase8/eval_results/qual_vqa_summary.md` |
| 9-A | SFT post-training | ✓ done | `phase5/runs/sft_v11_llava_instruct_150k_4d/checkpoint/step-490` |
| 9-B | PPO toy (cross-mesh KL fabric) | ✓ done | `phase5/runs/ppo_smoke_no_vllm/` |
| 10 | SGLang inference + RLHF fabric (12 stages A-L) | ✓ done | `phase10/PHASE10_FABRIC_REPORT.md` + 12 stage MDs |

Total ~50 NCCL trace runs catalogued; 12+ ixia_config.json files.

## Phase 10 stage breakdown

| Stage | Content | Status | Output |
|---|---|---|---|
| A | SGLang fork submodule + Python deps | ✓ | sglang submodule on `attention_residual_inference` branch |
| B | DCP → HF kimi_linear conversion | ✓ | `phase10/dcp_to_hf_kimi_attn_res.py` + 2.6 GB safetensors |
| C | `kimi_block_attn_res.py` SGLang model class | ✓ structurally / ✗ runtime | 409-LOC PR-ready file on fork; can't run due to sgl_kernel env |
| D | 4D forward-only inference fabric trace | ✓ | `phase5/runs/inference_torchtitan_phase4_step8000/` |
| E | Training ↔ inference fabric asymmetry analysis | ✓ | `phase10/TRAINING_INFERENCE_FABRIC_ASYMMETRY.md` |
| F | Real PPO smoke (kimi_linear actor + frozen ref) | ✓ | `phase5/runs/ppo_real_torchtitan/` |
| G | Cross-regime aggregate fabric report | ✓ | `phase10/PHASE10_FABRIC_REPORT.md` |
| H | Phase 10 cross-references in phase 7 catalog | ✓ | catalog updated |
| I | Two-phase TP RS+AG synthetic demo | ✓ | `phase5/runs/two_phase_tp_{allreduce,rs_ag}/` |
| J | Autoregressive (growing-prefix) inference | ✓ | `phase5/runs/inference_autoregressive_growing/` |
| K | Two-phase RS+AG injection in real-model context | ✓ | `phase5/runs/inference_two_phase_real/` |
| L | Sustained inference workload sweep (4 shapes) | ✓ | `phase5/runs/workload_{short_high_bs,mid,long,prod}/` |
| M | commId-aware axis labels | deferred | needs upstream `torch.distributed` PR |

## Production-grade gaps (next-machine TODO)

### High-priority (project-critical)

1. **SGLang kimi_block_attn_res runtime** — the PR-ready model class
   on `attention_residual_inference` branch can't import on this box
   because sgl_kernel needs:
   - SM 120 wheel (have sm90/sm100 only) — RTX 5090 has no compiled
     variant
   - CUDA 12 ABI (libnvrtc.so.12, libcublas.so.12, libcudart.so.12)
     — we have CUDA 13
   - Python 3.10–3.13 (we have 3.14)
   On a fresh box with CUDA 12 + Python 3.12 + supported GPU (sm90),
   the wheel should install cleanly and the model should load via
   `EntryClass = KimiBlockAttnResForCausalLM`.
   - Once loadable, SGLang inference + PPO production both unblock.

2. **Real KV cache (KDA + MLA)** — `KimiMLAAttention.forward` and
   `KimiDeltaAttention.forward` in
   `torchtitan/torchtitan/experiments/kimi_linear/model.py` both
   carry `# No cache path — training only` comments. KDA needs the
   recurrent state plumbed through fla-core's `chunk_kda`
   `initial_state`/`output_final_state` API; MLA cache is the
   standard append-on-update pattern. Without this, autoregressive
   single-token decode fabric (Stage J ideal-cache mode) is blocked
   by fla-core Triton autotuner at seq=1.

3. **Quantitative multimodal eval (lmms-eval)** — Phase 8 only has
   qualitative side-by-side. To produce VQAv2 / GQA / ScienceQA /
   MMVet / OCRBench scores, need:
   - `lmms-eval` package (~30 min install if env friendly)
   - HF `AutoModel` registration for kimi_linear AttnRes model_type
   - HF safetensors with vision_tower + projector packed in
     (Phase 10 Stage B converts LM only; needs extension to multimodal)

### Mid-priority (optimization / depth)

4. **Two-phase computation full integration** — Stage K *injects*
   RS+AG ops to demonstrate the fabric pattern; production-grade is
   replacing `o_proj`'s `RowwiseParallel` AllReduce with
   `Shard(seq_dim)` output + explicit `AllGather` in
   `apply_tp_kimi_linear`. ~1 day work + numerical equivalence test.

5. **commId-aware axis labels** — heuristic in
   `phase7/expand_to_flows.py` conflates PP and EP at `nranks=2`
   `Send/Recv`. Clean fix is upstream `torch.distributed` PR to
   expose `group_name` in NCCL trace lines.

6. **CP via fla-core ring-recurrence** — KDA blocks CP support; needs
   fla-core upstream PR for `chunk_kda` ring updates.

## What's runnable on a fresh CUDA 12 + Python 3.12 box (no code change)

- All `phase{2,3,4,5,6}` training paths (single-GPU through 4D)
- `phase6/launch_8gpu_mm.sh` (SFT / pretrain) and
  `phase{6,9}/run_*_pretrain.sh`
- `phase10/run_inference_torchtitan.sh` (4D forward-only inference)
- `phase10/run_ppo_real.sh` (PPO smoke at 4D mesh)
- `phase10/run_two_phase_*.sh` (RS+AG fabric demos, real + synthetic)
- `phase10/run_workload_sweep.sh` (sustained workload sweep)
- `phase10/run_autoregressive.sh` (growing-prefix autoregressive)
- `phase10/dcp_to_hf_kimi_attn_res.py` (DCP→HF conversion)

The above already produced the full fabric catalog on the current
machine. The remaining unmade products require **either** a fresh env
(SGLang inference, lmms-eval) **or** dedicated engineering days
(KV cache port, two-phase full integration).

## Repository tree (post-2026-05-06 cleanup)

```
torchtitan_attention_residual/
├── README.md, ROOT_PLAN.md, RFC_DRAFT_v{2,3}.md
├── PROJECT_STATUS.md      # this file
├── PHASE7_8_9_REPORT.md   # consolidated post-pretrain report
├── torchtitan/            # submodule, fork of pytorch/torchtitan
├── sglang/                # submodule, fork of sgl-project/sglang on
│                          # branch attention_residual_inference
├── phase2-phase4/         # single-GPU + PP + Kimi Linear scaffolding
├── phase5/
│   ├── train_mm.py, multimodal_*.py, generate_caption.py
│   ├── runs/README.md     # categorical index of ~50 run dirs
│   └── runs/{v11_4d_*, v12_4d_*, sft_v11_*, inference_*, ppo_*,
│              workload_*, two_phase_*, ...}
├── phase6/                # ACTIVE: launch_8gpu_mm.sh +
│                          # run_v11_pretrain.sh, run_v12_pretrain.sh
│   ├── alignment_archive/
│   ├── cache_adapter_ablation/
│   ├── summaries_archive/
│   ├── pretrain_archive/
│   ├── torchtitan_patches/
│   ├── logs/
│   └── utils/
├── phase7/                # FINAL_CATALOG.md + extract/expand/ixia
│                          # pipeline scripts
├── phase8/                # qual VQA eval
├── phase9/                # SFT (run_sft_pretrain.sh) + ppo toy
└── phase10/               # 12 stages of inference + RLHF fabric:
    ├── PHASE10_FABRIC_REPORT.md
    ├── TRAINING_INFERENCE_FABRIC_ASYMMETRY.md
    ├── TWO_PHASE_TP_FABRIC_DEMO.md
    ├── TWO_PHASE_REAL_MODEL_FABRIC.md
    ├── AUTOREGRESSIVE_FABRIC.md
    └── SUSTAINED_INFERENCE_WORKLOAD.md
```

## Git state

- `main` HEAD: `7879ee7` (post-reorg, all Phase 10 stages pushed)
- `sglang` submodule HEAD: `4a27b32e1` on `attention_residual_inference`
- All commits pushed; nothing un-staged that's project-critical.

`PROJECT_STATUS.md` (this file) supersedes
`PHASE7_8_9_REPORT.md` as the entry-point doc when picking up on a
fresh box; keep `PHASE7_8_9_REPORT.md` for the deeper Phase 7-9
narrative.
