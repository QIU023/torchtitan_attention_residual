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
| 2 | Single-GPU loss-curve alignment vs baseline | ✓ done | `phase2_attnres_baseline_loss/README.md` + Llama3-175M curves |
| 3 | Pipeline-parallel integration + cross-stage cache adapter | ✓ done | `phase3_attnres_pp_integration/adapter_design.md` + `experiments/attn_res/pipeline_adapter.py` |
| 4 | Kimi Linear (48B-A3B faithful) port + scale-up | ✓ done | `experiments/kimi_linear/`; phase4 step-8000 ckpt |
| 5 | Multimodal AttnRes-Kimi-VL dual-arm validation | ✓ done | `phase5_vlm_multimodal_sft/train_mm.py` + LLaVA-Pretrain + Arms 1/2 |
| 6 | Pre-merge infra completeness for upstream PR | ✓ done | `phase6_upstream_pr_prep/launch_8gpu_mm.sh` + alignment matrix + cache-adapter ablation |
| 7 | NCCL fabric pattern catalog under 3D-5D parallelism | ✓ done | `phase7_nccl_traffic_catalog/FINAL_CATALOG.md` + 8+ ixia_config.json |
| 8 | VQA evaluation (qualitative) | ✓ qual / ✗ quant | `phase8_vqa_eval/eval_results/qual_vqa_summary.md` |
| 9-A | SFT post-training | ✓ done | `phase5_vlm_multimodal_sft/runs/sft_v11_llava_instruct_150k_4d/checkpoint/step-490` |
| 9-B | PPO toy (cross-mesh KL fabric) | ✓ done | `phase5_vlm_multimodal_sft/runs/ppo_smoke_no_vllm/` |
| 10 | SGLang inference + RLHF fabric (12 stages A-L) | ✓ done | `phase10_ckpt_dcp_to_hf/PHASE10_FABRIC_REPORT.md` + 12 stage MDs |

Total ~50 NCCL trace runs catalogued; 12+ ixia_config.json files.

## Phase 10 stage breakdown

| Stage | Content | Status | Output |
|---|---|---|---|
| A | SGLang fork submodule + Python deps | ✓ | sglang submodule on `attention_residual_inference` branch |
| B | DCP → HF kimi_linear conversion | ✓ | `phase10_ckpt_dcp_to_hf/dcp_to_hf_kimi_attn_res.py` + 2.6 GB safetensors |
| C | `kimi_block_attn_res.py` SGLang model class | ✓ structurally / ✗ runtime | 409-LOC PR-ready file on fork; can't run due to sgl_kernel env |
| D | 4D forward-only inference fabric trace | ✓ | `phase5_vlm_multimodal_sft/runs/inference_torchtitan_phase4_step8000/` |
| E | Training ↔ inference fabric asymmetry analysis | ✓ | `phase10_ckpt_dcp_to_hf/TRAINING_INFERENCE_FABRIC_ASYMMETRY.md` |
| F | Real PPO smoke (kimi_linear actor + frozen ref) | ✓ | `phase5_vlm_multimodal_sft/runs/ppo_real_torchtitan/` |
| G | Cross-regime aggregate fabric report | ✓ | `phase10_ckpt_dcp_to_hf/PHASE10_FABRIC_REPORT.md` |
| H | Phase 10 cross-references in phase 7 catalog | ✓ | catalog updated |
| I | Two-phase TP RS+AG synthetic demo | ✓ | `phase5_vlm_multimodal_sft/runs/two_phase_tp_{allreduce,rs_ag}/` |
| J | Autoregressive (growing-prefix) inference | ✓ | `phase5_vlm_multimodal_sft/runs/inference_autoregressive_growing/` |
| K | Two-phase RS+AG injection in real-model context | ✓ | `phase5_vlm_multimodal_sft/runs/inference_two_phase_real/` |
| L | Sustained inference workload sweep (4 shapes) | ✓ | `phase5_vlm_multimodal_sft/runs/workload_{short_high_bs,mid,long,prod}/` |
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
   `phase7_nccl_traffic_catalog/expand_to_flows.py` conflates PP and EP at `nranks=2`
   `Send/Recv`. Clean fix is upstream `torch.distributed` PR to
   expose `group_name` in NCCL trace lines.

6. **CP via fla-core ring-recurrence** — KDA blocks CP support; needs
   fla-core upstream PR for `chunk_kda` ring updates.

## Local environment snapshot vs target (8× 5090)

Hardware stays (RTX 5090, 8 cards). Only software stack swaps. The
SGLang `sgl_kernel` wheel ships sm90 + sm100 binaries; SM 120 falls
through, and the wheel ABI also requires CUDA 12.x + Python 3.10-3.13.

| Component | Local snapshot (broken) | Target (next box) | Action |
|---|---|---|---|
| GPU | 8× RTX 5090 (SM 120) | same | keep |
| Driver | 595.58.03 (CUDA 13 capable, also CUDA 12 capable) | same | keep |
| CUDA toolkit | 13.2 | **12.4 (or 12.6)** | downgrade |
| nvcc | 13.2 | 12.4 | downgrade with toolkit |
| Python | **3.14.3** | **3.12.x** | downgrade |
| PyTorch | 2.11.0+cu130 (nightly) | **2.6.x cu124** stable | downgrade |
| torchvision / torchaudio / torchcodec | 0.26 / 2.11 / 0.11 cu130 | matching cu124 builds | downgrade |
| triton | 3.6.0 | 3.2.x (matching torch 2.6) | replaced via torch downgrade |
| transformers | 5.7.0 | 4.46.x (LLaVA + safetensors compatible) | downgrade |
| fla-core | 0.5.0 | 0.5.0 | keep (KDA Triton kernels) |
| **sgl_kernel** | 0.3.21 (broken: `libnvrtc.so.12` missing, `common_ops` SM 120 absent) | 0.3.x cu124 wheel for sm90 | re-install after Python+CUDA swap |
| sglang | 0.0.0.dev (editable, can `import sglang` but `srt.layers.*` blocked by sgl_kernel) | same dev build, fully importable once sgl_kernel loads | resolves automatically |

### Concrete swap script (target box)

```bash
# 1. uv-managed Python 3.12 venv
uv venv --python 3.12 /venv/main_312

# 2. CUDA 12.4 toolkit (apt or conda)
apt-get install cuda-toolkit-12-4   # or conda install cudatoolkit=12.4

# 3. PyTorch 2.6 cu124
uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. SGLang via our editable install
cd /root/torchtitan_attention_residual/sglang/python
uv pip install -e .   # sgl_kernel pulled as dep, should match cu124+py312+sm90

# 5. fla-core (unchanged)
uv pip install fla-core==0.5.0
```

### What "still broken" looks like after the swap

If sgl_kernel cu124+py312 wheel **still** has no SM 120 binary
(possible — most wheel matrices stop at sm90/sm100), two fallbacks:

1. **Build sgl_kernel from source** for SM 120 (~2-4h, needs CUDA
   12.4 nvcc + the source repo at `sgl-project/sgl-kernel`). This
   is the proper fix.
2. **Patch our `sglang/python/sglang/srt/models/kimi_block_attn_res.py`**
   to use `torch.nn.RMSNorm` / `torch.nn.Linear` instead of
   `sglang.srt.layers.{RMSNorm,Linear}` in the AttnRes paths only.
   Slower (no fused norm) but bypasses sgl_kernel for our model
   class. ~4-6h with numerical equivalence test.

The fallback decision lives in `phase11_rlhf_grpo_infra/` (next session) once we see
which sgl_kernel wheel resolution we get on the target box.

## What's runnable on a fresh CUDA 12 + Python 3.12 box (no code change)

- All `phase{2,3,4,5,6}` training paths (single-GPU through 4D)
- `phase6_upstream_pr_prep/launch_8gpu_mm.sh` (SFT / pretrain) and
  `phase{6,9}/run_*_pretrain.sh`
- `phase10_ckpt_dcp_to_hf/run_inference_torchtitan.sh` (4D forward-only inference)
- `phase10_ckpt_dcp_to_hf/run_ppo_real.sh` (PPO smoke at 4D mesh)
- `phase10_ckpt_dcp_to_hf/run_two_phase_*.sh` (RS+AG fabric demos, real + synthetic)
- `phase10_ckpt_dcp_to_hf/run_workload_sweep.sh` (sustained workload sweep)
- `phase10_ckpt_dcp_to_hf/run_autoregressive.sh` (growing-prefix autoregressive)
- `phase10_ckpt_dcp_to_hf/dcp_to_hf_kimi_attn_res.py` (DCP→HF conversion)

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
├── phase2-phase4_kimi_attnres_lm_pretrain/         # single-GPU + PP + Kimi Linear scaffolding
├── phase5_vlm_multimodal_sft/
│   ├── train_mm.py, multimodal_*.py, generate_caption.py
│   ├── runs/README.md     # categorical index of ~50 run dirs
│   └── runs/{v11_4d_*, v12_4d_*, sft_v11_*, inference_*, ppo_*,
│              workload_*, two_phase_*, ...}
├── phase6_upstream_pr_prep/                # ACTIVE: launch_8gpu_mm.sh +
│                          # run_v11_pretrain.sh, run_v12_pretrain.sh
│   ├── alignment_archive/
│   ├── cache_adapter_ablation/
│   ├── summaries_archive/
│   ├── pretrain_archive/
│   ├── torchtitan_patches/
│   ├── logs/
│   └── utils/
├── phase7_nccl_traffic_catalog/                # FINAL_CATALOG.md + extract/expand/ixia
│                          # pipeline scripts
├── phase8_vqa_eval/                # qual VQA eval
├── phase9_post_training_ppo_trace/                # SFT (run_sft_pretrain.sh) + ppo toy
└── phase10_ckpt_dcp_to_hf/               # 12 stages of inference + RLHF fabric:
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
