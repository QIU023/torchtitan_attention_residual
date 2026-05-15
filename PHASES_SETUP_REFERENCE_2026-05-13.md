# Phases Setup Reference — 2026-05-13

Per-phase setup steps, hardware requirements, and entry-point scripts. Use this to bootstrap any phase from scratch on a fresh box.

## Global prerequisites (all phases)

| Item | Spec |
|---|---|
| Python | 3.11 |
| CUDA | 12.9-13.0 |
| GPU mem (min) | 24 GB per rank (32 GB for Kimi 447M FSDP=8 with FP8) |
| Disk | 50 GB for Phase 2-3, ~250 GB for Phase 4-11 (ckpt + HF cache + dataset) |
| Network | PCIe Gen4 OK; NVLink preferred for Phase 7+ NCCL studies |
| Conda env | `attnres` (created by `phase2_attnres_baseline_loss/setup_env.sh`) |

**Universal bootstrap**:
```bash
git clone -b main git@github.com:QIU023/torchtitan_attention_residual.git
cd torchtitan_attention_residual
git submodule update --init torchtitan
bash phase2_attnres_baseline_loss/setup_env.sh                              # creates conda env + tokenizer + smoke tests
python /usr/bin/python3 -m pip install fla-core       # KDA kernel (Phase 4+)
```

Per the working memory: torchrun uses system python (`/usr/bin/python3`), so install deps via system python rather than `/venv/main`.

---

## Phase 2 — Block AttnRes Loss-Curve Alignment

**Purpose**: Validate Block Attention Residual primitive on Llama3-175M; compare loss curves against baseline over ~650M tokens.

**Status**: ✅ Complete. Artifacts in `phase2_attnres_baseline_loss/runs/{baseline,attn_res}/`.

**Hardware**: 1× GPU ≥24 GB; 50 GB disk.

**Software env**:
- PyTorch nightly + torchtitan fork (`feat/block-attn-res`)
- Llama-3.1 tokenizer (ungated from `NousResearch/Meta-Llama-3.1-8B`)
- TensorBoard

**Setup**:
```bash
bash phase2_attnres_baseline_loss/setup_env.sh        # conda env "attnres" + smoke tests + tokenizer download
conda activate attnres
```

**Entry points**:
- **Launcher**: `bash phase2_attnres_baseline_loss/launch.sh` — tmux session with baseline (20k steps) → attn_res (20k steps)
- **Overrides**: `STEPS=N bash phase2_attnres_baseline_loss/launch.sh`
- **Compare**: `python phase2_attnres_baseline_loss/compare_losses.py --baseline phase2_attnres_baseline_loss/runs/baseline/tb --attn_res phase2_attnres_baseline_loss/runs/attn_res/tb --out phase2_attnres_baseline_loss/runs/comparison.png`
- **Smoke (no torchtitan)**: `python phase2_attnres_baseline_loss/smoke_test_attn_res.py`
- **Monitor**: `tensorboard --logdir phase2_attnres_baseline_loss/runs --port 6006`

---

## Phase 3 — PP Pressure Tests + Cache Adapter

**Purpose**: Verify Block AttnRes under Interleaved 1F1B PP × VP with cross-stage cache adapter; prove per-stage bandwidth is constant.

**Status**: ✅ Complete. Final report: `phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md`. Max |Δ loss adapter vs naive| = 0.0044 on L=16 Llama3, +0.011/+0.04 on Kimi 48B-layout.

**Hardware**: 8× GPU (PCIe or NVLink); ~45 GB disk for C4 shards.

**Software env**: Same as Phase 2 + fla-core (for Kimi tests).

**Setup**:
```bash
bash phase2_attnres_baseline_loss/setup_env.sh
python phase3_attnres_pp_integration/prefetch_c4.py            # 150 C4 shards → HF cache, ~45 GB
python phase3_attnres_pp_integration/fake_pg_test.py           # optional: single-GPU PP smoke
```

**Entry points**:
- **Orchestrator**: `bash phase3_attnres_pp_integration/go_8gpu.sh` — env check → install → C4 prefetch → naive + adapter PP smoke → loss compare
- **Naive PP**: `NGPU=8 bash phase3_attnres_pp_integration/launch_8gpu_naive.sh`
- **Adapter PP**: `NGPU=8 TORCHTITAN_ATTNRES_CACHE=1 bash phase3_attnres_pp_integration/launch_8gpu_adapter.sh`
- **L=16 pressure test**: `bash phase3_attnres_pp_integration/run_pp_pressure_test.sh` — PP=4/8 × VP=2/4 grid with `--checkpoint.no-enable`
- **Compare**: `python phase3_attnres_pp_integration/compare_pp_vs_single.py --single phase3_attnres_pp_integration/runs/single_ref/tb --pp ... --pp_cached ...`

---

## Phase 4 — Kimi LM Pretrain (current focus)

**Purpose**: From-scratch Kimi Linear 447M (KDA+MLA+MoE+AttnRes) on C4 with paper Table-2 hparams + FP8 quantization.

**Status**: 🔄 In progress. Current run: `phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_B/` (lr=1.5e-3, warmup=1000, GBS=384, FP8 rowwise, target 12750 steps = 10.03 B tokens, ETA ~4 days = 2026-05-17). First ckpt at step 200 ✓.

**Hardware**: 4-8× GPU (5090 32 GB tested). 8 GPU: 28.5s/step = 102 M tok/h.

**Software env**:
- torchtitan with `kimi_linear` experiment module
- **fla-core** ≥ 0.5.0 (`chunk_kda`, `fused_recurrent_kda`, `fused_kda_gate` for KDA layer)
- **torchao** 0.17.0 (FP8 Float8LinearConverter + grouped_mm)
- Llama-3.1 BPE tokenizer (vocab 128,256 → 163,840 with Kimi extras)
- C4 dataset (streamed from HF or prefetched via Phase 3)

**Setup**:
```bash
bash phase2_attnres_baseline_loss/setup_env.sh
python /usr/bin/python3 -m pip install fla-core torchao
python -m pytest torchtitan/experiments/kimi_linear/tests/ -q   # CPU validation
```

**Entry points**:
- **Stage 0 redo (current production)**: `NGPU=8 bash phase4_kimi_attnres_lm_pretrain/launch_redo_paperalign_10B.sh`
  - Env overrides: `LOCAL_BS=4 GLOBAL_BS=384 LR=1.5e-3 WARMUP=1000 STEPS=12750 SAVE_FREQ=200 KEEP_K=2`
  - Output: `phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_B/`
- **Original from-scratch (pre-redo)**: `bash phase4_kimi_attnres_lm_pretrain/launch_from_scratch_paperhparams.sh` (4-GPU, 2.5B tokens — undertrained)
- **Continuation (failed attempt)**: `bash phase4_kimi_attnres_lm_pretrain/launch_continuation_100k.sh`
- **Downscale sweep (48B carrier search)**: `bash phase4_kimi_attnres_lm_pretrain/run_kimi48b_downscale_sweep.sh`
- **Monitor**: `tail -f phase4_kimi_attnres_lm_pretrain/runs/*/train.log` + `tensorboard --logdir phase4_kimi_attnres_lm_pretrain/runs/*/tb`

**Critical knobs (Stage 0)**:
- `SAVE_FREQ=200` (not the flavor's default 1000 nor my original 2500) — 92 min between saves, 0.55% wall-clock overhead
- `keep_latest_k=2` — bound disk to 34 GB ongoing (17 GB per ckpt)
- `--checkpoint.no-enable` for smoke/pressure runs only

---

## Phase 5 — Multimodal Caption Pretrain (LLaVA-Pretrain)

**Purpose**: SigLIP-Base (frozen) + 2-layer projector (trainable) + Phase 4 LM (trainable) on LLaVA-Pretrain-558K captions.

**Status**: Stage 1 caption pretrain done (loss 2.23→1.85). Will need rerun with Phase 4 redo ckpt.

**Hardware**: 4-8× GPU; 32 GB VRAM per rank; ~20 GB disk for LLaVA images.

**Software env**:
- Phase 4 step-12500 (or step-25500 from redo) DCP ckpt
- SigLIP-Base (`google/siglip-base-patch16-224`, ~92 M frozen)
- LLaVA-Pretrain-558K dataset

**Setup**:
```bash
conda activate attnres
python phase5_vlm_multimodal_sft/data_prep.py             # downloads LLaVA images + metadata, ~10-20 GB
ls phase4_kimi_attnres_lm_pretrain/runs/.../checkpoint/step-12500   # validate base LM ckpt present
STEPS=5 LOCAL_BS=2 bash phase5_vlm_multimodal_sft/launch_train.sh   # 5-step single-GPU smoke
```

**Entry points**:
- **Arm 1 (FSDP)**: `bash phase5_vlm_multimodal_sft/launch_train.sh` — FSDP=4 PP=1, 3 epochs ~5h
  - Overrides: `STEPS=20000 LOCAL_BS=8 GLOBAL_BS=32`
- **Arm 2 (PP + cache adapter)**: `NGPU=4 PP=4 V=2 LOCAL_BS=1 GLOBAL_BS=12 STEPS=2000 ADAPTER=1 bash phase5_vlm_multimodal_sft/launch_pp_adapter.sh`
- **Caption eval**: `bash phase5_vlm_multimodal_sft/eval_caption.sh`
- **Cross-validate**: `python phase5_vlm_multimodal_sft/compare_pp_vs_fsdp.py --pp ... --fsdp ...`

---

## Phase 5_distillation (DEPRECATED)

**Status**: ❌ Negative result, archived. Online KD with Llama-3.1-8B teacher + MiniPLM data filtering both degraded val loss. Multimodal work moved on with raw Phase 4 ckpt. **Do not re-run** — same hardware + budget → same negative result.

See `phase5_distillation_deprecated/README.md`.

---

## Phase 6 — Multi-Dimensional Parallelism (FSDP+PP+TP+EP+CP)

**Purpose**: Fill parallelism gaps before torchtitan upstream PR; validate PP=8×VP=4, TP+PP+AttnRes composition, async DCP, multimodal scatter, cross-parallelism determinism.

**Status**: 🔄 In flight. 4-GPU alignment complete (median |Δ|=0.024 nats). 8-GPU matrix queued.

**Hardware**: 8× RTX 5090 PCIe; ~100 GB disk.

**Software env**: Same as Phase 4 + parallelism plans (`parallelize_kimi_linear`).

**Setup**:
```bash
bash phase2_attnres_baseline_loss/setup_env.sh
rsync -av source:/phase4_kimi_attnres_lm_pretrain/step-8000 /destination/phase4_kimi_attnres_lm_pretrain/    # transfer LM seed ckpt
```

**Entry points**:
- **Generic 8-GPU launcher (driver for Phase 6+7+9+10)**: `phase6_upstream_pr_prep/launch_8gpu_mm.sh`
  - Parameterized: `FSDP=N PP=N TP=N EP=N CP=N STEPS=N LOCAL_BS=N GLOBAL_BS=N OUT_DIR=...`
  - Tier C alignment: `GBS=12 STEPS=500`
  - Tier B production: `GBS=120 STEPS=50`
  - Tier A production: `GBS=384 STEPS=100` (paper Table-2 match)
- **Orchestrators**: `alignment_archive/run_a1_alignment.sh`, `alignment_archive/run_all_pp_pressure.sh`
- **Perf regression check**: `python phase6_upstream_pr_prep/perf_regression_check.py` (5% tolerance)
- **Disk discipline**: `phase6_upstream_pr_prep/disk_watchdog.sh` — runs in background, alarms < 50 GB free

---

## Phase 7 — NCCL Pattern Catalog

**Purpose**: Record NCCL collective sequences (AllReduce, ReduceScatter, Send/Recv, AllToAll) across 8-GPU 3D parallelism matrix; 6 configs × 4 tiers = 24 traces.

**Status**: 🔄 In flight. Tier-C piggybacking on Phase 6.

**Hardware**: 8× GPU; NVIDIA nsys + NCCL debug.

**Software env**:
```bash
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,INIT
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
```

**Setup**:
```bash
# Set env vars above, then run via phase6_upstream_pr_prep/launch_8gpu_mm.sh with TRACE_TIER set
FSDP=2 PP=2 TP=2 NGPU=8 STEPS=100 GBS=384 SEQ_LEN=2048 \
  TRACE_TIER=tier_a TRACE_STEPS=100 \
  OUT_DIR=phase7_nccl_traffic_catalog/traces/config1_fsdp2pp2tp2 \
  bash phase6_upstream_pr_prep/launch_8gpu_mm.sh
```

**Entry points**:
- **Full sweep**: `bash phase7_nccl_traffic_catalog/run_all_traces.sh`
- **Parser**: `python phase7_nccl_traffic_catalog/extract_collectives.py phase7_nccl_traffic_catalog/traces/<config>/tier_X/nccl-rank-*.log` → `collective_summary.csv`
- **Catalog**: `phase7_nccl_traffic_catalog/FINAL_CATALOG.md`, `phase7_nccl_traffic_catalog/pattern_catalog.md`
- **Auto-publish**: `bash phase7_nccl_traffic_catalog/publish_archive.sh`

---

## Phase 8 — VQA Evaluation

**Purpose**: Quantify VLM ckpt on VQAv2, GQA, ScienceQA.

**Status**: ⏳ Pending (deferred until Phase 4 redo + Phase 5 rerun finishes).

**Hardware**: 1-2× GPU for inference.

**Software env**: `lmms-eval` (LMMs-Lab) + HF transformers + DCP→HF converter.

**Setup**:
```bash
python phase8_vqa_eval/dcp_to_hf.py --dcp-ckpt phase4_kimi_attnres_lm_pretrain/runs/.../step-12500 --output phase11_rlhf_grpo_infra/hf/kimi_447m_attnres.safetensors
pip install lmms-eval
```

**Entry points**:
- **Convert**: `python phase8_vqa_eval/dcp_to_hf.py --dcp-ckpt <path> --output <safetensors>`
- **Eval**: `bash phase8_vqa_eval/run_vqa_eval.sh` (VQAv2, GQA, ScienceQA)
- **Qualitative**: `bash phase8_vqa_eval/run_qual_vqa.sh`
- **Results**: `phase8_vqa_eval/eval_results/<ckpt>_<benchmark>.json`

---

## Phase 9 — Post-Training (SFT + PPO Infrastructure)

**Purpose**: 9-A: visual instruction SFT on LLaVA-Instruct-150K. 9-B: PPO infra smoke (4-model mesh).

**Status**: ⏳ Pending Phase 4 redo + Phase 5 rerun. PPO `PPO_TRACE_DEFERRED.md` documents deferred trace work.

**Hardware**: 8× GPU (same as Phase 6).

**Software env**: Same as Phase 6 + OpenRLHF for PPO.

**Setup**:
```bash
STUDENT_CKPT=phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_B/checkpoint/step-12500 \
  bash phase9_post_training_ppo_trace/run_sft_pretrain.sh

python phase9_post_training_ppo_trace/ppo_actor_ref_real_ckpt.py    # validate on real ckpt before full run
```

**Entry points**:
- **SFT**: `bash phase9_post_training_ppo_trace/run_sft_pretrain.sh` (Tier B trace auto-captured)
- **SFT dataset**: `python phase9_post_training_ppo_trace/multimodal_sft_dataset.py`
- **PPO smoke**: `bash phase9_post_training_ppo_trace/run_ppo_smoke.sh` (OpenRLHF, 4-model mesh)
- **PPO validation**: `python phase9_post_training_ppo_trace/ppo_actor_ref_real_ckpt.py`

---

## Phase 10 — Autoregressive Inference Fabric

**Purpose**: Capture inference-phase NCCL fabric (growing-prefix generation); validate Stage J communication patterns.

**Status**: 🔄 Partial. Growing-prefix mode traced (67.3s). Single-token blocked on KDA Triton autotuner at seq_len=1. Real KV-cache port → Phase 11.

**Hardware**: 8× GPU.

**Software env**: Same as Phase 6 + fla-core KDA (with autotuner constraint).

**Setup**:
```bash
python phase10_ckpt_dcp_to_hf/inference_autoregressive.py \
  --ckpt phase4_kimi_attnres_lm_pretrain/runs/.../step-12500 \
  --mode growing \
  --prompts 20 --tokens-per-gen 20 --initial-prefix 64
```

**Entry points**:
- **Generation loops**: `python phase10_ckpt_dcp_to_hf/inference_autoregressive.py --mode {growing,single_token}`
- **Launcher**: `bash phase10_ckpt_dcp_to_hf/run_autoregressive.sh`
- **Two-phase smoke**: `bash phase10_ckpt_dcp_to_hf/run_two_phase_smoke.sh`
- **TT inference**: `bash phase10_ckpt_dcp_to_hf/run_inference_torchtitan.sh`
- **PPO real**: `bash phase10_ckpt_dcp_to_hf/run_ppo_real.sh`

---

## Phase 11 — SGLang AttnRes Overlay + SFT + RLHF (largest phase)

**Purpose**: Bring Block AttnRes to SGLang as reusable overlay; benchmark TP=1/8 + 3D mesh; ckpt convert DCP↔HF; SFT + GRPO/PPO actor-ref smoke.

**Status**: ✅ Most components complete (5/5 perf optimizations, 27/27 audit items closed). **Production blocker**: flashinfer_mla bf16 NaN on RTX 5090 SM 12.0 → mitigated by **fp32 MLA fallback in overlay** (env-gated, ATTNRES_MLA_FP32_FALLBACK=1).

**Hardware**: 8× RTX 5090 PCIe 32 GB; **+27% decode tps** with Phase-2 fused Triton kernel.

**Software env**:
- SGLang fork: `attention_residual_inference` branch at `b3f6b543f` (just merged vlm-sglang-overlay + local fp32 fallback)
- torch 2.11 + torchao 0.17
- flashinfer (for MLA), fla-core (for KDA at training time)
- Phase 4 step-12500 ckpt converted to HF safetensors

**SGLang fork setup**:
```bash
# Clone fork
git clone -b attention_residual_inference git@github.com:QIU023/sglang.git /sgl-workspace/sglang
cd /sgl-workspace/sglang
pip install -e python/

# Or apply patches in-place to existing sglang
# (see phase11_rlhf_grpo_infra/TORCHTITAN_VAST_AI_PATCHES.md for vast.ai-specific patches)
```

**Ckpt conversion**:
```bash
# HF → DCP (LM only)
python phase11_rlhf_grpo_infra/hf_to_dcp_kimi_attn_res.py --in /path/hf --out /path/dcp \
    --config kimi_linear_447m_aligned_block_attn_res_n4

# DCP → HF (VLM)
torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
    --in phase5_vlm_multimodal_sft/runs/.../step-2500 \
    --out phase11_rlhf_grpo_infra/hf/vlm_pretrain \
    --config kimi_linear_447m_aligned_block_attn_res_n4 \
    --vision-tower google/siglip-base-patch16-224

# Dummy smoke ckpts
python phase11_rlhf_grpo_infra/dump_aligned_smoke.py --out phase11_rlhf_grpo_infra/hf_aligned   # 1.4B Kimi
python phase11_rlhf_grpo_infra/dump_qwen3_attn_res_smoke.py --out phase11_rlhf_grpo_infra/hf_qwen3   # 120M Qwen3
```

**FP8 MLA fallback (production unblock for Blackwell)**:
```bash
export ATTNRES_MLA_FP32_FALLBACK=1   # 必须 — fp32 MLA prefill, bypass flashinfer_mla NaN
export ATTNRES_FP32_NORM=1           # 推荐 — RMSNorm in fp32 to avoid bf16 outlier overflow
export ATTNRES_INPUT_CLAMP=32        # 可选 — hard clamp post-RMSNorm input magnitude
```

**Entry points**:
- **4-mode bench**: `python phase11_rlhf_grpo_infra/bench_attn_res.py --mode {vanilla,naive,two_phase,shard} --tp {1,8} --context 16384`
- **Full bench suite**: `bash phase11_rlhf_grpo_infra/run_all_traces.sh` (TP=1/8 × 3D mesh × 4 modes)
- **Long context**: `bash phase11_rlhf_grpo_infra/run_long_ctx_bench.sh` (4K-24K context sweep)
- **VLM SFT**: `bash phase11_rlhf_grpo_infra/run_sft_447m_llava_instruct_150k.sh` (LLaVA-Instruct-150K, GBS=64 SEQ=512 LR=2e-5)
- **VLM SFT 3ep continuation**: `bash phase11_rlhf_grpo_infra/run_stage2_continuation.sh`
- **SFT eval**: `python phase11_rlhf_grpo_infra/eval_sft_3ep_qualitative.py --model-path phase11_rlhf_grpo_infra/hf/sft_3ep`
- **VLM smoke**: `bash phase11_rlhf_grpo_infra/post_sft_vlm_smoke.sh`
- **Memory probe**: `python phase11_rlhf_grpo_infra/probe_memory.py`
- **CUDA graph check**: `python phase11_rlhf_grpo_infra/probe_cuda_graph.py`
- **PPO actor+ref validation**: `python phase11_rlhf_grpo_infra/ppo_actor_ref_real_ckpt.py`
- **GRPO orchestrator**: `bash phase11_rlhf_grpo_infra/run_overnight_full_pipeline.sh` → `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py`
- **PP→SFT→GRPO chain**: `bash phase11_rlhf_grpo_infra/run_pp_then_sft_grpo.sh`

**Key docs to read first**:
- `phase11_rlhf_grpo_infra/PHASE11_SGLANG_REPORT.md` — overall status
- `phase11_rlhf_grpo_infra/SGLANG_ATTNRES_AUDIT.md` — A1-A6/B1-B4/C1-C5 closure
- `phase11_rlhf_grpo_infra/PROFILING_REPORT.md` — +27% decode tps validation
- `phase11_rlhf_grpo_infra/SGLANG_PR_PROPOSALS.md` — upstream PR plan
- `phase11_rlhf_grpo_infra/B5_ATTNRES_INFERENCE_KV_CACHE.md` — KV cache design rationale
- `phase11_rlhf_grpo_infra/SGLANG_ATTNRES_INFERENCE_SUMMARY.md` — file-by-file LOC inventory

---

## Summary table

| Phase | Goal | Status | Final artifact | Entry point |
|---|---|---|---|---|
| **2** | Single-GPU loss alignment | ✅ Complete | Loss plot + numbers | `bash phase2_attnres_baseline_loss/launch.sh` |
| **3** | 8-GPU PP + cache adapter | ✅ Complete | `PRESSURE_TEST_REPORT_2026-05-12.md` | `bash phase3_attnres_pp_integration/go_8gpu.sh` |
| **4** | Kimi LM pretrain FP8 (stage 0 redo) | 🔄 Running | Target: step-12750 / 10 B tokens | `NGPU=8 bash phase4_kimi_attnres_lm_pretrain/launch_redo_paperalign_10B.sh` |
| **5** | Multimodal FSDP + PP arms | 🟡 Done v1, redo TBD | Caption loss curve | `bash phase5_vlm_multimodal_sft/launch_train.sh` |
| **5-dep** | KD experiments | ❌ Negative | archived docs | — |
| **6** | Multi-dim parallelism | 🟡 In flight | Alignment matrix + traces | `phase6_upstream_pr_prep/launch_8gpu_mm.sh` |
| **7** | NCCL pattern catalog | 🟡 In flight | `pattern_catalog.md` | `bash phase7_nccl_traffic_catalog/run_all_traces.sh` |
| **8** | VQA eval | ⏳ Pending | `eval_results/*.json` | `bash phase8_vqa_eval/run_vqa_eval.sh` |
| **9** | SFT + PPO infra | ⏳ Pending | sft+ppo logs | `bash phase9_post_training_ppo_trace/run_sft_pretrain.sh` |
| **10** | Autoregressive fabric | 🟡 Partial | Growing-prefix traces | `bash phase10_ckpt_dcp_to_hf/run_autoregressive.sh` |
| **11** | SGLang overlay + SFT + RLHF | ✅ Most done | `b3f6b543f` + bench results | `bash phase11_rlhf_grpo_infra/run_all_traces.sh` |

## Common pitfalls

1. **torchrun uses system Python** — install deps via `/usr/bin/python3 -m pip`, **not** into `/venv/main` (memory note `reference_torchrun_uses_system_python.md`)
2. **tyro CLI boolean conventions** — use `--checkpoint.no-enable`, not `=false` or `=False` (memory note `reference_tyro_cli_bools.md`)
3. **Smoke/pressure runs save no ckpt** — pass `--checkpoint.no-enable`; production runs keep_latest_k=2 (memory note `feedback_no_ckpt_smoke_pressure.md`)
4. **Fork default branches** — torchtitan → `attention_residual_dev`, sglang → `attention_residual_inference`; don't open new branches (memory note `feedback_fork_default_branches.md`)
5. **SGLang install path drift** — code changes in user submodule (`/root/torchtitan_attention_residual/sglang/`) won't fire unless cp'd to install path (`/sgl-workspace/sglang/`). Add `print(sglang.srt.layers.attn_res.__file__)` to bench scripts as verification (lesson from `PROFILING_REPORT.md`)
6. **flashinfer_mla NaN on RTX 5090** — set `ATTNRES_MLA_FP32_FALLBACK=1` for inference; KDA + MoE layers OK, only MLA 4 layers need fallback (Phase 11)
