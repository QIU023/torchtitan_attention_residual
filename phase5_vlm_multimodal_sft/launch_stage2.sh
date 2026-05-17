#!/usr/bin/env bash
# Phase 5 Stage 2 — LLaVA-1.5 paper recipe: VISUAL INSTRUCTION TUNING.
#
# Full-parameter SFT (projector + LM, vision tower still frozen) on the
# LLaVA-1.5 mix665k Visual Instruction dataset. Loads the Stage-1
# alignment ckpt (well-calibrated projector + base LM) and trains both
# projector and LM to follow visual instructions.
#
# Differences from launch_stage1.sh:
#   - NO --mm.freeze-lm                    LM is trainable here
#   - --mm.layout sft                      LlavaInstructSFTDataset (multi-turn, gpt-only loss)
#   - LR=2e-5  PROJ_LR_MULT=1              LLaVA-1.5 paper LR (same for both, low for LM-FT)
#   - GBS=128  STEPS=5200                  1 epoch of 665K @ gbs128
#   - warmup=156 (~3%)                     paper match
#   - cosine decay over last 20%           --lr_scheduler.decay_ratio 0.2
#   - --mm.global-seq-len 580              196 vision + 384 text (dataset default)
#   - --checkpoint.initial_load_path STAGE1_CKPT (carries projector forward)
#
# Expected wall clock: ~3-5h on 8×5090 (665K samples, multi-turn, gbs128).
#
# Prereqs:
#   1. Stage 1 finished → STAGE1_CKPT exists with trained projector
#   2. Instruct-665K data downloaded → mix665k JSON + images present
#      (see download_instruct_665k.sh)
#
# Output: runs/stage2_instruct_sft_447m/
#   checkpoint/step-NNNN/   ← projector + LM both updated
#   tb/                     ← tensorboard for SFT loss curve

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

# ---- knobs ----
STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4_fp8}"
STAGE1_CKPT="${STAGE1_CKPT:-${SCRIPT_DIR}/runs/stage1_alignment_447m/checkpoint/step-2000}"
INSTRUCT_DIR="${INSTRUCT_DIR:-/workspace/.hf_home/LLaVA-Instruct}"
JSON="${JSON:-${INSTRUCT_DIR}/llava_v1_5_mix665k.json}"
IMAGES="${IMAGES:-${INSTRUCT_DIR}/images}"
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
CACHE_DIR="${CACHE_DIR:-/workspace/.hf_home}"

STEPS="${STEPS:-5200}"          # ceil(665000 / 128) ≈ 1 epoch
LOCAL_BS="${LOCAL_BS:-16}"      # 16 × 8 = 128 effective batch
GLOBAL_BS="${GLOBAL_BS:-128}"
SEQ_LEN="${SEQ_LEN:-580}"       # 196 vision + 384 text — must match LlavaInstructSFTDataset default
LR="${LR:-2e-5}"                # LLaVA-1.5 paper SFT LR
PROJ_LR_MULT="${PROJ_LR_MULT:-1.0}"
MAX_NORM="${MAX_NORM:-1.0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/stage2_instruct_sft_447m}"
NGPU="${NGPU:-8}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-500}"
KEEP_K="${KEEP_K:-2}"           # torchtitan requires k>=2 (see launch_stage1.sh comment)

# Pre-flight validations
if [[ ! -d "${STAGE1_CKPT}" ]]; then
    echo "ERROR: stage 1 ckpt missing: ${STAGE1_CKPT}" >&2
    echo "Hint: run launch_stage1.sh first." >&2
    exit 1
fi
if [[ ! -f "${JSON}" ]]; then
    echo "ERROR: mix665k JSON missing: ${JSON}" >&2
    echo "Hint: run download_instruct_665k.sh first." >&2
    exit 1
fi
if [[ ! -d "${IMAGES}/coco/train2017" ]]; then
    echo "ERROR: COCO train2017 images missing under ${IMAGES}/coco/train2017/" >&2
    echo "Hint: re-run download_instruct_665k.sh (resumable)." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
exec /usr/local/bin/torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_vlm_multimodal_sft.train_mm \
    --mm.json "${JSON}" \
    --mm.images "${IMAGES}" \
    --mm.vision-model "${VISION}" \
    --mm.tokenizer "${TOKENIZER}" \
    --mm.cache-dir "${CACHE_DIR}" \
    --mm.proj-lr-mult "${PROJ_LR_MULT}" \
    --mm.global-seq-len "${SEQ_LEN}" \
    --mm.layout sft \
    --mm.val-samples 0 \
    --mm.val-freq 0 \
    --module kimi_linear --config "${STUDENT_CONFIG}" \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --training.max_norm "${MAX_NORM}" \
    --parallelism.data_parallel_shard_degree "${NGPU}" \
    --optimizer.lr "${LR}" \
    --lr_scheduler.warmup_steps 156 \
    --lr_scheduler.decay_ratio 0.2 \
    --lr_scheduler.min_lr_factor 0.1 \
    --checkpoint.enable \
    --checkpoint.interval "${SAVE_FREQ}" \
    --checkpoint.keep_latest_k "${KEEP_K}" \
    --checkpoint.initial_load_path "${STAGE1_CKPT}" \
    --checkpoint.initial_load_model_only \
    --metrics.log_freq "${LOG_FREQ}" \
    --metrics.save_tb_folder tb \
    --dump_folder "${OUT_DIR}"
