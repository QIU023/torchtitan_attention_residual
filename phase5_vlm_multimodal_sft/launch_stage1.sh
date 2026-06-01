#!/usr/bin/env bash
# Phase 5 Stage 1 — LLaVA-1.5 paper recipe: ALIGNMENT.
#
# Trains the MLP projector ONLY (LM + vision tower both frozen). The
# projector learns to map SigLIP vision tokens into the Kimi-Linear LM's
# embedding space using LLaVA-Pretrain 558K caption pairs.
#
# Differences from launch_train.sh (which is joint single-stage):
#   - --mm.freeze-lm                       freeze 447M LM params
#   - LR=1e-3  PROJ_LR_MULT=1              LLaVA-1.5 paper LR for projector
#   - GBS=256  STEPS=2180                  1 epoch of 558K @ gbs256
#   - warmup=66 (~3% of total steps)       paper match
#   - NGPU=8                               full node
#
# Expected wall clock: ~30-45 min on 8×5090 (projector-only, cheap).
#
# Output: runs/stage1_alignment_447m/
#   checkpoint/step-NNNN/   ← projector state + frozen LM (unchanged)
#   tb/                     ← tensorboard for loss curve
#
# Resume path: pass the final stage-1 ckpt as --checkpoint.initial_load_path
# in launch_stage2.sh (which will UNfreeze LM and do full instruct SFT).

set -euo pipefail

# Disable core dumps in child processes. NCCL watchdog drops 8-12GB cores
# per rank on KDA crashes (task #46). With auto-retry, 8 ranks × 4 crashes
# accumulates 100GB+ of cores we never inspect — they fill the disk and
# trigger the panic watchdog. Disable at launcher level; gdb users can
# override with `ulimit -c unlimited` before invoking the script.
ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

# ---- knobs ----
STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4_fp8}"
# After stage 0 completes (step 12750) point this at the final ckpt;
# until then use whatever ckpt is freshest.
STUDENT_CKPT="${STUDENT_CKPT:-${WORKSPACE_DIR}/phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_C/checkpoint/step-12700}"
DATA_DIR="${DATA_DIR:-/workspace/.hf_home/LLaVA-Pretrain}"
JSON="${JSON:-${DATA_DIR}/blip_laion_cc_sbu_558k.json}"
IMAGES="${IMAGES:-${DATA_DIR}}"
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
CACHE_DIR="${CACHE_DIR:-/workspace/.hf_home}"

STEPS="${STEPS:-8720}"          # ceil(558000 / 64) ≈ 1 epoch @ smaller gbs
LOCAL_BS="${LOCAL_BS:-8}"       # 8 × 8 = 64 effective batch — OOM-safe on 5090 32GB
GLOBAL_BS="${GLOBAL_BS:-64}"
SEQ_LEN="${SEQ_LEN:-260}"       # 196 vision + ~60 text + bos/eos
WARMUP_STEPS="${WARMUP_STEPS:-260}"  # ~3% of 8720
LR="${LR:-1e-3}"                # LLaVA-1.5 paper projector LR; LM frozen so doesn't matter
PROJ_LR_MULT="${PROJ_LR_MULT:-1.0}"
MAX_NORM="${MAX_NORM:-1.0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/stage1_alignment_447m}"
NGPU="${NGPU:-8}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-500}"
KEEP_K="${KEEP_K:-2}"           # torchtitan requires k>=2 (refuses k=1 at init,
                                # see torchtitan/components/checkpoint.py:378).
                                # Orchestrator trims older stage1 dir after
                                # stage2 starts, so peak disk stays bounded.
VAL_FREQ="${VAL_FREQ:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-512}"
VAL_BATCHES="${VAL_BATCHES:-16}"

if [[ ! -d "${STUDENT_CKPT}" ]]; then
    echo "ERROR: student ckpt not found: ${STUDENT_CKPT}" >&2
    echo "Hint: stage 0 must finish (or be at a usable step) first." >&2
    exit 1
fi
if [[ ! -f "${JSON}" ]]; then
    echo "ERROR: caption json not found: ${JSON}" >&2; exit 1
fi
if [[ ! -d "${IMAGES}" ]]; then
    echo "ERROR: images dir not found: ${IMAGES}" >&2; exit 1
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
    --mm.val-samples "${VAL_SAMPLES}" \
    --mm.val-freq "${VAL_FREQ}" \
    --mm.val-batches "${VAL_BATCHES}" \
    --mm.freeze-lm \
    --module attention_residual --config "${STUDENT_CONFIG}" \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --training.max_norm "${MAX_NORM}" \
    --parallelism.data_parallel_shard_degree "${NGPU}" \
    --optimizer.lr "${LR}" \
    --lr_scheduler.warmup_steps "${WARMUP_STEPS}" \
    --activation_checkpoint.mode full \
    --lr_scheduler.decay_ratio 0.0 \
    --lr_scheduler.min_lr_factor 1.0 \
    --checkpoint.enable \
    --checkpoint.interval "${SAVE_FREQ}" \
    --checkpoint.keep_latest_k "${KEEP_K}" \
    --checkpoint.initial_load_path "${STUDENT_CKPT}" \
    --checkpoint.initial_load_model_only \
    --metrics.log_freq "${LOG_FREQ}" \
    --metrics.save_tb_folder tb \
    --dump_folder "${OUT_DIR}"
