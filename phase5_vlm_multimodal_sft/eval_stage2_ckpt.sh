#!/usr/bin/env bash
# Run held-out validation on a specific stage 2 ckpt and exit.
#
# Reuses launch_stage2.sh's full stack (FSDP/projector/etc.) but with:
#   - training.steps set just past the loaded step, so the trainer runs
#     a single val pass on boot and exits
#   - eval-only validation freq + tail samples (no train)
#
# Usage:
#   STAGE2_CKPT=runs/stage2_instruct_sft_447m/checkpoint/step-3500 ./eval_stage2_ckpt.sh
#
# Output: val_loss line in the log; greppable as "mm: val_loss=...".

set -euo pipefail
ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

STAGE2_CKPT="${STAGE2_CKPT:?STAGE2_CKPT required (full path to a step-N dir)}"
if [[ ! -d "${STAGE2_CKPT}" ]]; then
    echo "ERROR: ckpt missing: ${STAGE2_CKPT}" >&2; exit 1
fi
# Extract step number from path "step-NNNN"
STEP_N=$(basename "${STAGE2_CKPT}" | sed 's/step-//')
TARGET_STEPS=$((STEP_N + 1))   # run one extra step to trigger val (val triggers at step % VAL_FREQ == 0 too; this is a fallback)

STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4_fp8}"
INSTRUCT_DIR="${INSTRUCT_DIR:-/workspace/.hf_home/LLaVA-Instruct}"
JSON="${JSON:-${INSTRUCT_DIR}/llava_v1_5_mix665k.json}"
IMAGES="${IMAGES:-${INSTRUCT_DIR}/images}"
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
CACHE_DIR="${CACHE_DIR:-/workspace/.hf_home}"

NGPU="${NGPU:-8}"
LOCAL_BS="${LOCAL_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-64}"
SEQ_LEN="${SEQ_LEN:-580}"
VAL_SAMPLES="${VAL_SAMPLES:-0}"
VAL_STRAT_PER_SOURCE="${VAL_STRAT_PER_SOURCE:-64}"  # ~5 sources × 64 = ~320 records
VAL_BATCHES="${VAL_BATCHES:-32}"   # more batches for a tighter eval

# Use a TEMP dump folder so we don't clobber the source ckpt dir's state file
EVAL_OUT_DIR="${EVAL_OUT_DIR:-/tmp/stage2_eval_$(basename ${STAGE2_CKPT})}"
# CRITICAL: do NOT pre-create checkpoint/ subdir. torchtitan's checkpointer
# silently ignores --checkpoint.initial_load_path when a checkpoint folder
# already exists (warning: "Checkpointer will use the checkpoints from the
# checkpoint.folder"). That makes the trainer start from random init, with
# loss ≈ log(vocab) = 12.0 instead of the loaded model's actual loss.
rm -rf "${EVAL_OUT_DIR}"
mkdir -p "${EVAL_OUT_DIR}"

echo "[$(date)] eval_stage2_ckpt: source ckpt = ${STAGE2_CKPT}"
echo "[$(date)] eval_stage2_ckpt: target_steps = ${TARGET_STEPS}, val_samples = ${VAL_SAMPLES}"

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
    --mm.proj-lr-mult 1.0 \
    --mm.global-seq-len "${SEQ_LEN}" \
    --mm.layout sft \
    --mm.val-samples "${VAL_SAMPLES}" \
    --mm.val-stratified-per-source "${VAL_STRAT_PER_SOURCE}" \
    --mm.val-freq 1 \
    --mm.val-batches "${VAL_BATCHES}" \
    --module kimi_linear --config "${STUDENT_CONFIG}" \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${TARGET_STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --training.max_norm 1.0 \
    --parallelism.data_parallel_shard_degree "${NGPU}" \
    --optimizer.lr 2e-5 \
    --lr_scheduler.warmup_steps 312 \
    --lr_scheduler.decay_ratio 0.2 \
    --lr_scheduler.min_lr_factor 0.1 \
    --checkpoint.enable \
    --checkpoint.interval 999999 \
    --checkpoint.keep_latest_k 2 \
    --checkpoint.initial_load_path "${STAGE2_CKPT}" \
    --checkpoint.initial_load_model_only \
    --metrics.log_freq 1 \
    --metrics.save_tb_folder tb \
    --dump_folder "${EVAL_OUT_DIR}"
