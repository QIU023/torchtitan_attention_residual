#!/usr/bin/env bash
# STAGE eval — held-out val-loss on the distilled val split + qualitative
# caption/VQA generation, comparing a checkpoint against the SFT-5200 base.
# Consumes DCP directly (no HF conversion, no sglang). 2 GPUs.
set -uo pipefail
source /home/seqkd_overnight/lib.sh

CKPT="${1:?usage: 03_eval.sh <ckpt_dir> <tag>}"
TAG="${2:-eval}"
SCRIPT_DIR="${REPO}/phase5_vlm_multimodal_sft"
OUTLOG="${ROOT}/logs/eval_${TAG}.log"
RESULT="${ROOT}/eval_${TAG}.txt"

ckpt_ok "${CKPT}" || { log "[eval:${TAG}] FATAL bad ckpt ${CKPT}"; exit 2; }
log "[eval:${TAG}] ckpt=${CKPT}"

# --- held-out val loss (forward-only) via the repo's eval harness ---
STAGE2_CKPT="${CKPT}" \
STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4}" \
NGPU="${NGPU:-2}" \
VAL_SAMPLES="${VAL_SAMPLES:-1024}" VAL_BATCHES="${VAL_BATCHES:-32}" \
INSTRUCT_DIR=/home/.hf_home/LLaVA-Instruct \
CACHE_DIR=/home/.hf_home \
JSON="${JSON:-${REPO}/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_full.json}" \
bash "${SCRIPT_DIR}/eval_stage2_ckpt.sh" > "${OUTLOG}" 2>&1
rc=$?
VL=$(grep -aoE 'val_loss=[0-9.]+' "${OUTLOG}" | tail -1)
log "[eval:${TAG}] rc=${rc} ${VL:-no_val_loss}"
echo "ckpt=${CKPT} tag=${TAG} ${VL:-val_loss=NA} rc=${rc}" >> "${RESULT}"
exit 0
