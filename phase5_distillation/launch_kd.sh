#!/usr/bin/env bash
# Phase 5 KD training launcher. Distills Kimi-Linear-48B-A3B-Base
# into the 436M student initialized from the Phase 4 step-12500 ckpt.
#
# Defaults are tuned for a single-node 4× RTX 5090 box. Override via
# environment variables:
#
#   STUDENT_CONFIG  flavor name (default: kimi_linear_436m_block_attn_res_n4)
#   STUDENT_CKPT    path to step-N ckpt directory
#   TEACHER         HF repo id (default: moonshotai/Kimi-Linear-48B-A3B-Base)
#   STEPS           total KD steps
#   LOCAL_BS        per-rank micro-batch
#   GLOBAL_BS       global batch (drives grad accum: G/(L*W) accum steps)
#   SEQ_LEN         context length
#   LR              constant LR for the distillation phase
#   ALPHA           KD CE weight (rest goes to KL)
#   T               KD temperature
#   OUT_DIR         output dir for logs + ckpts
#   NGPU            torchrun nproc_per_node

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_436m_block_attn_res_n4}"
STUDENT_CKPT="${STUDENT_CKPT:-${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500}"
TEACHER="${TEACHER:-moonshotai/Kimi-Linear-48B-A3B-Base}"
STEPS="${STEPS:-5000}"
LOCAL_BS="${LOCAL_BS:-2}"
GLOBAL_BS="${GLOBAL_BS:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-2e-4}"
ALPHA="${ALPHA:-0.3}"
T="${T:-2.0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/kd_run_overnight}"
NGPU="${NGPU:-4}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-500}"

if [[ ! -d "${STUDENT_CKPT}" ]]; then
    echo "ERROR: student ckpt dir not found: ${STUDENT_CKPT}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${WORKSPACE_DIR}"

PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/torchtitan${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_distillation.train_kd \
    --student-config "${STUDENT_CONFIG}" \
    --student-ckpt "${STUDENT_CKPT}" \
    --teacher "${TEACHER}" \
    --output-dir "${OUT_DIR}" \
    --steps "${STEPS}" \
    --local-bs "${LOCAL_BS}" \
    --global-bs "${GLOBAL_BS}" \
    --seq-len "${SEQ_LEN}" \
    --lr "${LR}" \
    --kd-alpha "${ALPHA}" \
    --kd-temperature "${T}" \
    --log-freq "${LOG_FREQ}" \
    --save-freq "${SAVE_FREQ}" \
    2>&1 | tee "${OUT_DIR}/train.log"
