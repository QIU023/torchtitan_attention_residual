#!/usr/bin/env bash
# Phase 4: resume to break val 3.0 with smooth LR ramp.
#
# Why this exists:
# The original from-scratch run at STEPS=12500 lands at val ~3.04
# (cosine schedule shrinks LR to min 2.2e-4 by end → late descent stalls).
# To break 3.0 we need higher LR, but a discontinuous jump from 2.2e-4 to
# anywhere near peak risks oscillation (Adam moments calibrated for tiny
# updates suddenly seeing 3x bigger gradients).
#
# Trick: torchtitan's warmup formula is ``lr = min(1, step/warmup) × peak``.
# If we resume at step 12500 with warmup_steps=12700 (current_step + 200),
# step 12500 lands at lr = 12500/12700 × peak = 0.984 × peak — NO jump
# from 0, NO discontinuity. The 200-step warmup phase is just a gentle
# ramp from ~current_LR up to new peak, then cosine takes over.
#
# Conservative knob choices:
# * peak = 3e-4 = 1.36× the resume-start LR (2.2e-4). Modest enough that
#   Adam state can absorb the 1.36× update-magnitude shift over 200 steps.
# * warmup_steps = 12700 (current_step 12500 + 200 buffer)
# * decay_ratio = 0.8 cosine, min_lr_factor = 0.1 → final LR = 3e-5
# * total_steps = 18000 (5500 more steps from resume = ~23h compute)
#
# Expected trajectory:
#   step 12500: LR 2.95e-4, val 3.04 (start)
#   step 13000: LR 3.0e-4 (peak hit), val ~3.00
#   step 16000: LR 1.5e-4 (cosine middle), val ~2.91
#   step 18000: LR 3e-5 (cooldown), val ~2.88
#
# Stop manually at any val checkpoint that crosses 3.0 if you don't
# want the full 23h.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INIT_CKPT="${INIT_CKPT:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_paperhparams/checkpoint/step-12000}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_paperhparams}"

if [[ ! -d "${INIT_CKPT}" ]]; then
    echo "ERROR: init ckpt not found: ${INIT_CKPT}" >&2
    echo "Available ckpts:" >&2
    ls "${OUT_DIR}/checkpoint/" 2>&1 >&2 || true
    exit 1
fi

# Conservative tuning. To go more aggressive:
#   PEAK_LR=4e-4 WARMUP=13000  (peak 1.82× current, longer ramp)
#   PEAK_LR=5e-4 WARMUP=13500 STEPS=20000  (peak 2.27×, oscillation risk)
PEAK_LR="${PEAK_LR:-3e-4}"
WARMUP="${WARMUP:-12700}"
STEPS="${STEPS:-18000}"

MODULE="attention_residual" \
CONFIG="kimi_linear_436m_block_attn_res_n4" \
NGPU="${NGPU:-4}" \
STEPS="${STEPS}" \
LOCAL_BS="${LOCAL_BS:-3}" \
GLOBAL_BS="${GLOBAL_BS:-96}" \
SEQ_LEN="${SEQ_LEN:-2048}" \
LR="${PEAK_LR}" \
COMPILE="${COMPILE:-1}" \
VAL="${VAL:-1}" \
VAL_FREQ="${VAL_FREQ:-1000}" \
VAL_STEPS="${VAL_STEPS:-100}" \
OUT_DIR="${OUT_DIR}" \
EXTRA_ARGS_APPEND="\
--checkpoint.enable \
--checkpoint.initial_load_path ${INIT_CKPT} \
--checkpoint.interval ${SAVE_FREQ:-2000} \
--checkpoint.keep_latest_k ${KEEP_K:-2} \
--lr_scheduler.warmup_steps ${WARMUP} \
--lr_scheduler.decay_ratio 0.8 \
--lr_scheduler.min_lr_factor 0.1 \
" \
bash "${SCRIPT_DIR}/launch_fsdp_small.sh"
