#!/usr/bin/env bash
# Phase 4 resume from-scratch run with STRETCHED cosine schedule.
#
# Why this exists:
# launch_from_scratch_paperhparams.sh set STEPS=12500 (matched original
# Phase 4 step count), but original Phase 4 had no grad_accum. This run
# uses grad_accum=8 (effective bs=96), so 12500 steps = 2.46B tokens —
# only 27% of chinchilla-optimal (~9B for 436M). The cosine decay was
# being applied to a budget that was way too short, slowing late-run
# descent prematurely.
#
# The fix: full-ckpt resume from the latest checkpoint, with
# total_steps=45000 (~9B tokens), so cosine "stretches". The scheduler
# state restores last_epoch (current step counter) but recomputes LR
# from the new total_steps. At step 8000 (resume start), LR jumps from
# the over-decayed ~5e-4 back to ~1.9e-3 (close to peak) because step
# 8000 is now early in a 45000-step cosine.
#
# User can stop manually at any val checkpoint (every 1000 steps).
# Recommended stops: val<=3.0 (target) or whenever satisfied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resume from the latest ckpt of the existing run
INIT_CKPT="${INIT_CKPT:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_paperhparams/checkpoint/step-8000}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/kimi_436m_block_attn_res_fsdp_paperhparams}"

if [[ ! -d "${INIT_CKPT}" ]]; then
    echo "ERROR: init ckpt not found: ${INIT_CKPT}" >&2
    echo "Available ckpts:" >&2
    ls "${OUT_DIR}/checkpoint/" 2>&1 >&2 || true
    exit 1
fi

# IMPORTANT: do NOT use --checkpoint.initial_load_model_only.
# We want full ckpt load (model + optimizer state + step counter)
# so Adam m/v are preserved (no fresh-Adam cold start) and the
# scheduler picks up at the right step in the new cosine curve.

MODULE="kimi_linear" \
CONFIG="kimi_linear_436m_block_attn_res_n4" \
NGPU="${NGPU:-4}" \
STEPS="${STEPS:-45000}" \
LOCAL_BS="${LOCAL_BS:-3}" \
GLOBAL_BS="${GLOBAL_BS:-96}" \
SEQ_LEN="${SEQ_LEN:-2048}" \
LR="" \
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
" \
bash "${SCRIPT_DIR}/launch_fsdp_small.sh"
