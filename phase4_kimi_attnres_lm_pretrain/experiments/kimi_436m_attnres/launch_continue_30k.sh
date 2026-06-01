#!/usr/bin/env bash
# Phase A closure continuation — resume both baseline and AttnRes
# runs from their step-12,500 checkpoints and push to step 30,000.
#
# Per docs/pretraining_closure_and_kd_plan.md the "minimum closure"
# target for the pretraining-behavior-validation plank is 30K steps
# (~0.74B tokens). This script chains the two continuation runs
# sequentially on the 4× RTX 5090 box — one arm at a time.
#
# torchtitan auto-resumes from the latest step in
# {OUT_DIR}/checkpoint/, so we just point at the original run dirs
# and bump --training.steps to 30000. The original train.log is
# preserved as train.log.part_to_N before re-launch.
#
# LR-SCHEDULE GOTCHA (why lr_scheduler.total_steps=12500 below):
#   torchtitan only persists `last_epoch` in the scheduler state;
#   the LR-lambda itself is rebuilt from config at launch. If we
#   let total_steps default to the new training.steps=30000, the
#   schedule is recomputed with warmup=500, decay_ratio=0.8 over
#   30K, and the LR at the resume point (step 12,500) lands at
#   ~1.7e-3 — ~8× higher than the ~2.0e-4 the model ended at under
#   the original 12,500-step schedule. That is a hot-restart with
#   a stale optimizer state; we'd see a loss spike that takes
#   1K+ steps to recover.
#
#   Pinning --lr_scheduler.total_steps 12500 keeps the original
#   cosine curve and leaves the continuation running at the min LR
#   (min_lr_factor=0.1 × peak = ~2.0e-4) from step 12,500 → 30,000.
#   That is the standard continued-pretraining recipe — low-LR
#   extension on an already-decayed schedule.
#
# Usage:
#   bash launch_continue_30k.sh                # both arms, sequential
#   ARM=baseline bash launch_continue_30k.sh   # baseline only
#   ARM=attnres  bash launch_continue_30k.sh   # attnres only
#   STEPS=60000 bash launch_continue_30k.sh    # push to ideal closure
#   LR_TOTAL_STEPS=30000 bash launch_continue_30k.sh  # full fresh schedule (accept LR jump)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."

ARM="${ARM:-both}"
STEPS="${STEPS:-30000}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-12500}"  # preserve original schedule — see header
# Validation: on by default for the continuation. C4 validation set,
# run every 2500 steps x 100 val batches — so each check reads ~200K
# tokens, costs ~60s on 4x 5090. Adds ~7 samples to the 30K run.
VAL="${VAL:-1}"
VAL_FREQ="${VAL_FREQ:-2500}"
VAL_STEPS="${VAL_STEPS:-100}"

BASELINE_DIR="${PHASE4_DIR}/runs/kimi_436m_baseline_fsdp_overnight"
ATTNRES_DIR="${PHASE4_DIR}/runs/kimi_436m_block_attn_res_fsdp_overnight"

run_arm() {
    local name="$1"
    local out_dir="$2"
    local config="$3"

    local ckpt_dir="${out_dir}/checkpoint"
    if [[ ! -d "${ckpt_dir}" ]] || ! ls "${ckpt_dir}"/step-* >/dev/null 2>&1; then
        echo "[$(date -Is)] ERROR ${name}: no checkpoint found under ${ckpt_dir}" >&2
        return 1
    fi

    local latest_step
    latest_step=$(ls -d "${ckpt_dir}"/step-* | sed 's|.*step-||' | sort -n | tail -1)
    echo "[$(date -Is)] === ${name}: resuming from step ${latest_step}, target ${STEPS} ==="

    # Preserve prior log so the 0→12500 curve isn't clobbered by tee.
    if [[ -f "${out_dir}/train.log" ]]; then
        local rotated="${out_dir}/train.log.part_to_${latest_step}"
        if [[ ! -f "${rotated}" ]]; then
            cp "${out_dir}/train.log" "${rotated}"
            echo "[$(date -Is)]      preserved prior log -> ${rotated}"
        else
            echo "[$(date -Is)]      prior rotated log ${rotated} already exists, skipping copy"
        fi
    fi

    OUT_DIR="${out_dir}" \
    MODULE=attention_residual \
    CONFIG="${config}" \
    NGPU=4 \
    STEPS="${STEPS}" \
    LOCAL_BS=3 \
    GLOBAL_BS=12 \
    SEQ_LEN=2048 \
    COMPILE=1 \
    VAL="${VAL}" VAL_FREQ="${VAL_FREQ}" VAL_STEPS="${VAL_STEPS}" \
    EXTRA_ARGS_APPEND="--lr_scheduler.total_steps ${LR_TOTAL_STEPS}" \
    bash "${PHASE4_DIR}/launch_fsdp_small.sh"

    echo "[$(date -Is)] === ${name}: continuation exited rc=$? ==="
}

case "${ARM}" in
    baseline)
        run_arm baseline "${BASELINE_DIR}" kimi_linear_436m_baseline
        ;;
    attnres)
        run_arm attnres "${ATTNRES_DIR}" kimi_linear_436m_block_attn_res_n4
        ;;
    both)
        run_arm baseline "${BASELINE_DIR}" kimi_linear_436m_baseline
        sleep 30  # gpu drain
        run_arm attnres "${ATTNRES_DIR}" kimi_linear_436m_block_attn_res_n4
        ;;
    *)
        echo "Unknown ARM=${ARM}. Use baseline | attnres | both." >&2
        exit 1
        ;;
esac
