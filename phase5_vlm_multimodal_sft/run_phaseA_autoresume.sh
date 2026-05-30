#!/usr/bin/env bash
# Phase A (mix665k full SFT) with KDA-crash auto-resume.
#
# The fla KDA Triton kernel throws a deterministic device-side assert at
# data-dependent iterations (task #46/#74). Single-shot launch dies
# permanently; this wrapper retries, and on each retry:
#   1. torchtitan auto-resumes from the latest ckpt in OUT_DIR/checkpoint/
#      (it scans the dump folder on startup — no flag needed)
#   2. MM_SHUFFLE_SEED rotates so the post-resume data order differs,
#      dodging the specific batch that triggered the assert
#
# SAVE_FREQ=200 (not 1300) so a crash loses ≤200 steps. keep_latest_k=2.
#
# Init: first attempt loads stage2 step-5200 weights (--initial-load-model-only);
# subsequent attempts auto-resume from Phase-A's own latest ckpt (full state).
#
# Usage:  bash run_phaseA_autoresume.sh
# Env:    MAX_ATTEMPTS (default 30), DEADLINE_HOURS (default 14)
set -uo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

SCRIPT_DIR="phase5_vlm_multimodal_sft"
OUT_DIR="$(pwd)/${SCRIPT_DIR}/runs/phaseA_mix665k_full"
INIT_CKPT="$(ls -d $(pwd)/${SCRIPT_DIR}/runs/phaseA_mix665k_full/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
LOG_DIR="${OUT_DIR}/autoresume_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
DEADLINE_HOURS="${DEADLINE_HOURS:-14}"
START_TS=$(date +%s)
RETRY_GRACE=20

log() { echo "[$(date '+%H:%M:%S')] $*"; }

latest_ckpt() { ls -d "${OUT_DIR}/checkpoint/step-"* 2>/dev/null | sort -t- -k2 -n | tail -1; }

# Disk watchdog
(
    while true; do
        sleep 120
        F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
        if (( F < 8 )); then
            echo "[watchdog] PANIC disk ${F}G; killing train"
            pkill -9 -f train_mm 2>/dev/null; pkill -9 -f torchrun 2>/dev/null
            touch "${OUT_DIR}/DISK_PANIC"; exit 1
        fi
    done
) &
WD=$!
trap 'kill -9 ${WD} 2>/dev/null' EXIT

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt + 1))

    # deadline check
    elapsed_h=$(( ($(date +%s) - START_TS) / 3600 ))
    if (( elapsed_h >= DEADLINE_HOURS )); then
        log "DEADLINE ${DEADLINE_HOURS}h reached; stopping. latest=$(latest_ckpt)"
        break
    fi

    # disk preflight
    F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
    if (( F < 10 )); then
        log "ABORT: disk ${F}G < 15G preflight"
        break
    fi

    LC=$(latest_ckpt)
    if [[ -z "${LC}" ]]; then
        # first attempt — init-load from stage2 step-5200
        log "attempt ${attempt}/${MAX_ATTEMPTS}: fresh start, init-load step-5200, seed=${attempt}"
        STAGE1_CKPT="${INIT_CKPT}"
    else
        # resume — torchtitan auto-loads latest from OUT_DIR; STAGE1_CKPT
        # still points at init (ignored once a resume ckpt exists, but
        # launch_stage2 requires the path to exist for its preflight).
        log "attempt ${attempt}/${MAX_ATTEMPTS}: resume from ${LC}, seed=${attempt}"
        STAGE1_CKPT="${INIT_CKPT}"
    fi

    STAGE1_CKPT="${STAGE1_CKPT}" \
    OUT_DIR="${OUT_DIR}" \
    SAVE_FREQ="${SAVE_FREQ:-200}" \
    KEEP_K=2 \
    MM_SHUFFLE_SEED="${attempt}" \
    bash "${SCRIPT_DIR}/launch_stage2.sh" \
        > "${LOG_DIR}/attempt${attempt}.log" 2>&1
    rc=$?

    if (( rc == 0 )) && grep -qE "step:\s*5200" "${LOG_DIR}/attempt${attempt}.log" 2>/dev/null; then
        log "✅ Phase A COMPLETE (attempt ${attempt}). latest=$(latest_ckpt)"
        echo "PHASE_A_DONE" > "${OUT_DIR}/STATUS"
        exit 0
    fi

    laststep=$(grep -oE "step:\s*[0-9]+" "${LOG_DIR}/attempt${attempt}.log" 2>/dev/null | tail -1)
    log "attempt ${attempt} failed rc=${rc} (${laststep:-no steps}); retry in ${RETRY_GRACE}s"
    grep -E "device-side assert|RuntimeError|out of memory|Error" "${LOG_DIR}/attempt${attempt}.log" 2>/dev/null | tail -3
    pkill -9 -f train_mm 2>/dev/null; pkill -9 -f torchrun 2>/dev/null
    sleep "${RETRY_GRACE}"
done

log "EXHAUSTED ${attempt} attempts (or deadline). latest ckpt=$(latest_ckpt)"
echo "PHASE_A_INCOMPLETE" > "${OUT_DIR}/STATUS"
exit 1
