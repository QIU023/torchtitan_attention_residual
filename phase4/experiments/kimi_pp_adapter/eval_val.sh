#!/usr/bin/env bash
# Evaluate Problem A + Problem B step-12500 ckpts on c4_validation.
#
# Strategy: re-launch the same trainer with training.steps = ckpt_step + 1
# and validation.enable on. Trainer auto-resumes from step-12500, runs 1
# step at the min-LR floor (lr_scheduler.total_steps=12500 pins the
# original cosine schedule, so resume LR ≈ min_lr_factor × peak ≈ 2e-4),
# then runs validation. The 1-step LR=2e-4 update is negligible vs the
# val loss differences we care about (O(1e-3) perturbation << O(0.05) signal).
#
# Each arm appends to its own train.log; we grep "val_loss" from the
# tail to read out the result.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE4_DIR="${SCRIPT_DIR}/../.."
RUNS_DIR="${PHASE4_DIR}/runs"

VAL_STEPS="${VAL_STEPS:-200}"  # number of val batches per arm

backup_log() {
    local out_dir="$1"
    [[ -f "${out_dir}/train.log" ]] || return 0
    [[ -f "${out_dir}/train.log.preeval" ]] && return 0  # already backed up
    cp "${out_dir}/train.log" "${out_dir}/train.log.preeval"
}

restore_log() {
    local out_dir="$1"
    [[ -f "${out_dir}/train.log.preeval" ]] || return 0
    # Save the eval-run log under eval.log, restore canonical train.log.
    if [[ -f "${out_dir}/train.log" ]]; then
        mv "${out_dir}/train.log" "${out_dir}/eval.log"
    fi
    mv "${out_dir}/train.log.preeval" "${out_dir}/train.log"
}

run_fsdp_eval() {
    local name="$1"
    local out_dir="$2"
    local config="$3"
    echo "[$(date -Is)] === eval ${name} on c4_validation ==="
    backup_log "${out_dir}"
    OUT_DIR="${out_dir}" \
    MODULE=kimi_linear \
    CONFIG="${config}" \
    NGPU=4 \
    STEPS=12501 \
    LOCAL_BS=3 \
    GLOBAL_BS=12 \
    SEQ_LEN=2048 \
    COMPILE=1 \
    VAL=1 VAL_FREQ=1 VAL_STEPS="${VAL_STEPS}" \
    EXTRA_ARGS_APPEND="--lr_scheduler.total_steps 12500" \
    bash "${PHASE4_DIR}/launch_fsdp_small.sh"
    restore_log "${out_dir}"
    echo "[$(date -Is)] === ${name} done ==="
}

run_pp_eval() {
    local name="$1"
    local out_dir="$2"
    local config="$3"
    echo "[$(date -Is)] === eval ${name} on c4_validation (PP) ==="
    backup_log "${out_dir}"
    OUT_DIR="${out_dir}" \
    MODULE=kimi_linear \
    CONFIG="${config}" \
    NGPU=4 \
    STEPS=12501 \
    LOCAL_BS=1 \
    GLOBAL_BS=12 \
    SEQ_LEN=2048 \
    CACHE=1 \
    VAL=1 VAL_FREQ=1 VAL_STEPS="${VAL_STEPS}" \
    EXTRA_ARGS_APPEND="--lr_scheduler.total_steps 12500" \
    bash "${PHASE4_DIR}/launch_pp4_kimi.sh"
    restore_log "${out_dir}"
    echo "[$(date -Is)] === ${name} done ==="
}

# Problem A: baseline FSDP (dense)
run_fsdp_eval baseline_fsdp \
    "${RUNS_DIR}/kimi_436m_baseline_fsdp_overnight" \
    kimi_linear_436m_baseline

# Problem A: AttnRes FSDP N=4
run_fsdp_eval attnres_fsdp_n4 \
    "${RUNS_DIR}/kimi_436m_block_attn_res_fsdp_overnight" \
    kimi_linear_436m_block_attn_res_n4

# Problem B: Adapter PP N=8
run_pp_eval adapter_pp_n8 \
    "${RUNS_DIR}/kimi_pp_adapter_bench/adapter_pp" \
    kimi_linear_436m_block_attn_res

# Summarize — read from eval.log (the eval run's output)
echo ""
echo "=== Validation summary ==="
for name in baseline_fsdp attnres_fsdp_n4 adapter_pp_n8; do
    case "${name}" in
        baseline_fsdp)    log="${RUNS_DIR}/kimi_436m_baseline_fsdp_overnight/eval.log" ;;
        attnres_fsdp_n4)  log="${RUNS_DIR}/kimi_436m_block_attn_res_fsdp_overnight/eval.log" ;;
        adapter_pp_n8)    log="${RUNS_DIR}/kimi_pp_adapter_bench/adapter_pp/eval.log" ;;
    esac
    if [[ -f "${log}" ]]; then
        val_line=$(sed -E 's/\x1b\[[0-9;]*m//g' "${log}" | grep -E "validate step:" | tail -1)
        echo "${name}: ${val_line:-(no validate line found in eval.log)}"
    else
        echo "${name}: (eval.log missing at ${log})"
    fi
done
