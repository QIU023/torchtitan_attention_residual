#!/usr/bin/env bash
# Overnight pipeline: stage 0 (in-flight) → stage 1 → stage 2 → DCP→HF → GRPO.
#
# Sequence:
#   1. Wait for Instruct-665K download to complete.
#   2. Gracefully stop stage 0 (SIGTERM after current ckpt write).
#   3. Run launch_stage1.sh against latest stage 0 ckpt.
#   4. Run launch_stage2.sh against final stage 1 ckpt.
#   5. Convert final stage 2 DCP → HF safetensors (multimodal).
#   6. Launch GRPO multimodal smoke run.
#
# State file written at each transition so the script is resumable:
#   ${STATE_FILE} contains the highest completed stage tag.
#
# Disk safety: aborts if free < 40G at any stage boundary.
# Time safety: bails after ${DEADLINE_HOURS} hours total wall clock.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/overnight_pipeline_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
exec >>"${LOG_DIR}/orchestrator.log" 2>&1

STATE_FILE="${STATE_FILE:-${LOG_DIR}/state}"
DEADLINE_HOURS="${DEADLINE_HOURS:-12}"
DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

DOWNLOAD_LOG="/workspace/.hf_home/LLaVA-Instruct/download.log"
STAGE0_OUT="${WORKSPACE_DIR}/phase4_kimi_attnres_lm_pretrain/runs/lm_447m_fp8_paperalign_C"
STAGE1_OUT="${SCRIPT_DIR}/runs/stage1_alignment_447m"
STAGE2_OUT="${SCRIPT_DIR}/runs/stage2_instruct_sft_447m"
HF_OUT="${WORKSPACE_DIR}/phase11_rlhf_grpo_infra/hf/stage2_overnight"

CONFIG_NAME="${CONFIG_NAME:-kimi_linear_447m_aligned_block_attn_res_n4_fp8}"

# ---- helpers ----
free_gb() { df -BG /workspace | awk 'NR==2{gsub("G","",$4);print $4}'; }
now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(now)] $*"; }

state_set() { echo "$1" > "${STATE_FILE}"; log "STATE → $1"; }
state_get() { [[ -f "${STATE_FILE}" ]] && cat "${STATE_FILE}" || echo "init"; }

CHECK_DISK_THRESHOLD="${CHECK_DISK_THRESHOLD:-25}"
check_disk() {
    local req="${1:-${CHECK_DISK_THRESHOLD}}"
    local free=$(free_gb)
    if (( free < req )); then
        log "ABORT: disk free=${free}G < required=${req}G"
        state_set "ABORTED_DISK"
        exit 1
    fi
}

# Clean older ckpt dirs in a tree, keep only the newest one.
trim_ckpts_to_latest() {
    local dir="$1"
    [[ ! -d "${dir}/checkpoint" ]] && return 0
    local latest=$(ls -d "${dir}/checkpoint/step-"* 2>/dev/null | sort -V | tail -1)
    [[ -z "${latest}" ]] && return 0
    log "trim ckpts under ${dir}: keep $(basename "${latest}")"
    for d in $(ls -d "${dir}/checkpoint/step-"* 2>/dev/null); do
        if [[ "${d}" != "${latest}" ]]; then
            log "  removing $(basename "${d}") ($(du -sh "${d}" | awk '{print $1}'))"
            rm -rf "${d}"
        fi
    done
}

check_deadline() {
    if (( $(date +%s) > DEADLINE_EPOCH )); then
        log "ABORT: deadline of ${DEADLINE_HOURS}h exceeded"
        state_set "ABORTED_DEADLINE"
        exit 2
    fi
}

latest_ckpt() {
    local dir="$1"
    ls -d "${dir}/checkpoint/step-"* 2>/dev/null | sort -V | tail -1
}

# ---- step 1: wait for download ----
wait_for_download() {
    log "step 1: wait for Instruct-665K download to complete"
    while true; do
        check_deadline
        if grep -q "ALL DONE" "${DOWNLOAD_LOG}" 2>/dev/null; then
            log "download complete"
            break
        fi
        if grep -q "ABORT:" "${DOWNLOAD_LOG}" 2>/dev/null; then
            log "ABORT: download script failed; see ${DOWNLOAD_LOG}"
            state_set "ABORTED_DOWNLOAD"
            exit 3
        fi
        local last=$(tail -1 "${DOWNLOAD_LOG}" 2>/dev/null | head -c 80)
        log "still downloading… (last: ${last}) disk=$(free_gb)G"
        sleep 120
    done
    state_set "download_done"
}

# ---- step 2: stop stage 0 ----
stop_stage0() {
    log "step 2: stop stage 0 gracefully"
    local pid=$(ps -ef | grep '/usr/local/bin/torchrun' | grep -v grep \
                | grep paperalign_C | head -1 | awk '{print $2}' || true)
    if [[ -z "${pid:-}" ]]; then
        log "stage 0 already stopped"
        state_set "stage0_stopped"
        return 0
    fi
    log "stage 0 PID=${pid}; sending SIGTERM and waiting for clean shutdown"
    kill -TERM "${pid}" 2>/dev/null || true
    # Wait up to 5 min for process tree to exit
    local waited=0
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 5
        waited=$((waited + 5))
        if (( waited > 300 )); then
            log "WARN: SIGTERM timeout, sending SIGKILL"
            kill -KILL "${pid}" 2>/dev/null || true
            sleep 10
            break
        fi
    done
    sleep 30  # let any in-flight ckpt write finalize
    log "stage 0 stopped. latest ckpt: $(latest_ckpt "${STAGE0_OUT}")"
    state_set "stage0_stopped"
}

# ---- step 3: stage 1 alignment ----
# torchtitan + fla has a known KDA Triton crash that hits every ~2500 steps
# (task #46). Each retry resumes from latest ckpt; we cap attempts so a real
# bug doesn't infinite-loop, but a high cap absorbs the expected KDA cycles.
STAGE1_MAX_ATTEMPTS="${STAGE1_MAX_ATTEMPTS:-15}"
STAGE2_MAX_ATTEMPTS="${STAGE2_MAX_ATTEMPTS:-15}"
RETRY_GRACE_SECONDS="${RETRY_GRACE_SECONDS:-30}"

run_stage1() {
    check_deadline
    check_disk 40
    log "step 3: stage 1 alignment"
    local s0_ckpt=$(latest_ckpt "${STAGE0_OUT}")
    if [[ -z "${s0_ckpt}" ]]; then
        log "ABORT: no stage 0 ckpt"
        state_set "ABORTED_S1_INPUT"
        exit 4
    fi
    log "using stage 0 ckpt: ${s0_ckpt}"

    local attempt=0
    while (( attempt < STAGE1_MAX_ATTEMPTS )); do
        attempt=$((attempt + 1))
        check_deadline
        check_disk 18
        log "stage 1 attempt ${attempt}/${STAGE1_MAX_ATTEMPTS} (latest stage1 ckpt: $(latest_ckpt "${STAGE1_OUT}"))"
        STUDENT_CKPT="${s0_ckpt}" \
        STUDENT_CONFIG="${CONFIG_NAME}" \
        OUT_DIR="${STAGE1_OUT}" \
        bash "${SCRIPT_DIR}/launch_stage1.sh" > "${LOG_DIR}/stage1_attempt${attempt}.log" 2>&1
        local rc=$?
        if (( rc == 0 )); then
            log "stage 1 completed (attempt ${attempt})"
            log "stage 1 done. latest ckpt: $(latest_ckpt "${STAGE1_OUT}")"
            # NOTE: deliberately do NOT trim stage 0 ckpts here.
            # User reserves keep/delete decision for LM pretrain ckpts.
            state_set "stage1_done"
            return 0
        fi
        log "stage 1 attempt ${attempt} failed rc=${rc}; tail of log:"
        tail -30 "${LOG_DIR}/stage1_attempt${attempt}.log" | grep -E 'Error|error|RuntimeError|step:' | tail -10
        log "sleeping ${RETRY_GRACE_SECONDS}s before retry; torchtitan auto-resumes from latest ckpt"
        sleep "${RETRY_GRACE_SECONDS}"
    done
    log "ABORT: stage 1 exhausted ${STAGE1_MAX_ATTEMPTS} attempts"
    state_set "ABORTED_S1_RUN"
    exit 5
}

# ---- step 4: stage 2 SFT ----
run_stage2() {
    check_deadline
    check_disk 40
    log "step 4: stage 2 visual instruction tuning"
    local s1_ckpt=$(latest_ckpt "${STAGE1_OUT}")
    if [[ -z "${s1_ckpt}" ]]; then
        log "ABORT: no stage 1 ckpt"
        state_set "ABORTED_S2_INPUT"
        exit 6
    fi
    log "using stage 1 ckpt: ${s1_ckpt}"

    local attempt=0
    while (( attempt < STAGE2_MAX_ATTEMPTS )); do
        attempt=$((attempt + 1))
        check_deadline
        check_disk 18
        # Rotate train-split shuffle seed per attempt so a deterministic
        # data-driven crash (e.g. KDA assert at a fixed step caused by a
        # specific mix665k sample at that iteration index) does NOT recur
        # forever. Seed 0 = no shuffle; we start at 1 so even attempt 1
        # differs from the original deterministic order tested in v2.
        log "stage 2 attempt ${attempt}/${STAGE2_MAX_ATTEMPTS} (latest stage2 ckpt: $(latest_ckpt "${STAGE2_OUT}"), MM_SHUFFLE_SEED=${attempt})"
        STAGE1_CKPT="${s1_ckpt}" \
        STUDENT_CONFIG="${CONFIG_NAME}" \
        OUT_DIR="${STAGE2_OUT}" \
        MM_SHUFFLE_SEED="${attempt}" \
        SAVE_FREQ="${STAGE2_SAVE_FREQ:-200}" \
        bash "${SCRIPT_DIR}/launch_stage2.sh" > "${LOG_DIR}/stage2_attempt${attempt}.log" 2>&1
        local rc=$?
        if (( rc == 0 )); then
            log "stage 2 completed (attempt ${attempt})"
            log "stage 2 done. latest ckpt: $(latest_ckpt "${STAGE2_OUT}")"
            trim_ckpts_to_latest "${STAGE1_OUT}"
            log "disk after stage 1 trim: $(free_gb)G"
            state_set "stage2_done"
            return 0
        fi
        log "stage 2 attempt ${attempt} failed rc=${rc}; tail of log:"
        tail -30 "${LOG_DIR}/stage2_attempt${attempt}.log" | grep -E 'Error|error|RuntimeError|step:' | tail -10
        log "sleeping ${RETRY_GRACE_SECONDS}s before retry; torchtitan auto-resumes from latest ckpt"
        sleep "${RETRY_GRACE_SECONDS}"
    done
    log "ABORT: stage 2 exhausted ${STAGE2_MAX_ATTEMPTS} attempts"
    state_set "ABORTED_S2_RUN"
    exit 7
}

# ---- step 4.5: eval on final stage 2 ckpt (LLaVA benchmark suite, OCR skipped) ----
run_eval() {
    check_deadline
    check_disk 30
    log "step 4.5: eval final stage 2 ckpt on LLaVA benchmark suite"
    local s2_ckpt=$(latest_ckpt "${STAGE2_OUT}")
    if [[ -z "${s2_ckpt}" ]]; then
        log "WARN: no stage 2 ckpt for eval; skip"
        state_set "eval_skipped"
        return 0
    fi
    local eval_dir="${SCRIPT_DIR}/eval_benchmarks/runs/overnight_$(date +%Y%m%d-%H%M%S)_$(basename ${s2_ckpt})"
    STAGE2_CKPT="${s2_ckpt}" \
    RUN_DIR="${eval_dir}" \
    bash "${SCRIPT_DIR}/eval_benchmarks/run_all_evals.sh" \
        > "${LOG_DIR}/eval_overnight.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "WARN: eval rc=${rc} (partial results may still be useful); continuing pipeline"
    fi
    log "eval done. results: ${eval_dir}"
    state_set "eval_done"
}

# ---- step 5: DCP→HF ----
convert_to_hf() {
    check_deadline
    check_disk 30
    log "step 5: DCP→HF (multimodal)"
    local s2_ckpt=$(latest_ckpt "${STAGE2_OUT}")
    if [[ -z "${s2_ckpt}" ]]; then
        log "ABORT: no stage 2 ckpt"
        state_set "ABORTED_HF_INPUT"
        exit 8
    fi
    rm -rf "${HF_OUT}"
    mkdir -p "${HF_OUT}"
    cd "${WORKSPACE_DIR}"
    PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/torchtitan${PYTHONPATH:+:${PYTHONPATH}}" \
    /usr/local/bin/torchrun --nproc_per_node=1 \
        phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
        --in "${s2_ckpt}" \
        --out "${HF_OUT}" \
        --config "${CONFIG_NAME}" \
        --vision-tower google/siglip-base-patch16-224 \
        > "${LOG_DIR}/dcp_to_hf.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "ABORT: DCP→HF exited rc=${rc}; tail of log:"
        tail -50 "${LOG_DIR}/dcp_to_hf.log"
        state_set "ABORTED_HF_RUN"
        exit 9
    fi
    # Copy tokenizer files from existing hf_step3100
    cp -n "${SCRIPT_DIR}/runs/mm_sft_447m_full/hf_step3100/"{tokenizer.json,tokenizer_config.json,special_tokens_map.json,preprocessor_config.json,processor_config.json} "${HF_OUT}/" 2>/dev/null || true
    log "DCP→HF done. output: ${HF_OUT}"
    state_set "hf_done"
}

# ---- step 6: GRPO ----
launch_grpo() {
    check_deadline
    log "step 6: launch GRPO multimodal smoke"
    cd "${WORKSPACE_DIR}"
    PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/torchtitan${PYTHONPATH:+:${PYTHONPATH}}" \
    # Bumped 30 → ${GRPO_NUM_STEPS:-400} for overnight thorough run. Orchestrator
    # deadline (DEADLINE_HOURS) will kill if it exceeds budget.
    # Default GRPO task expects LLaVA-Pretrain caption JSON. We have mix665k
    # Instruct on disk (schema-compatible: image + conversations[from=gpt]).
    # First gpt response serves as the "gold caption" for the reward model.
    /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_caption.py \
        --model-path "${HF_OUT}" \
        --num-steps "${GRPO_NUM_STEPS:-400}" \
        --llava-json "${GRPO_JSON:-/workspace/.hf_home/LLaVA-Instruct/llava_v1_5_mix665k.json}" \
        --llava-images "${GRPO_IMAGES:-/workspace/.hf_home/LLaVA-Instruct/images}" \
        > "${LOG_DIR}/grpo.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "WARN: GRPO exited rc=${rc}; tail of log:"
        tail -80 "${LOG_DIR}/grpo.log"
        state_set "GRPO_FAIL_RC${rc}"
        return $rc
    fi
    log "GRPO smoke completed cleanly"
    state_set "ALL_DONE"
}

# ---- main ----
log "============================================================"
log "OVERNIGHT PIPELINE START  (deadline ${DEADLINE_HOURS}h)"
log "LOG_DIR=${LOG_DIR}"
log "STAGE0_OUT=${STAGE0_OUT}"
log "STAGE1_OUT=${STAGE1_OUT}"
log "STAGE2_OUT=${STAGE2_OUT}"
log "HF_OUT=${HF_OUT}"
log "============================================================"

state=$(state_get)
log "resume from state: ${state}"

[[ "${state}" == "init" ]] && wait_for_download && state=$(state_get) || true
[[ "${state}" == "download_done" ]] && stop_stage0 && state=$(state_get) || true
[[ "${state}" == "stage0_stopped" ]] && run_stage1 && state=$(state_get) || true
[[ "${state}" == "stage1_done" ]] && run_stage2 && state=$(state_get) || true
[[ "${state}" == "stage2_done" ]] && run_eval && state=$(state_get) || true
[[ "${state}" == "eval_done" || "${state}" == "eval_skipped" ]] && convert_to_hf && state=$(state_get) || true
[[ "${state}" == "hf_done" ]] && launch_grpo && state=$(state_get) || true

log "PIPELINE END — final state: $(state_get)"
log "============================================================"
