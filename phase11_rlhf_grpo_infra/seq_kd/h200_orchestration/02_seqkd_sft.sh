#!/usr/bin/env bash
# STAGE 2 — seq-KD student SFT on 2xH200 (FSDP=2).
# Init model-only from the SFT-5200 multimodal ckpt; train on the teacher-
# distilled mix665k (same images/questions, Qwen3-VL-rewritten answers).
# sm_90 (Hopper): upstream fla KDA kernels work; NO vendored_fla shadow.
set -uo pipefail
source /home/seqkd_overnight/lib.sh

DISTILLED="${DISTILLED:-${REPO}/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_full.json}"
INIT_CKPT="${INIT_CKPT:-${REPO}/phase5_vlm_multimodal_sft/runs/sft_5200_base/checkpoint/step-5200}"
OUT_DIR="${OUT_DIR:-${REPO}/phase5_vlm_multimodal_sft/runs/seqkd_sft_447m}"
SCRIPT_DIR="${REPO}/phase5_vlm_multimodal_sft"
LOG_DIR="${ROOT}/logs"
mkdir -p "${OUT_DIR}"

NGPU="${NGPU:-2}"
GLOBAL_BS="${GLOBAL_BS:-128}"
LOCAL_BS="${LOCAL_BS:-16}"          # accum = 128/(16*2) = 4
SEQ_LEN="${SEQ_LEN:-1024}"
TEXT_LEN="${TEXT_LEN:-828}"
LR="${LR:-2e-5}"
SAVE_FREQ="${SAVE_FREQ:-200}"
KEEP_K="${KEEP_K:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4}"

[[ -f "${DISTILLED}" ]] || { log "FATAL: distilled json missing ${DISTILLED}"; exit 2; }
ckpt_ok "${INIT_CKPT}" || { log "FATAL: init ckpt missing/invalid ${INIT_CKPT}"; exit 2; }

# 1 epoch over the distilled set, capped, with a step floor
NROWS=$("${CPY}" -c "import json;print(len(json.load(open('${DISTILLED}'))))")
EPOCH_STEPS=$(( (NROWS + GLOBAL_BS - 1) / GLOBAL_BS ))
STEPS="${STEPS:-$EPOCH_STEPS}"
CAP="${STEPS_CAP:-3000}"
(( STEPS > CAP )) && STEPS="${CAP}"
(( STEPS < 50 )) && STEPS=50
WARMUP_STEPS="${WARMUP_STEPS:-$(( STEPS / 20 + 1 ))}"
log "[sft] rows=${NROWS} epoch_steps=${EPOCH_STEPS} -> STEPS=${STEPS} warmup=${WARMUP_STEPS} lbs=${LOCAL_BS} gbs=${GLOBAL_BS} ngpu=${NGPU}"

latest() { ls -d "${OUT_DIR}/checkpoint/step-"* 2>/dev/null | sort -t- -k2 -n | tail -1; }

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
    attempt=$((attempt+1))
    # clean incomplete ckpts (crash-during-save: no .metadata)
    for c in "${OUT_DIR}"/checkpoint/step-*; do
        [[ -d "$c" && ! -f "$c/.metadata" ]] && { log "[sft] rm incomplete $(basename "$c")"; rm -rf "$c"; }
    done
    LC=$(latest)
    log "[sft] attempt ${attempt}/${MAX_ATTEMPTS}: $([[ -z "$LC" ]] && echo "fresh model-only from SFT-5200" || echo "auto-resume ${LC}") seed=${attempt}"

    STUDENT_CONFIG="${STUDENT_CONFIG}" \
    STAGE1_CKPT="${LC:-${INIT_CKPT}}" \
    JSON="${DISTILLED}" \
    IMAGES=/home/.hf_home/LLaVA-Instruct/images \
    INSTRUCT_DIR=/home/.hf_home/LLaVA-Instruct \
    CACHE_DIR=/home/.hf_home \
    OUT_DIR="${OUT_DIR}" \
    NGPU="${NGPU}" GLOBAL_BS="${GLOBAL_BS}" LOCAL_BS="${LOCAL_BS}" \
    SEQ_LEN="${SEQ_LEN}" TEXT_LEN="${TEXT_LEN}" LR="${LR}" \
    STEPS="${STEPS}" WARMUP_STEPS="${WARMUP_STEPS}" \
    SAVE_FREQ="${SAVE_FREQ}" KEEP_K="${KEEP_K}" \
    MM_SHUFFLE_SEED="${attempt}" \
    bash "${SCRIPT_DIR}/launch_stage2.sh" > "${LOG_DIR}/seqkd_sft_attempt${attempt}.log" 2>&1
    rc=$?

    if (( rc == 0 )) && grep -qE "step:\s*${STEPS}\b" "${LOG_DIR}/seqkd_sft_attempt${attempt}.log" 2>/dev/null; then
        log "[sft] COMPLETE attempt=${attempt} latest=$(latest)"; echo DONE > "${OUT_DIR}/STATUS"; exit 0
    fi
    laststep=$(grep -oE "step:\s*[0-9]+" "${LOG_DIR}/seqkd_sft_attempt${attempt}.log" 2>/dev/null | tail -1)
    log "[sft] attempt ${attempt} rc=${rc} (${laststep:-no-steps}); retrying"
    grep -E "device-side assert|RuntimeError|out of memory|Error|Traceback" "${LOG_DIR}/seqkd_sft_attempt${attempt}.log" 2>/dev/null | tail -3
    pkill -9 -f '[t]rain_mm' 2>/dev/null; pkill -9 -f '[t]orchrun' 2>/dev/null
    sleep 15
done
log "[sft] EXHAUSTED attempts; latest=$(latest)"; echo INCOMPLETE > "${OUT_DIR}/STATUS"; exit 1
