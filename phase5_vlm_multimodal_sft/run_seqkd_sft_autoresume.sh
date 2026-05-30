#!/usr/bin/env bash
# S5: seq-KD student SFT — train 447M Kimi-AttnRes on the Qwen3-VL-30B-distilled
# mix665k (same images/questions, teacher-rewritten answers), with KDA-crash
# auto-resume (same deterministic fla Triton assert as phaseA, task #46/#74).
#
# Recipe (confirmed with user 2026-05-29):
#   data    = distilled_mix665k_full.json   (665298 convs, teacher answers)
#   init    = phaseA step-2200 (model-only; = stage2-equivalent base)
#   seq_len = 1536  (distilled p99=1355; < 2048 pretrain len, no re-pretrain)
#   lr      = 2e-5  (same-task / stronger-labels → learn fully, not just nudge)
#   steps   = 5200  (1 epoch @ gbs128), warmup 156, cosine decay last 20%
#   save    = every 200 steps (~23min @ ~7s/step), keep_latest_k=2
#
# On each retry MM_SHUFFLE_SEED rotates to dodge the specific batch that
# triggered the KDA assert. torchtitan auto-resumes the latest ckpt in OUT_DIR.
#
# Usage:  bash run_seqkd_sft_autoresume.sh        # full
#         SMOKE=1 bash run_seqkd_sft_autoresume.sh # 5-step no-ckpt smoke
set -uo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

SCRIPT_DIR="phase5_vlm_multimodal_sft"
# PR13 fla fix: put vendored_fla on PYTHONPATH so its sitecustomize.py shadows
# the buggy fla.modules.fused_norm_gate with our patched copy (no site-packages
# edit). launch_stage2.sh appends this to its own PYTHONPATH.
export PYTHONPATH="$(pwd)/${SCRIPT_DIR}/vendored_fla${PYTHONPATH:+:${PYTHONPATH}}"
OUT_DIR="$(pwd)/${SCRIPT_DIR}/runs/seqkd_sft_447m"
DISTILLED="$(pwd)/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_full.json"
# Init/preflight ckpt: prefer S5's own latest ckpt (so we don't depend on the
# phaseA step-2200 init ckpt once S5 has progressed — lets it be freed for disk).
# torchtitan auto-resumes OUT_DIR's latest anyway; STAGE1_CKPT only needs to be
# an existing dir for launch_stage2's preflight + the very-first model-only load.
_S5_LATEST="$(ls -d ${OUT_DIR}/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
PHASEA_CKPT="${_S5_LATEST:-$(pwd)/${SCRIPT_DIR}/runs/phaseA_mix665k_full/checkpoint/step-2200}"
LOG_DIR="${OUT_DIR}/autoresume_logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
DEADLINE_HOURS="${DEADLINE_HOURS:-14}"
SAVE_FREQ="${SAVE_FREQ:-200}"
SEQ_LEN="${SEQ_LEN:-1024}"     # 1536 triggered frequent fla causal_conv1d Triton
                                # device-asserts (every ~10-370 steps); 1024 is the
                                # proven-stable seq_len (stage2/phaseA). Truncates
                                # distilled p88+ long tail, but dataloader appends
                                # EOS so truncation is benign for SFT labels.
TEXT_LEN="${TEXT_LEN:-828}"    # 1024 - 196 vision
LR="${LR:-2e-5}"
START_TS=$(date +%s)
RETRY_GRACE=20

log() { echo "[$(date '+%H:%M:%S')] $*"; }
latest_ckpt() { ls -d "${OUT_DIR}/checkpoint/step-"* 2>/dev/null | sort -t- -k2 -n | tail -1; }

# preflight: distilled data + init ckpt present
[[ -f "${DISTILLED}" ]] || { log "FATAL: distilled json missing: ${DISTILLED}"; exit 1; }
[[ -d "${PHASEA_CKPT}" ]] || { log "FATAL: init ckpt missing: ${PHASEA_CKPT}"; exit 1; }

# ---- SMOKE: 5 steps, no ckpt ----
if [[ "${SMOKE:-0}" == "1" ]]; then
    log "SMOKE: 5 steps on distilled data, no ckpt"
    STAGE1_CKPT="${PHASEA_CKPT}" \
    JSON="${DISTILLED}" \
    OUT_DIR="${OUT_DIR}_smoke" \
    SEQ_LEN="${SEQ_LEN}" TEXT_LEN="${TEXT_LEN}" LR="${LR}" \
    STEPS=5 SAVE_FREQ=1000 KEEP_K=2 MM_SHUFFLE_SEED=0 \
    bash "${SCRIPT_DIR}/launch_stage2.sh" 2>&1 | tee "${LOG_DIR}/smoke.log"
    exit ${PIPESTATUS[0]}
fi

# ---- disk watchdog (10G floor; 73G free, peak transient leaves ~22G) ----
(
    while true; do
        sleep 120
        F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
        if (( F < 10 )); then
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
    elapsed_h=$(( ($(date +%s) - START_TS) / 3600 ))
    if (( elapsed_h >= DEADLINE_HOURS )); then
        log "DEADLINE ${DEADLINE_HOURS}h reached; stopping. latest=$(latest_ckpt)"; break
    fi
    F=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
    if (( F < 12 )); then log "ABORT: disk ${F}G < 12G preflight"; break; fi

    # Clean incomplete ckpts left by a crash-during-save (no .metadata). These
    # are NOT purged by torchtitan's keep_latest_k and pile up across the
    # frequent KDA-crash retries → disk panic. Remove them before each attempt.
    for _c in "${OUT_DIR}"/checkpoint/step-*; do
        [[ -d "$_c" ]] || continue
        if [[ ! -f "$_c/.metadata" ]]; then
            log "cleaning incomplete ckpt $(basename "$_c") (no .metadata)"
            rm -rf "$_c"
        fi
    done

    LC=$(latest_ckpt)
    # First attempt: OUT_DIR empty → torchtitan loads PHASEA_CKPT (model-only).
    # Resume attempts: torchtitan auto-resumes OUT_DIR latest (full state);
    # STAGE1_CKPT still must exist for launch_stage2 preflight.
    log "attempt ${attempt}/${MAX_ATTEMPTS}: $( [[ -z "${LC}" ]] && echo 'fresh init from step-2200' || echo "resume from ${LC}" ), seed=${attempt}"

    STAGE1_CKPT="${LC:-${PHASEA_CKPT}}" \
    JSON="${DISTILLED}" \
    OUT_DIR="${OUT_DIR}" \
    SEQ_LEN="${SEQ_LEN}" TEXT_LEN="${TEXT_LEN}" LR="${LR}" \
    STEPS=5200 WARMUP_STEPS=156 \
    SAVE_FREQ="${SAVE_FREQ}" KEEP_K=2 \
    MM_SHUFFLE_SEED="${attempt}" \
    bash "${SCRIPT_DIR}/launch_stage2.sh" \
        > "${LOG_DIR}/attempt${attempt}.log" 2>&1
    rc=$?

    if (( rc == 0 )) && grep -qE "step:\s*5200" "${LOG_DIR}/attempt${attempt}.log" 2>/dev/null; then
        log "✅ S5 COMPLETE (attempt ${attempt}). latest=$(latest_ckpt)"
        echo "S5_DONE" > "${OUT_DIR}/STATUS"; exit 0
    fi
    laststep=$(grep -oE "step:\s*[0-9]+" "${LOG_DIR}/attempt${attempt}.log" 2>/dev/null | tail -1)
    log "attempt ${attempt} failed rc=${rc} (${laststep:-no steps}); retry in ${RETRY_GRACE}s"
    grep -E "device-side assert|RuntimeError|out of memory|Error" "${LOG_DIR}/attempt${attempt}.log" 2>/dev/null | tail -3
    pkill -9 -f train_mm 2>/dev/null; pkill -9 -f torchrun 2>/dev/null
    sleep "${RETRY_GRACE}"
done

log "EXHAUSTED ${attempt} attempts (or deadline). latest=$(latest_ckpt)"
echo "S5_INCOMPLETE" > "${OUT_DIR}/STATUS"; exit 1
