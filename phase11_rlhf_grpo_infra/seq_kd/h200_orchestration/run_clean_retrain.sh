#!/usr/bin/env bash
# Clean VLM retrain chain with the projector-preservation fix.
#
# Why this exists: the original pipeline used --checkpoint.initial_load_model_only
# at every stage transition, which DROPS mm_state/projector (checkpoint.py:788
# returns states[MODEL] only). So the projector was re-randomized at every stage
# (Stage-1 alignment + each stage's projector training discarded; proven by
# cross-ckpt projector cos~0). The fix (train_mm.py --mm.pretrain-projector-path,
# wired through launch_stage2.sh) explicitly carries the prior stage's projector
# and HARD-VERIFIES the load (PROJ_VERIFY cos>=0.999) or aborts.
#
# Chain (no Stage-1 rerun, no 558k needed — start from the GDrive step-8720):
#   Stage-1 base : runs/stage1_alignment_447m/checkpoint/step-8720
#                  (frozen-LM aligned projector + base LM; verified non-random)
#   Stage-2  SFT : LM+proj on mix665k         (projector loaded from step-8720)
#   seq-KD   SFT : LM+proj on distilled TASKMIX (projector loaded from Stage-2)
#
# Each stage's projector carries forward (PROJ_VERIFY in every launch). Crash
# auto-resume is per-stage (resume reloads that stage's own mm_state).
set -uo pipefail
source /home/seqkd_overnight/lib.sh

S5="${REPO}/phase5_vlm_multimodal_sft"
STAGE1="${STAGE1:-${S5}/runs/stage1_alignment_447m/checkpoint/step-8720}"
MIX665K="${MIX665K:-/home/.hf_home/LLaVA-Instruct/llava_v1_5_mix665k.json}"
TASKMIX="${TASKMIX:-${REPO}/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_TASKMIX.json}"
STAGE2_OUT="${STAGE2_OUT:-${S5}/runs/stage2_clean_447m}"
SEQKD_OUT="${SEQKD_OUT:-${S5}/runs/seqkd_clean_447m}"
STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_447m_aligned_block_attn_res_n4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

# ---- preflight: all inputs present (fail fast, not mid-overnight) ----
ckpt_ok "${STAGE1}" || { log "FATAL: stage1 ckpt missing/invalid ${STAGE1}"; exit 2; }
[[ -f "${MIX665K}" ]] || { log "FATAL: mix665k json missing ${MIX665K}"; exit 2; }
[[ -f "${TASKMIX}" ]] || { log "FATAL: distilled TASKMIX missing ${TASKMIX}"; exit 2; }
# PROJ_VERIFY at the source: stage1 ckpt must carry a non-random projector.
"${CPY}" - "${STAGE1}" <<'PY' || { log "FATAL: stage1 projector check failed"; exit 2; }
import sys, torch, torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
r = FileSystemReader(sys.argv[1]); md = r.read_metadata()
pk = [k for k in md.state_dict_metadata if k.startswith("mm_state.projector.")]
assert pk, "no mm_state.projector.* in stage1 ckpt"
tgt = {}
for k in pk:
    m = md.state_dict_metadata[k]
    tgt[k] = torch.empty(tuple(m.size), dtype=getattr(m.properties, "dtype", torch.float32))
dcp.load(tgt, storage_reader=r)
bias = [k for k in pk if k.endswith("fc1.bias")][0]
assert tgt[bias].float().abs().max() > 1e-6, "stage1 projector fc1.bias is zero (untrained)"
print(f"  stage1 projector OK: fc1.bias absmax={tgt[bias].float().abs().max():.5f}")
PY

# ---- generic SFT stage (LM+proj, projector carried from $init) ----
run_sft_stage() {   # $1=name  $2=init_ckpt  $3=json  $4=out_dir  $5=steps_cap
    local name="$1" init="$2" json="$3" out="$4" cap="$5"
    mkdir -p "${out}"
    local NROWS STEPS WARMUP attempt LC rc lg
    NROWS=$("${CPY}" -c "import json;print(len(json.load(open('${json}'))))")
    STEPS=$(( (NROWS + 128 - 1) / 128 ))
    (( STEPS > cap )) && STEPS="${cap}"
    (( STEPS < 50 )) && STEPS=50
    WARMUP=$(( STEPS / 20 + 1 ))
    log "[${name}] rows=${NROWS} STEPS=${STEPS} warmup=${WARMUP} out=${out}"
    attempt=0
    while (( attempt < MAX_ATTEMPTS )); do
        attempt=$((attempt+1))
        # drop crash-during-save partials (no .metadata)
        for c in "${out}"/checkpoint/step-*; do
            [[ -d "$c" && ! -f "$c/.metadata" ]] && { log "[${name}] rm incomplete $(basename "$c")"; rm -rf "$c"; }
        done
        LC=$(ls -d "${out}/checkpoint/step-"* 2>/dev/null | sort -t- -k2 -n | tail -1)
        log "[${name}] attempt ${attempt}/${MAX_ATTEMPTS}: $([[ -z "$LC" ]] && echo "fresh, projector<-${init}" || echo "auto-resume ${LC}")"
        lg="${ROOT}/logs/clean_${name}_attempt${attempt}.log"
        AC_MODE="${AC_MODE:-full}" STUDENT_CONFIG="${STUDENT_CONFIG}" \
        STAGE1_CKPT="${LC:-${init}}" \
        JSON="${json}" \
        IMAGES=/home/.hf_home/LLaVA-Instruct/images \
        INSTRUCT_DIR=/home/.hf_home/LLaVA-Instruct \
        CACHE_DIR=/home/.hf_home \
        OUT_DIR="${out}" \
        NGPU="${NGPU:-2}" GLOBAL_BS="${GLOBAL_BS:-128}" LOCAL_BS="${LOCAL_BS:-64}" \
        SEQ_LEN="${SEQ_LEN:-1024}" TEXT_LEN="${TEXT_LEN:-828}" LR="${LR:-2e-5}" \
        STEPS="${STEPS}" WARMUP_STEPS="${WARMUP}" \
        SAVE_FREQ="${SAVE_FREQ:-200}" KEEP_K="${KEEP_K:-2}" \
        MM_SHUFFLE_SEED="${attempt}" \
        bash "${S5}/launch_stage2.sh" > "${lg}" 2>&1
        rc=$?
        # confirm PROJ_VERIFY actually fired (projector was carried, not reset)
        if ! grep -qE "PROJ_VERIFY .* OK" "${lg}" 2>/dev/null && [[ -z "$LC" ]]; then
            log "[${name}] WARN: no PROJ_VERIFY OK in log on a fresh start — projector may not have loaded"
        fi
        if (( rc == 0 )) && grep -qE "step:[[:space:]]*${STEPS}\b" "${lg}" 2>/dev/null; then
            log "[${name}] COMPLETE attempt=${attempt} latest=$(ls -d ${out}/checkpoint/step-* | sort -t- -k2 -n | tail -1)"
            return 0
        fi
        log "[${name}] attempt ${attempt} rc=${rc} ($(grep -oE 'step:[[:space:]]*[0-9]+' "${lg}" 2>/dev/null | tail -1)); retry"
        grep -E "PROJ_VERIFY FAIL|device-side assert|RuntimeError|out of memory|Traceback" "${lg}" 2>/dev/null | tail -4
        pkill -9 -f '[t]rain_mm' 2>/dev/null; pkill -9 -f '[t]orchrun' 2>/dev/null
        sleep 15
    done
    log "[${name}] EXHAUSTED attempts"
    return 1
}

# ---- Stage-2: instruct SFT on mix665k, projector from step-8720 ----
log "=== CLEAN RETRAIN: Stage-2 (mix665k) from ${STAGE1} ==="
run_sft_stage stage2 "${STAGE1}" "${MIX665K}" "${STAGE2_OUT}" "${STAGE2_CAP:-5200}" \
    || { log "FATAL: Stage-2 failed"; echo INCOMPLETE > "${STAGE2_OUT}/STATUS"; exit 1; }
STAGE2_CKPT=$(ls -d "${STAGE2_OUT}/checkpoint/step-"* 2>/dev/null | sort -t- -k2 -n | tail -1)
ckpt_ok "${STAGE2_CKPT}" || { log "FATAL: Stage-2 produced no valid ckpt"; exit 1; }
echo DONE > "${STAGE2_OUT}/STATUS"
log "=== Stage-2 done: ${STAGE2_CKPT} ==="

# ---- seq-KD: SFT on distilled TASKMIX, projector from Stage-2 ----
log "=== CLEAN RETRAIN: seq-KD (TASKMIX) from ${STAGE2_CKPT} ==="
run_sft_stage seqkd "${STAGE2_CKPT}" "${TASKMIX}" "${SEQKD_OUT}" "${SEQKD_CAP:-3000}" \
    || { log "FATAL: seq-KD failed"; echo INCOMPLETE > "${SEQKD_OUT}/STATUS"; exit 1; }
SEQKD_CKPT=$(ls -d "${SEQKD_OUT}/checkpoint/step-"* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo DONE > "${SEQKD_OUT}/STATUS"
log "=== CLEAN RETRAIN DONE. final=${SEQKD_CKPT} ==="
