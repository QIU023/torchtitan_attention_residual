#!/usr/bin/env bash
# OPD Stage D-5 OVERNIGHT — task-aligned 600-step OPD + per-ckpt eval cascade.
#
# Goal: produce PR14 capability-lift evidence with substantively-trained
# distilled student. Stage D-4 confirmed pipeline works (loss stable,
# no degradation) at 50 steps but landed below baseline (9.3% vs
# 12.3%) due to task mismatch (caption OPD vs VQA eval). D-5 fixes
# task mismatch by using mix665k real VQA conversations as OPD prompts.
#
# Schedule (15h budget):
#   Phase 1: 1-step smoke    ~10 min
#   Phase 2: 600-step run    ~8 h    (~47s/step × 600 + boot)
#   Phase 3: 6× eval cascade ~1.5h   (DCP→HF + GQA testdev N=300, each ~15 min)
#   Phase 4: REPORT.md       (manual; logs already in place)
#   Buffer:  ~4 h            for retries, additional ablations
#
# Disk safety:
#   * Pre-flight: requires ≥30 G free.
#   * Background watchdog: kills the run if /workspace <12 G during training.
#   * Post-eval: delete each ckpt's DCP after its HF conversion + GQA eval,
#     keeping only HF/eval logs for the report (saves ~3 G per ckpt).
#
# Disable core dumps (Lance/Monarch crashes previously dumped 122 G).
set -euo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

OVERNIGHT_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/opd_d5_overnight"
mkdir -p "$OVERNIGHT_DIR"
MAIN_LOG="$OVERNIGHT_DIR/orchestrator.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

# ---- Phase 0: pre-flight ----
FREE_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if [[ "$FREE_G" -lt 30 ]]; then
    log "FATAL: only ${FREE_G}G free; need ≥30G for 600-step + 6 ckpts."
    exit 1
fi
log "pre-flight: disk OK (${FREE_G}G free)"

GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "pre-flight: cuda:0 free = ${GPU_FREE} MiB"
if [[ "$GPU_FREE" -lt 30000 ]]; then
    log "WARN: cuda:0 has <30G free; another process may be running"
fi

# ---- Disk watchdog (background) ----
WATCHDOG_PID=""
start_watchdog() {
    (
        while true; do
            sleep 60
            FREE=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
            if [[ "$FREE" -lt 12 ]]; then
                echo "[watchdog $(date '+%H:%M:%S')] PANIC: disk ${FREE}G < 12G threshold; killing all OPD processes" \
                    | tee -a "$MAIN_LOG"
                pkill -9 -f run_grpo_llava_kimi.py || true
                pkill -9 -f dcp_to_hf_kimi_attn_res_vl.py || true
                pkill -9 -f gqa_eval.py || true
                touch "$OVERNIGHT_DIR/DISK_PANIC"
                exit 1
            fi
        done
    ) &
    WATCHDOG_PID=$!
    log "disk watchdog started (PID=$WATCHDOG_PID, threshold=12G)"
}

stop_watchdog() {
    if [[ -n "$WATCHDOG_PID" ]]; then
        kill -9 "$WATCHDOG_PID" 2>/dev/null || true
        log "disk watchdog stopped"
    fi
}
trap stop_watchdog EXIT

start_watchdog

# ---- Phase 1: 1-step smoke ----
SMOKE_LOG="$OVERNIGHT_DIR/smoke_1step.log"
rm -rf phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts || true
log "Phase 1: 1-step VQA-task smoke → $SMOKE_LOG"

NEPS=2 CKPT_INTERVAL=1 OPD_LR=1e-5 OPD_WD=0.01 OPD_BETA=0.5 OPD_T=2.0 \
    OPD_TASK_TYPE=vqa \
    timeout 900 \
    bash phase11_rlhf_grpo_infra/rlhf/run_opd_overnight_inner.sh 1 \
    &> "$SMOKE_LOG" || true

if ! grep -q "^\[actor=<root>\] step\s*0" "$SMOKE_LOG"; then
    log "FATAL: smoke didn't reach step 0; see $SMOKE_LOG"
    tail -30 "$SMOKE_LOG" | tee -a "$MAIN_LOG"
    exit 1
fi
log "Phase 1: smoke PASSED"
rm -rf phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts

# ---- Phase 2: 600-step real run ----
RUN_LOG="$OVERNIGHT_DIR/run_600step.log"
log "Phase 2: 600-step OPD run → $RUN_LOG"
log "       LR=1e-5  T=2.0  β=0.5  NEPS=2  task=vqa  ckpt every 100"
log "       ETA ~8h"

NEPS=2 CKPT_INTERVAL=100 OPD_LR=1e-5 OPD_WD=0.01 OPD_BETA=0.5 OPD_T=2.0 \
    OPD_TASK_TYPE=vqa \
    bash phase11_rlhf_grpo_infra/rlhf/run_opd_overnight_inner.sh 600 \
    &> "$RUN_LOG"
RUN_EXIT=$?
log "Phase 2: run exit=$RUN_EXIT"

# Check what ckpts landed
CKPT_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts"
log "Phase 2: ckpts present:"
ls -la "$CKPT_DIR" 2>&1 | tee -a "$MAIN_LOG"

if [[ ! -d "$CKPT_DIR" ]]; then
    log "FATAL: no ckpts produced; cannot proceed to evals"
    exit 1
fi

# ---- Phase 3: eval cascade with per-ckpt cleanup ----
log "Phase 3: eval cascade (DCP→HF + GQA testdev N=300 per ckpt)"
EVAL_REPORT="$OVERNIGHT_DIR/eval_summary.md"
{
    echo "# Stage D-5 OPD Eval Cascade — GQA testdev greedy N=300"
    echo
    echo "Baseline (SFT step-5200): 37/300 = **12.3%**"
    echo "D-4 (50-step caption, lr=1e-5): 28/300 = **9.3%**"
    echo
    echo "| Ckpt | GQA acc | dt vs baseline | Notes |"
    echo "|---|---|---|---|"
} > "$EVAL_REPORT"

SRC_HF="phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
for STEP_DIR in $(ls -d "$CKPT_DIR"/step-* 2>/dev/null | sort -V); do
    STEP=$(basename "$STEP_DIR" | sed 's/step-//')
    HF_OUT="phase11_rlhf_grpo_infra/hf/opd_d5_step${STEP}"
    EVAL_LOG="$OVERNIGHT_DIR/eval_step${STEP}.log"

    log "  Phase 3.${STEP}: convert → eval → cleanup"

    # Convert
    /usr/bin/python3 -c "import torch.distributed as d; d.init_process_group(backend='gloo')" >/dev/null 2>&1 || true
    torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
        --in "$STEP_DIR" --out "$HF_OUT" \
        --config kimi_linear_447m_aligned_block_attn_res_n4 \
        --vision-tower google/siglip-base-patch16-224 \
        --processor-source "$SRC_HF" --projector-from-hf "$SRC_HF" \
        &>> "$EVAL_LOG" || { log "    convert FAILED at step $STEP"; continue; }

    # Eval
    HF_MODEL_PATH="$(pwd)/$HF_OUT" N_EVAL=300 \
        ATTNRES_MLA_FP32_FALLBACK=1 SGLANG_DISABLE_SHM_MM=1 \
        SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts" \
        /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/gqa_eval.py \
        &>> "$EVAL_LOG" || true

    ACC_LINE=$(grep "GQA testdev greedy accuracy:" "$EVAL_LOG" | tail -1 || true)
    if [[ -n "$ACC_LINE" ]]; then
        ACC=$(echo "$ACC_LINE" | sed -n 's/.*= \(.*\) =====/\1/p')
        log "    step-$STEP: $ACC_LINE"
        echo "| step-$STEP | $ACC | (baseline 12.3%) | see eval_step${STEP}.log |" >> "$EVAL_REPORT"
    else
        log "    step-$STEP: no acc line found"
        echo "| step-$STEP | EVAL_FAILED | — | see eval_step${STEP}.log |" >> "$EVAL_REPORT"
    fi

    # Per-ckpt cleanup: delete DCP after HF+eval done (saves ~3G)
    rm -rf "$STEP_DIR"
    # Also delete HF after eval (saves another ~3G) — keep only the LAST one for inspection
    # Keep all HF dirs for now; can manually clean later.
    # rm -rf "$HF_OUT"

    FREE=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
    log "    disk after step-$STEP cleanup: ${FREE}G free"
done

log "Phase 3 DONE. Eval summary: $EVAL_REPORT"
cat "$EVAL_REPORT" | tee -a "$MAIN_LOG"

log "ORCHESTRATOR DONE"
