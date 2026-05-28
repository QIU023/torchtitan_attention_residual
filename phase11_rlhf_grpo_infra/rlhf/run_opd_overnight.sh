#!/usr/bin/env bash
# D-5 overnight: VQA-aligned 600-step OPD + eval cascade.
# Simplified rewrite — no traps, no double-tee, errors logged not silenced.
set -uo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

OVERNIGHT_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/opd_d5_overnight"
mkdir -p "$OVERNIGHT_DIR"

# Mark start so subsequent bash calls can append to a flat log via direct redirect
# (no tee — caller redirects whole script's stdout/stderr).
echo "============================================================"
echo "D-5 OVERNIGHT @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- Pre-flight ----
FREE_G=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "[preflight] disk = ${FREE_G}G"
if (( FREE_G < 30 )); then
    echo "FATAL: only ${FREE_G}G free; need ≥30G"
    exit 1
fi
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
echo "[preflight] cuda:0 free = ${GPU_FREE} MiB"

# ---- Watchdog (no trap; bg PID file so we can kill from a different bash) ----
(
    while true; do
        sleep 90
        F=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
        if (( F < 12 )); then
            echo "[watchdog $(date '+%H:%M:%S')] PANIC: disk ${F}G; killing OPD procs"
            pkill -9 -f run_grpo_llava_kimi.py 2>/dev/null
            pkill -9 -f dcp_to_hf_kimi_attn_res_vl.py 2>/dev/null
            pkill -9 -f gqa_eval.py 2>/dev/null
            touch "$OVERNIGHT_DIR/DISK_PANIC"
            exit 1
        fi
    done
) &
WD_PID=$!
echo "[watchdog] PID=$WD_PID"

cleanup_watchdog() {
    kill -9 "$WD_PID" 2>/dev/null
    echo "[watchdog] stopped"
}

# ---- Phase 1: 1-step smoke ----
SMOKE_LOG="$OVERNIGHT_DIR/smoke_1step.log"
rm -rf phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts 2>/dev/null
echo "[$(date '+%H:%M:%S')] Phase 1: 1-step smoke → $SMOKE_LOG"
NEPS=2 CKPT_INTERVAL=1 OPD_LR=1e-5 OPD_WD=0.01 OPD_BETA=0.5 OPD_T=2.0 \
    OPD_TASK_TYPE=vqa \
    timeout 1200 \
    bash phase11_rlhf_grpo_infra/rlhf/run_opd_overnight_inner.sh 1 \
    > "$SMOKE_LOG" 2>&1
SMOKE_EXIT=$?
echo "[$(date '+%H:%M:%S')] Phase 1: smoke exit=$SMOKE_EXIT"
if ! grep -q "step   0" "$SMOKE_LOG" 2>/dev/null; then
    echo "FATAL: smoke didn't reach step 0; tail:"
    tail -30 "$SMOKE_LOG"
    cleanup_watchdog
    exit 1
fi
echo "[$(date '+%H:%M:%S')] Phase 1: smoke PASSED"
rm -rf phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts

# ---- Phase 2: 600-step real run ----
RUN_LOG="$OVERNIGHT_DIR/run_600step.log"
echo "[$(date '+%H:%M:%S')] Phase 2: 600-step OPD run → $RUN_LOG"
echo "       LR=1e-5  T=2.0  β=0.5  NEPS=2  task=vqa  ckpt every 100  ETA ~8h"
NEPS=2 CKPT_INTERVAL=100 OPD_LR=1e-5 OPD_WD=0.01 OPD_BETA=0.5 OPD_T=2.0 \
    OPD_TASK_TYPE=vqa \
    bash phase11_rlhf_grpo_infra/rlhf/run_opd_overnight_inner.sh 600 \
    > "$RUN_LOG" 2>&1
RUN_EXIT=$?
echo "[$(date '+%H:%M:%S')] Phase 2: run exit=$RUN_EXIT"

CKPT_DIR="phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts"
echo "[$(date '+%H:%M:%S')] Phase 2: ckpts present:"
ls -la "$CKPT_DIR" 2>&1 | head -10
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "FATAL: no ckpts; aborting eval"
    cleanup_watchdog
    exit 1
fi

# ---- Phase 3: eval cascade ----
echo "[$(date '+%H:%M:%S')] Phase 3: eval cascade"
EVAL_REPORT="$OVERNIGHT_DIR/eval_summary.md"
{
    echo "# Stage D-5 OPD Eval Cascade"
    echo
    echo "GQA testdev greedy N=300. Baseline (SFT step-5200): 37/300 = **12.3%**."
    echo "D-4 (50-step caption, lr=1e-5): 28/300 = **9.3%**."
    echo
    echo "| Ckpt | GQA acc | Notes |"
    echo "|---|---|---|"
} > "$EVAL_REPORT"

SRC_HF="phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
for STEP_DIR in $(ls -d "$CKPT_DIR"/step-* 2>/dev/null | sort -V); do
    STEP=$(basename "$STEP_DIR" | sed 's/step-//')
    HF_OUT="phase11_rlhf_grpo_infra/hf/opd_d5_step${STEP}"
    EVAL_LOG="$OVERNIGHT_DIR/eval_step${STEP}.log"
    echo "[$(date '+%H:%M:%S')]   step-$STEP: convert + eval"

    torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
        --in "$STEP_DIR" --out "$HF_OUT" \
        --config kimi_linear_447m_aligned_block_attn_res_n4 \
        --vision-tower google/siglip-base-patch16-224 \
        --processor-source "$SRC_HF" --projector-from-hf "$SRC_HF" \
        > "$EVAL_LOG" 2>&1
    CONV_EXIT=$?
    if (( CONV_EXIT != 0 )); then
        echo "[$(date '+%H:%M:%S')]     convert FAILED (exit $CONV_EXIT)"
        echo "| step-$STEP | CONVERT_FAILED | exit=$CONV_EXIT |" >> "$EVAL_REPORT"
        continue
    fi

    HF_MODEL_PATH="$(pwd)/$HF_OUT" N_EVAL=300 \
        ATTNRES_MLA_FP32_FALLBACK=1 SGLANG_DISABLE_SHM_MM=1 \
        SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts" \
        /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/gqa_eval.py \
        >> "$EVAL_LOG" 2>&1
    EVAL_EXIT=$?

    ACC_LINE=$(grep "GQA testdev greedy accuracy:" "$EVAL_LOG" | tail -1)
    if [[ -n "$ACC_LINE" ]]; then
        ACC=$(echo "$ACC_LINE" | sed -n 's/.*= \(.*\) =====/\1/p')
        echo "[$(date '+%H:%M:%S')]     step-$STEP acc: $ACC"
        echo "| step-$STEP | $ACC | eval_step${STEP}.log |" >> "$EVAL_REPORT"
    else
        echo "[$(date '+%H:%M:%S')]     step-$STEP: no acc line; eval_exit=$EVAL_EXIT"
        echo "| step-$STEP | EVAL_FAILED (exit=$EVAL_EXIT) | eval_step${STEP}.log |" >> "$EVAL_REPORT"
    fi

    # Cleanup DCP after eval to save disk
    rm -rf "$STEP_DIR"
    F=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
    echo "[$(date '+%H:%M:%S')]     disk after cleanup: ${F}G"
done

echo "[$(date '+%H:%M:%S')] Phase 3 DONE. Summary:"
cat "$EVAL_REPORT"

cleanup_watchdog
echo "[$(date '+%H:%M:%S')] ORCHESTRATOR DONE"
