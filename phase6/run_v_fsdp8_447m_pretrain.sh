#!/usr/bin/env bash
# v_fsdp8 multimodal continued pretrain on 447m aligned ckpt.
#
# Fallback config: pure FSDP=8 (no PP/TP/EP/V) instead of v11/v12's
# 4D mesh, because torchtitan @ attention_residual_dev tip + torch
# 2.9 stable on 8x 5090 hits an FSDP/TP parent-mesh assertion when
# the 4D mesh is built. Pure FSDP sidesteps the assertion.
#
# Tradeoff vs v11/v12: we lose 4D fabric (PP send/recv, TP all-reduce,
# EP all-to-all) coverage for the multimodal continued-pretrain run,
# but the resulting trained ckpt is the same shape and downstream-
# compatible with SFT + PPO. Fabric coverage from phase 5/6 (436m)
# already exists in the catalog.
#
# Workload: STEPS=2500 (half of v11/v12's 5000) because:
#   * starting from step-12500 (vs original step-8000) → already
#     +4500 steps of LM-only training in the 447m phase 4 retrain
#   * 2500 multimodal steps gives a proper continued-pretrain signal
#     at ~12-13h on 8x 5090, fits in overnight budget
set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4/runs/lm_447m_base/checkpoint/step-12500"
OUT_DIR="$WORKSPACE_DIR/phase5/runs/vlm_447m_pretrain"

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase6/v_fsdp8_447m_orchestrator.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] v_fsdp8 447m multimodal pretrain START"
echo "==============================================================="

MAX_RETRIES=10
attempt=0
while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    free_gb=$(df -BG --output=avail "$WORKSPACE_DIR" | tail -1 | tr -d 'G ')
    if [[ "$free_gb" -lt 32 ]]; then
        echo "[$(date)] DISK ABORT: ${free_gb}GB free < 32GB"
        break
    fi
    echo "[$(date)] attempt #$attempt (disk: ${free_gb}GB)"

    OUT_DIR="$OUT_DIR" \
    FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
    PP_MICROBATCH=8 \
    STEPS=2500 LOCAL_BS=16 GLOBAL_BS=128 SEQ_LEN=260 \
    FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
    STUDENT_CKPT="$PHASE4_CKPT" \
    SEED=42 DETERMINISTIC=0 COMPILE=0 \
    LR=1e-5 WARMUP=200 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=250 KEEP_K=2 \
    TRACE_TIER= \
    bash "$LAUNCHER"
    rc=$?
    last_step=$(grep -oE "step:\s*[0-9]+" "$OUT_DIR/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step=${last_step:-0}
    echo "[$(date)] attempt #$attempt rc=$rc last_step=$last_step"
    if [[ "$last_step" -ge 2500 ]]; then
        echo "[$(date)] DONE at step $last_step"
        break
    fi
    if [[ "$rc" -eq 0 ]]; then
        echo "[$(date)] clean exit at step $last_step before STEPS=2500; stop"
        break
    fi
    sleep 5
done
echo "==============================================================="
echo "[$(date)] v_fsdp8 447m END"
echo "==============================================================="
