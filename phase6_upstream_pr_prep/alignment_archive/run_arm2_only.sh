#!/usr/bin/env bash
# Phase 6 A1 — Arm 2 only (Arm 1' already produced its loss curve).
#
# Background: the orchestrator's first attempt at Arm 2 used LOCAL_BS=2
# GLOBAL_BS=32 → torchtitan computed only 2 microbatches per virtual stage
# under Interleaved1F1B with V=2 PP=4 (8 stages). The schedule's
# lookahead window starved → NCCL P2P collective timed out at 6 min in.
#
# Fix here: keep GLOBAL_BS=32 (matches Arm 1' for valid alignment) but
# set LOCAL_BS=1 → 32 total microbatches → 4 per virtual stage, well
# above the lookahead floor.
#
# This script ONLY runs stages 2 and 3 (Arm 2 + compare); Arm 1' from
# the earlier run is preserved at runs/arm1prime_fsdp_seed42_from_p4_8k.

set -euo pipefail

WORKSPACE_DIR=/root/torchtitan_attention_residual
PHASE4_CKPT="${WORKSPACE_DIR}/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"

ARM1P_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/arm1prime_fsdp_seed42_from_p4_8k"
ARM2_DIR="${WORKSPACE_DIR}/phase5_vlm_multimodal_sft/runs/arm2_pp4v2_adapter_seed42_from_p4_8k"

STEPS_ARM2="${STEPS_ARM2:-2000}"
SEED="${SEED:-42}"
GLOBAL_BS="${GLOBAL_BS:-32}"
LOCAL_BS_PP="${LOCAL_BS_PP:-1}"   # 32/1 = 32 microbatches / 8 stages = 4 per stage

LOG=/root/phase6_alignment_orchestrator.log
exec >>"$LOG" 2>&1

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 Arm 2 retry START (LOCAL_BS=$LOCAL_BS_PP GLOBAL_BS=$GLOBAL_BS)"
echo "==============================================================="

# Stage 0: GPU sanity
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
    echo "[$(date)] WARNING: phase5_vlm_multimodal_sft.train_mm still running, killing"
    pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 30
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true
    sleep 10
fi

# Stage 1: Arm 2
echo "[$(date)] [stage 1] launching Arm 2 PP=4 V=2 + cache adapter (STEPS=$STEPS_ARM2, GLOBAL_BS=$GLOBAL_BS, LOCAL_BS_PP=$LOCAL_BS_PP, seed=$SEED)"
mkdir -p "$ARM2_DIR"
cd "$WORKSPACE_DIR"
source /venv/main/bin/activate
INIT=weak_ckpt INIT_CKPT="$PHASE4_CKPT" \
NGPU=4 PP=4 V=2 LOCAL_BS="$LOCAL_BS_PP" GLOBAL_BS="$GLOBAL_BS" STEPS="$STEPS_ARM2" \
ADAPTER=1 SEED="$SEED" DETERMINISTIC=1 \
COMPILE=1 \
OUT_DIR="$ARM2_DIR" \
bash phase5_vlm_multimodal_sft/launch_pp_adapter.sh

echo "[$(date)] [stage 1] Arm 2 done"

# Stage 2: alignment report
echo "[$(date)] [stage 2] running compare_pp_vs_fsdp.py"
cd "$WORKSPACE_DIR"
python phase5_vlm_multimodal_sft/compare_pp_vs_fsdp.py \
    --pp "$ARM2_DIR/tb" \
    --fsdp "$ARM1P_DIR/tb" \
    > "$WORKSPACE_DIR/phase6_upstream_pr_prep/alignment_report_arm2_real_mm.txt" 2>&1 || true

echo "[$(date)] [stage 2] alignment report:"
cat "$WORKSPACE_DIR/phase6_upstream_pr_prep/alignment_report_arm2_real_mm.txt"

echo ""
echo "==============================================================="
echo "[$(date)] phase6 A1 Arm 2 retry COMPLETE"
echo "==============================================================="
