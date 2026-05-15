#!/usr/bin/env bash
# Chain orchestrator: A3 alignment (already launched separately) → wait
# for step 500 → alignment report → v10 in 3D parallelism.
#
# v10 in 3D: FSDP=2 × PP=2 × TP=2 V=2 + cache adapter, GBS=120 LBS=15,
# 5000 steps, from phase4_kimi_attnres_lm_pretrain/step-8000. Tier B trace on first 50 steps.
#
# Memory math at LBS=15 SEQ=260: layer activations + AttnRes block stack
# scale ~linearly with LBS. Should fit ~22 GB / 32 GB per rank.

set -u
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6_upstream_pr_prep/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
ORCH_LOG="$WORKSPACE_DIR/phase6_upstream_pr_prep/a3_v10_3d_orchestrator.log"
A3_LOG="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a3_fsdp2_pp2_tp2_seed42/train.log"

exec >>"$ORCH_LOG" 2>&1
echo ""
echo "==============================================================="
echo "[$(date)] A3-then-v10-3D orchestrator START"
echo "==============================================================="

# Phase 1: wait for A3 step 500 or terminal failure
echo "[$(date)] waiting for A3 step:500 in $A3_LOG"
until grep -qE "(- step:[ ]+500[ ]+|Training completed|FAILED|Killed|^Traceback)" "$A3_LOG" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] A3 wait condition met"

# Cleanup any stragglers before phase 2
if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
    pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true; sleep 30
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true; sleep 10
fi

# Phase 2: alignment report A3 vs B0 + collective extraction
echo ""
echo "[$(date)] running alignment report"
bash "$WORKSPACE_DIR/phase6_upstream_pr_prep/run_alignment_reports.sh" || true
echo "[$(date)] extracting collectives"
/venv/main/bin/python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/extract_collectives.py" \
    "$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/8gpu_a3_fsdp2_pp2_tp2_seed42/tier_c_trace/" || true
/venv/main/bin/python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/build_pattern_catalog.py" || true

# Phase 3: v10 in 3D parallelism (FSDP=2 PP=2 TP=2)
echo ""
echo "==============================================================="
echo "[$(date)] launching v10 in 3D parallelism (FSDP=2 PP=2 TP=2 V=2)"
echo "==============================================================="

V10_OUT="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v10_3d_continue_8gpu_from_p4_step8000"
rm -rf "$V10_OUT"

OUT_DIR="$V10_OUT" \
FSDP=2 PP=2 TP=2 CP=1 EP=1 V=2 \
STEPS=5000 LOCAL_BS=15 GLOBAL_BS=120 SEQ_LEN=260 \
FLAVOR=kimi_linear_436m_block_attn_res_n4 \
STUDENT_CKPT="$PHASE4_CKPT" \
SEED=42 DETERMINISTIC=0 COMPILE=1 \
LR=1e-5 WARMUP=200 \
SAVE_FREQ=500 KEEP_K=3 \
TRACE_TIER=tier_b TRACE_STEPS=50 \
bash "$LAUNCHER" || {
    echo "[$(date)] [ERROR] v10 3D failed; check train.log"
}

echo ""
echo "[$(date)] orchestrator COMPLETE"
