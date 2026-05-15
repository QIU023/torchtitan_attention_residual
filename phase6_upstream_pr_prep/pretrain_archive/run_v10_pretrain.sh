#!/usr/bin/env bash
# v10 multimodal continued pretrain on 8 GPUs.
#
# Same recipe as v8/v9 production runs (GBS=120 LOCAL_BS=15 SEQ=260)
# from phase4_kimi_attnres_lm_pretrain/step-8000, FSDP=8 PP=1, 5000 steps. Includes Tier B
# trace capture on first 50 steps (production-realistic load).
#
# Crash-resilient: this script itself has no auto-resume loop, but
# the trainer registers projector + AdamW state with the checkpointer
# (commit 57a4b47) so manual resume from same dump_folder restores
# full state. If the run crashes, just rerun this script — the
# --checkpoint.initial_load_path is ignored on auto-resume.

set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6_upstream_pr_prep/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
OUT_DIR="$WORKSPACE_DIR/phase5_vlm_multimodal_sft/runs/v10_3d_lbs160_mb10_continue_8gpu_from_p4_step8000"

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase6_upstream_pr_prep/v10_orchestrator.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] v10 pretrain START"
echo "==============================================================="

# Cleanup any leftover ranks
if pgrep -f "phase5_vlm_multimodal_sft.train_mm" >/dev/null 2>&1; then
    pkill -TERM -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true; sleep 30
    pkill -KILL -f "phase5_vlm_multimodal_sft.train_mm" 2>/dev/null || true; sleep 10
fi

OUT_DIR="$OUT_DIR" \
FSDP=2 DP_REP=1 PP=2 TP=2 CP=1 EP=1 V=2 ADAPTER=1 \
PP_MICROBATCH=10 \
STEPS=5000 LOCAL_BS=160 GLOBAL_BS=320 SEQ_LEN=260 \
FLAVOR=kimi_linear_436m_block_attn_res_n4 \
STUDENT_CKPT="$PHASE4_CKPT" \
SEED=42 DETERMINISTIC=0 COMPILE=1 \
LR=1e-5 WARMUP=200 \
CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=2 \
TRACE_TIER=tier_b TRACE_STEPS=50 \
bash "$LAUNCHER" || {
    echo "[$(date)] [ERROR] v10 failed; user can re-run with same dump_folder"
}

echo "[$(date)] v10 pretrain DONE"
