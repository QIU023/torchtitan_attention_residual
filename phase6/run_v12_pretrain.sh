#!/usr/bin/env bash
# v12 multimodal continued pretrain — 4D parallel, EP replaces TP
#
# Compared to v11 (FSDP=2 PP=2 TP=2 EP=2):
#   v12 = FSDP=2 dp_replicate=2 PP=2 TP=1 EP=2
#
# Same GBS=400 as v11 (preserve hyperparam intent). Trade:
#   - No TP -> per-rank model params 2x (no head sharding)
#   - dp_world = FSDP × dp_replicate = 4 -> need LBS=100 to keep GBS=400
#   - micro=10 to halve activation pressure (compensate for missing TP)
#
# Mesh:
#   PP=2 × dp_replicate=2 × FSDP=2 × TP=1  = 8 (dense, 8 GPUs)
#   EP=2 borrows from FSDP=2 (efsdp size = 1)
#
# Compared to v11, fabric pattern shifts:
#   - TP AllReduce traffic disappears
#   - dp_replicate adds AllReduce of grads across 2 replicas
#   - same EP all-to-all volume
#
# Pre-req: torchtitan @ attention_residual_dev with the EP fixes
# (apply_tp skip_expert_params + apply_fsdp ep_degree/edp_mesh).
set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
OUT_DIR="$WORKSPACE_DIR/phase5/runs/v12_4d_fsdp2_dp2_pp2_ep2_continue_8gpu_from_p4_step8000"

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase6/v12_orchestrator.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] v12 4D pretrain START (EP replaces TP)"
echo "==============================================================="

OUT_DIR="$OUT_DIR" \
FSDP=2 DP_REP=2 PP=2 TP=1 CP=1 EP=2 V=2 ADAPTER=1 \
PP_MICROBATCH=10 \
STEPS=5000 LOCAL_BS=100 GLOBAL_BS=400 SEQ_LEN=260 \
FLAVOR=kimi_linear_436m_block_attn_res_n4 \
STUDENT_CKPT="$PHASE4_CKPT" \
SEED=42 DETERMINISTIC=0 COMPILE=0 \
LR=1e-5 WARMUP=200 \
CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=2 \
TRACE_TIER=tier_b TRACE_STEPS=50 \
bash "$LAUNCHER" || {
    echo "[$(date)] [ERROR] v12 failed; rerun same dump_folder to auto-resume"
}

echo "[$(date)] v12 4D pretrain DONE"
