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

# Same retry-loop pattern as run_v11: grouped_mm device-side asserts
# under EP=2 are upstream cuBLAS / kimi-MoE-routing tail-distribution;
# auto-restart from latest ckpt.
MAX_RETRIES=25
attempt=0
while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    echo "[$(date)] v12 attempt #$attempt"
    OUT_DIR="$OUT_DIR" \
    FSDP=2 DP_REP=2 PP=2 TP=1 CP=1 EP=2 V=2 ADAPTER=1 \
    PP_MICROBATCH=20 \
    STEPS=5000 LOCAL_BS=200 GLOBAL_BS=800 SEQ_LEN=260 \
    FLAVOR=kimi_linear_436m_block_attn_res_n4 \
    STUDENT_CKPT="$PHASE4_CKPT" \
    SEED=42 DETERMINISTIC=0 COMPILE=0 \
    LR=1e-5 WARMUP=200 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=200 KEEP_K=2 \
    TRACE_TIER=tier_b TRACE_STEPS=50 \
    bash "$LAUNCHER"
    rc=$?
    last_step=$(grep -aoE "step:\s*[0-9]+" "$OUT_DIR/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step=${last_step:-0}
    echo "[$(date)] v12 attempt #$attempt rc=$rc last_step=$last_step"
    if [[ "$last_step" -ge 5000 ]]; then
        echo "[$(date)] v12 done at step $last_step"
        break
    fi
    if [[ "$rc" -eq 0 ]]; then
        echo "[$(date)] v12 clean exit at step $last_step before STEPS=5000; stop"
        break
    fi
    echo "[$(date)] v12 crashed; sleeping 30s then retry"
    sleep 30
done

echo "[$(date)] v12 4D pretrain DONE (attempt=$attempt)"
