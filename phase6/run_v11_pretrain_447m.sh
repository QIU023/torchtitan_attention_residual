#!/usr/bin/env bash
# v11 multimodal continued pretrain — 4D parallel (FSDP+PP+TP+EP)
#
# Same data + per-step batch as v10 (GBS=320, LBS=160, micro=10) but
# upgrades 3D (FSDP=2 PP=2 TP=2) to 4D by adding EP=2 (borrowed from
# FSDP×TP=4). Routed experts shard across the EP mesh, non-expert
# params shard across edp_mesh = (dp_replicate × efsdp).
#
# Mesh:
#   PP=2 × FSDP=2 × TP=2  = 8 (dense, all 8 GPUs)
#   EP=2 borrows from FSDP×TP=4
#
#   ranks 0..7 group membership:
#     PP groups: {0,1,2,3}, {4,5,6,7}        (PP rank 0 / 1)
#     FSDP groups: {0,1}, {2,3}, {4,5}, {6,7}
#     TP groups: {0,2}, {1,3}, {4,6}, {5,7}
#     EP groups: same physical pairs as FSDP×TP combined
#
# Pre-req fixes:
#   - apply_tp_kimi_linear(skip_expert_params=True) under EP
#     (torchtitan @ attention_residual_dev tip)
#   - apply_fsdp(ep_degree, edp_mesh) for nested expert shard
#     (same commit)
#
# To run: just bash this script. It does NOT auto-kill prior runs —
# stop v10 first if shared box. Auto-resume happens via the trainer
# checkpointer when OUT_DIR contains step-N/.
set -u

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$WORKSPACE_DIR/phase6/launch_8gpu_mm.sh"
PHASE4_CKPT="$WORKSPACE_DIR/phase4/runs/kimi_447m_aligned_block_attn_res_fsdp_paperhparams/checkpoint/step-12500"
OUT_DIR="$WORKSPACE_DIR/phase5/runs/v11_4d_447m_aligned_continue_from_step12500"

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase6/v11_447m_orchestrator.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] v11 4D pretrain START"
echo "==============================================================="

# Auto-retry loop: grouped_mm device-side asserts hit ~every 300-500
# steps under EP=2 + micro=20, but the fix is upstream cublas. Each
# crash that lands past a SAVE_FREQ boundary just resumes from the
# latest ckpt; loop until 5000 done. Bound retries to avoid runaway.
MAX_RETRIES=20
attempt=0
while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))

    # Disk pre-flight + per-attempt cleanup; see phase6/DISK_DISCIPLINE.md.
    free_gb=$(df -BG --output=avail "$WORKSPACE_DIR" | tail -1 | tr -d 'G ')
    if [[ "$free_gb" -lt 32 ]]; then
        echo "[$(date)] v11 DISK ABORT: ${free_gb}GB free < 32GB required"
        break
    fi
    echo "[$(date)] v11 attempt #$attempt (disk free: ${free_gb}GB)"
    if [[ "$attempt" -gt 1 ]]; then
        rm -f "$OUT_DIR/tier_b_trace/nccl-rank-"*.log 2>/dev/null
    fi
    if [[ "$attempt" -eq 1 ]]; then
        TRACE_ENV_TIER=tier_b
    else
        TRACE_ENV_TIER=
    fi

    OUT_DIR="$OUT_DIR" \
    FSDP=2 DP_REP=1 PP=2 TP=2 CP=1 EP=2 V=2 ADAPTER=1 \
    PP_MICROBATCH=20 \
    STEPS=5000 LOCAL_BS=200 GLOBAL_BS=400 SEQ_LEN=260 \
    FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
    STUDENT_CKPT="$PHASE4_CKPT" \
    SEED=42 DETERMINISTIC=0 COMPILE=0 \
    LR=1e-5 WARMUP=200 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=200 KEEP_K=2 \
    TRACE_TIER="$TRACE_ENV_TIER" TRACE_STEPS=50 \
    bash "$LAUNCHER"
    rc=$?
    last_step=$(grep -oE "step:\s*[0-9]+" "$OUT_DIR/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step=${last_step:-0}
    echo "[$(date)] v11 attempt #$attempt rc=$rc last_step=$last_step"
    if [[ "$last_step" -ge 5000 ]]; then
        echo "[$(date)] v11 done at step $last_step"
        break
    fi
    if [[ "$rc" -eq 0 ]]; then
        # Clean exit before STEPS — unusual; bail
        echo "[$(date)] v11 clean exit at step $last_step before STEPS=5000; stop"
        break
    fi
    echo "[$(date)] v11 crashed; sleeping 30s then retry"
    sleep 30
done

echo "[$(date)] v11 4D pretrain DONE (attempt=$attempt)"
