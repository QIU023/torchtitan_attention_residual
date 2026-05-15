#!/usr/bin/env bash
# Phase 7 5D fabric trace (best effort on 8 GPUs).
#
# Goal: capture NCCL collective patterns under combined PP / FSDP / TP /
# CP / EP for IXIA fabric profiling. Run uses a fresh-init small model
# on c4 text data — we only need the COLLECTIVE patterns, not training
# quality, so 50 step tier_b trace is enough.
#
# Hard math: 8 GPUs cannot host 5 axes each >=2 in their dense
# product (2^5 = 32 minimum). So "5D" here means "5 logical axes
# with EP/CP borrowing from existing dense axes".
#
# Three matrices (toggle via MODE env, default A):
#
#   A. LLaMA-3 4D (no EP since model is dense)
#       PP=2 × FSDP=2 × CP=2 × TP=1, dense=8
#       Captures: PP send/recv, FSDP allgather/reduce-scatter,
#                 CP K/V allgather. NO EP.
#
#   B. DeepSeek-V3 (MoE) full 5D
#       PP=2 × FSDP=2 × CP=2 × TP=1 × EP=2 (EP borrows FSDP)
#       Captures: above + EP all-to-all dispatch/combine.
#                 The most fabric-rich profile possible on 8 GPUs.
#
#   C. LLaMA-3 4D + TP (alt)
#       PP=2 × FSDP=2 × CP=1 × TP=2, dense=8
#       Captures: PP, FSDP, TP. No CP. (Already covered by v11/v12;
#                 included for completeness when CP unavailable.)
#
# Outputs:
#   phase7_nccl_traffic_catalog/traces/5d_fabric_<MODE>/
#     collective_summary.csv.gz (compressed)
#     flows.csv.gz
#     ixia_config.json (canonical IXIA artifact)
#     recipe.json (mesh config)
#
# Usage:
#   bash phase7_nccl_traffic_catalog/run_5d_fabric_trace.sh           # mode A (default)
#   MODE=B bash phase7_nccl_traffic_catalog/run_5d_fabric_trace.sh    # DSv3 5D
#   MODE=C bash phase7_nccl_traffic_catalog/run_5d_fabric_trace.sh    # LLaMA-3 + TP
set -u

MODE="${MODE:-A}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$WORKSPACE_DIR/phase7_nccl_traffic_catalog/traces/5d_fabric_${MODE}"

case "$MODE" in
    A)
        MODULE=llama3
        FLAVOR=llama3_debugmodel
        PP=2; FSDP=2; CP=2; TP=1; EP=1
        DP_REP=1
        AXIS_COUNT="4D (no EP, model is dense)"
        ;;
    B)
        MODULE=deepseek_v3
        FLAVOR=deepseek_v3_debugmodel
        PP=2; FSDP=2; CP=2; TP=1; EP=2
        DP_REP=1
        AXIS_COUNT="5D (PP+FSDP+CP+TP+EP, EP borrows FSDP)"
        ;;
    C)
        MODULE=llama3
        FLAVOR=llama3_debugmodel
        PP=2; FSDP=2; CP=1; TP=2; EP=1
        DP_REP=1
        AXIS_COUNT="4D-TP (no CP, no EP)"
        ;;
    *)
        echo "ERROR: unknown MODE=$MODE; choose A, B, or C" >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"
LOG="$WORKSPACE_DIR/phase7_nccl_traffic_catalog/orchestrator_5d_${MODE}.log"
exec >>"$LOG" 2>&1

# Disk pre-flight (see phase6_upstream_pr_prep/DISK_DISCIPLINE.md). 50-step tier_b trace
# can be ~2 GB raw across 8 ranks; require 5 GB free.
free_gb=$(df -BG --output=avail "$WORKSPACE_DIR" | tail -1 | tr -d 'G ')
if [[ "$free_gb" -lt 5 ]]; then
    echo "[$(date)] 5D trace DISK ABORT: ${free_gb}GB < 5GB"
    exit 1
fi

echo "==============================================================="
echo "[$(date)] 5D fabric trace START — MODE=$MODE ($AXIS_COUNT)"
echo "  module=$MODULE flavor=$FLAVOR"
echo "  PP=$PP FSDP=$FSDP CP=$CP TP=$TP EP=$EP DP_REP=$DP_REP"
echo "==============================================================="

# Sanity: dense product must equal NGPU=8.
DENSE=$(( PP * DP_REP * FSDP * TP * CP ))
if [[ "$DENSE" != 8 ]]; then
    echo "ERROR: dense=$DENSE != 8; mesh invalid for 8 GPUs" >&2
    exit 1
fi

# Run via torchtitan's standard train.py (not phase5_vlm_multimodal_sft/train_mm — this
# is text-only c4, multimodal is out of scope for the fabric pattern).
# 50 steps at the smallest debugmodel flavor finishes in ~5 min.
TRACE_DIR="$OUT_DIR/tier_b_trace"
mkdir -p "$TRACE_DIR"
TORCHTITAN="$WORKSPACE_DIR/torchtitan"
cd "$TORCHTITAN"
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=COLL,INIT \
NCCL_DEBUG_FILE="$TRACE_DIR/nccl-rank-%h-%p.log" \
TORCH_NCCL_TRACE_BUFFER_SIZE=20000 \
PHASE7_PROFILE_DIR="$TRACE_DIR" \
PHASE7_PROFILE_STEPS=50 \
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train \
    --job.config_file <(echo "config = '${MODULE}.${FLAVOR}'") \
    --training.steps 50 \
    --parallelism.pipeline_parallel_degree "$PP" \
    --parallelism.pipeline_parallel_schedule Interleaved1F1B \
    --parallelism.pipeline_parallel_layers_per_stage 2 \
    --parallelism.data_parallel_shard_degree "$FSDP" \
    --parallelism.data_parallel_replicate_degree "$DP_REP" \
    --parallelism.tensor_parallel_degree "$TP" \
    --parallelism.context_parallel_degree "$CP" \
    --parallelism.expert_parallel_degree "$EP" \
    --metrics.log_freq 1 \
    --dump_folder "$OUT_DIR" \
    || echo "[$(date)] trace run failed (NCCL logs may be partial)"

# Post-process: extract collective summary, flows, ixia json.
echo "[$(date)] post-process..."
python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/extract_collectives.py" "$TRACE_DIR" || true
python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/expand_to_flows.py" "$TRACE_DIR" || true
python "$WORKSPACE_DIR/phase7_nccl_traffic_catalog/flows_to_ixia.py" "$TRACE_DIR" --mode aggregated || true

# Compress + delete raw logs (immediately, see DISK_DISCIPLINE).
gzip -f "$TRACE_DIR/collective_summary.csv" 2>/dev/null
gzip -f "$TRACE_DIR/flows.csv" 2>/dev/null
rm -f "$TRACE_DIR"/nccl-rank-*.log

# recipe.json captures the mesh config alongside the trace.
cat > "$OUT_DIR/recipe.json" <<EOF
{
  "mode": "$MODE",
  "axis_count": "$AXIS_COUNT",
  "module": "$MODULE",
  "flavor": "$FLAVOR",
  "ngpu": 8,
  "pp": $PP, "dp_replicate": $DP_REP, "fsdp": $FSDP,
  "cp": $CP, "tp": $TP, "ep": $EP,
  "dataset": "c4_text",
  "trace_steps": 50
}
EOF

echo "[$(date)] 5D fabric trace DONE — see $TRACE_DIR/ixia_config.json"
