#!/usr/bin/env bash
# Phase 10 Stage I — two-phase TP fabric pattern demo.
# Runs both AllReduce baseline and RS+AG two-phase variants with separate
# tier_b NCCL traces, then runs the standard pipeline -> ixia_config.json
# for each.
set -uo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

WS=/root/torchtitan_attention_residual
PY=$WS/phase10/two_phase_tp_smoke.py

run_mode() {
    local mode=$1
    local OUT=$WS/phase5/runs/two_phase_tp_${mode}
    local TRACE=$OUT/tier_b_trace
    mkdir -p "$TRACE"
    rm -f "$TRACE"/nccl-rank-*.log

    echo "================================================================"
    echo "[smoke] mode=$mode -> $OUT"
    echo "================================================================"

    NCCL_DEBUG=INFO \
    NCCL_DEBUG_FILE="$TRACE/nccl-rank-%h-%p.log" \
    NCCL_DEBUG_SUBSYS=COLL \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    torchrun \
        --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        "$PY" \
        --mode "$mode" \
        --n-steps 50 \
        --per-attn-msg-mb 12 \
        --n-attn-per-step 16 \
        2>&1 | tee "$OUT/run.log"
    echo "[smoke] $mode exit=$?"

    cd "$WS"
    python phase7/extract_collectives.py "$TRACE" 2>&1 | tail -8
    python phase7/expand_to_flows.py "$TRACE" 2>&1 | tail -5
    python phase7/flows_to_ixia.py "$TRACE" 2>&1 | tail -5
    gzip -9 "$TRACE"/nccl-rank-*.log "$TRACE"/collective_summary.csv "$TRACE"/flows.csv 2>&1
    rm "$TRACE"/nccl-rank-*.log.gz
    du -sh "$OUT"
}

run_mode allreduce
run_mode rs_ag

echo
echo "================================================================"
echo "Both modes complete. Compare:"
echo "  $WS/phase5/runs/two_phase_tp_allreduce/tier_b_trace/ixia_config.json"
echo "  $WS/phase5/runs/two_phase_tp_rs_ag/tier_b_trace/ixia_config.json"
echo "================================================================"
