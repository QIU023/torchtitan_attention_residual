#!/usr/bin/env bash
# Phase 10 Stage L — production-volume inference workload sweep.
#
# 4 workload shapes captured back-to-back to characterize how fabric
# volume + pattern shape scales with (batch, seq_len, n_steps).
# Results in 4 ixia_config.json files for IXIA load comparisons.
set -uo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

WS=/root/torchtitan_attention_residual
TT=$WS/torchtitan
CKPT=$WS/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: ckpt $CKPT missing"; exit 1
fi

export PYTHONPATH="$WS:$TT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

run_workload() {
    local label=$1
    local bs=$2
    local seq=$3
    local n=$4
    local OUT=$WS/phase5/runs/workload_${label}
    local TRACE=$OUT/tier_b_trace
    mkdir -p "$TRACE"
    rm -f "$TRACE"/nccl-rank-*.log

    echo "================================================================"
    echo "[workload] $label : bs=$bs seq=$seq steps=$n -> $OUT"
    echo "================================================================"

    NCCL_DEBUG=INFO \
    NCCL_DEBUG_FILE="$TRACE/nccl-rank-%h-%p.log" \
    NCCL_DEBUG_SUBSYS=COLL \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    torchrun \
        --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        -m phase10.inference_torchtitan \
        --ckpt "$CKPT" \
        --n-steps "$n" \
        --seq-len "$seq" \
        --micro-bs "$bs" \
        2>&1 | tee "$OUT/run.log"
    local rc=$?
    echo "[workload] $label exit=$rc"
    if [[ "$rc" -ne 0 ]]; then return; fi

    cd "$WS"
    python phase7/extract_collectives.py "$TRACE" 2>&1 | tail -5
    python phase7/expand_to_flows.py "$TRACE" 2>&1 | tail -3
    python phase7/flows_to_ixia.py "$TRACE" 2>&1 | tail -5
    rm -f "$TRACE"/collective_summary.csv.gz "$TRACE"/flows.csv.gz
    gzip -9 "$TRACE"/nccl-rank-*.log "$TRACE"/collective_summary.csv "$TRACE"/flows.csv 2>&1
    rm -f "$TRACE"/nccl-rank-*.log.gz
    du -sh "$OUT"
}

# 4 production workload shapes:
#   short_high_bs : decode-like burst (BS=16, seq=256, 200 steps)
#   mid           : typical inference (BS=4, seq=1024, 200 steps)
#   long          : long-context (BS=2, seq=4096, 100 steps)
#   prod          : production-shape (BS=8, seq=2048, 100 steps)

run_workload short_high_bs 16 256 200
run_workload mid 4 1024 200
run_workload long 2 4096 100
run_workload prod 8 2048 100

echo
echo "================================================================"
echo "Sweep complete. ixia configs:"
for label in short_high_bs mid long prod; do
    f=$WS/phase5/runs/workload_${label}/tier_b_trace/ixia_config.json
    if [[ -f "$f" ]]; then
        sz=$(wc -c < "$f")
        echo "  $label: $f ($((sz/1024)) KB)"
    fi
done
df -BG --output=avail / | tail -1
echo "================================================================"
