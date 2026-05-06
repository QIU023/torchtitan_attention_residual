#!/usr/bin/env bash
# Phase 10 Stage K — two-phase RS+AG fabric injected into real-model inference.
set -uo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

WS=/root/torchtitan_attention_residual
TT=$WS/torchtitan
CKPT=$WS/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000
OUT=$WS/phase5/runs/inference_two_phase_real
TRACE=$OUT/tier_b_trace
mkdir -p "$TRACE"
rm -f "$TRACE"/nccl-rank-*.log

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: ckpt $CKPT missing"; exit 1
fi

export PYTHONPATH="$WS:$TT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="$TRACE/nccl-rank-%h-%p.log" \
NCCL_DEBUG_SUBSYS=COLL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun \
    --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m phase10.inference_two_phase_real \
    --ckpt "$CKPT" \
    --n-steps 50 \
    --seq-len 512 \
    --micro-bs 4 \
    2>&1 | tee "$OUT/run.log"

cd "$WS"
python phase7/extract_collectives.py "$TRACE" 2>&1 | tail -10
python phase7/expand_to_flows.py "$TRACE" 2>&1 | tail -5
python phase7/flows_to_ixia.py "$TRACE" 2>&1 | tail -5
gzip -9 "$TRACE"/nccl-rank-*.log "$TRACE"/collective_summary.csv "$TRACE"/flows.csv 2>&1
rm -f "$TRACE"/nccl-rank-*.log.gz
du -sh "$OUT"
