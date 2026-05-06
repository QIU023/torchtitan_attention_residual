#!/usr/bin/env bash
# Phase 10 — torchtitan forward-only inference with tier_b NCCL trace.
set -uo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

WS=/root/torchtitan_attention_residual
TT=$WS/torchtitan
CKPT=$WS/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000
OUT=$WS/phase5/runs/inference_torchtitan_phase4_step8000
TRACE_DIR=$OUT/tier_b_trace
mkdir -p "$TRACE_DIR"
rm -f "$TRACE_DIR"/nccl-rank-*.log

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: ckpt $CKPT missing"; exit 1
fi

export PYTHONPATH="$WS:$TT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="$TRACE_DIR/nccl-rank-%h-%p.log" \
NCCL_DEBUG_SUBSYS=COLL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun \
    --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m phase10.inference_torchtitan \
    --ckpt "$CKPT" \
    --n-steps 50 \
    --seq-len 512 \
    --micro-bs 4 \
    2>&1 | tee "$OUT/run.log"
rc=$?
echo "exit=$rc"
ls "$TRACE_DIR"/nccl-rank-*.log 2>/dev/null | wc -l
