#!/usr/bin/env bash
# Phase 10 Stage J — autoregressive-style inference fabric (two modes).
# Captures fabric for "growing prefix" (no cache) + "single token decode"
# (idealized cache) modes side-by-side.
set -uo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

WS=/root/torchtitan_attention_residual
TT=$WS/torchtitan
CKPT=$WS/phase4_kimi_attnres_lm_pretrain/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: ckpt $CKPT missing"; exit 1
fi

export PYTHONPATH="$WS:$TT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

run_mode() {
    local mode=$1
    local OUT=$WS/phase5_vlm_multimodal_sft/runs/inference_autoregressive_${mode}
    local TRACE=$OUT/tier_b_trace
    mkdir -p "$TRACE"
    rm -f "$TRACE"/nccl-rank-*.log

    echo "================================================================"
    echo "[autoregress] mode=$mode -> $OUT"
    echo "================================================================"

    NCCL_DEBUG=INFO \
    NCCL_DEBUG_FILE="$TRACE/nccl-rank-%h-%p.log" \
    NCCL_DEBUG_SUBSYS=COLL \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    torchrun \
        --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        -m phase10_ckpt_dcp_to_hf.inference_autoregressive \
        --ckpt "$CKPT" \
        --mode "$mode" \
        --n-generations 20 \
        --n-tokens 20 \
        --prompt-len 64 \
        --micro-bs 2 \
        2>&1 | tee "$OUT/run.log"
    echo "[autoregress] $mode exit=$?"

    cd "$WS"
    python phase7_nccl_traffic_catalog/extract_collectives.py "$TRACE" 2>&1 | tail -10
    python phase7_nccl_traffic_catalog/expand_to_flows.py "$TRACE" 2>&1 | tail -5
    python phase7_nccl_traffic_catalog/flows_to_ixia.py "$TRACE" 2>&1 | tail -5
    gzip -9 "$TRACE"/nccl-rank-*.log "$TRACE"/collective_summary.csv "$TRACE"/flows.csv 2>&1
    rm -f "$TRACE"/nccl-rank-*.log.gz
    du -sh "$OUT"
}

run_mode growing
run_mode single_token

echo
echo "================================================================"
echo "Both modes complete. Compare:"
echo "  $WS/phase5_vlm_multimodal_sft/runs/inference_autoregressive_growing/tier_b_trace/"
echo "  $WS/phase5_vlm_multimodal_sft/runs/inference_autoregressive_single_token/tier_b_trace/"
echo "================================================================"
