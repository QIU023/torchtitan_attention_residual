#!/usr/bin/env bash
# Phase 10 Stage F — Real PPO smoke + tier_b NCCL trace.
set -uo pipefail

if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    source /venv/main/bin/activate
fi

WS=/root/torchtitan_attention_residual
TT=$WS/torchtitan
# SFT step-490 is the natural starting point for RLHF (post-SFT actor).
# Falls back to phase4 step-8000 if SFT ckpt missing.
CKPT="${CKPT:-$WS/phase5/runs/sft_v11_llava_instruct_150k_4d/checkpoint/step-490}"
if [[ ! -d "$CKPT" ]]; then
    CKPT="$WS/phase4/runs/kimi_436m_block_attn_res_fsdp/checkpoint/step-8000"
    echo "WARNING: SFT ckpt not found, using phase4 step-8000 instead"
fi
OUT=$WS/phase5/runs/ppo_real_torchtitan
TRACE_DIR=$OUT/tier_b_trace
mkdir -p "$TRACE_DIR"
rm -f "$TRACE_DIR"/nccl-rank-*.log

export PYTHONPATH="$WS:$TT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="$TRACE_DIR/nccl-rank-%h-%p.log" \
NCCL_DEBUG_SUBSYS=COLL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun \
    --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m phase10.ppo_real_torchtitan \
    --ckpt "$CKPT" \
    --n-steps 50 \
    --seq-len 256 \
    --micro-bs 2 \
    --lr 1e-6 \
    --kl-coef 0.05 \
    2>&1 | tee "$OUT/run.log"
rc=$?
echo "exit=$rc"
ls "$TRACE_DIR"/nccl-rank-*.log 2>/dev/null | wc -l
