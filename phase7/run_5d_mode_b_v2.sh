#!/usr/bin/env bash
# Phase 7 5D MODE=B (v2) — DSv3 debugmodel CP+FSDP+PP fabric trace.
#
# 8-GPU constraint: TP doesn't generate fabric traffic (intra-node SHM)
# so a "true 5D" with all 5 axes >= 2 needs >= 16 GPUs anyway. With 8
# we pick 3 fabric axes >= 2 to maximize new-pattern coverage:
#
#   PP=2 x FSDP=2 x CP=2 = 8   (no EP, no TP)
#
# This adds **CP** (context-parallel ring exchange) to the catalog,
# which v11/v12/SFT do NOT have. Other axes already covered:
#   - PP+FSDP+TP+EP -> v11
#   - PP+FSDP+EP+dp_rep -> v12
#   - PP+FSDP+TP+EP -> SFT (post-train mesh)
set -u
cd /root/torchtitan_attention_residual/torchtitan

OUT=/tmp/phase7_5d_mode_b_v2
rm -rf "$OUT"; mkdir -p "$OUT"

NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="$OUT/nccl-rank-%h-%p.log" \
NCCL_DEBUG_SUBSYS=COLL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.context_parallel_degree 2 \
    --parallelism.tensor_parallel_degree 1 \
    --training.steps 50 \
    --training.local_batch_size 2 \
    --training.seq_len 512 \
    --metrics.log_freq 5 \
    > "$OUT/train.log" 2>&1
rc=$?
echo "exit=$rc"
echo "log: $OUT/train.log"
echo "nccl: $OUT/nccl-rank-*.log"
ls -la "$OUT" | head -20
