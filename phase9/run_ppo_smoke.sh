#!/usr/bin/env bash
# Phase 9-B PPO smoke (vLLM-free) — fabric trace runner.
#
# 8-GPU run. Captures the unique RLHF fabric pattern: actor (ranks 0-3)
# + ref (ranks 4-7) on disjoint sub-meshes, with cross-mesh KL exchange
# via world_pg broadcast/reduce. 50 steps; tier_b NCCL trace at all
# steps because the fabric is small per step.
set -u

WS=/root/torchtitan_attention_residual
OUT="$WS/phase5/runs/ppo_smoke_no_vllm/tier_b_trace"
rm -rf "$OUT"; mkdir -p "$OUT"

cd "$WS"
NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="$OUT/nccl-rank-%h-%p.log" \
NCCL_DEBUG_SUBSYS=COLL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    phase9/ppo_smoke_no_vllm.py \
    > "$OUT/run.log" 2>&1
rc=$?
echo "exit=$rc"
echo "log: $OUT/run.log"
ls "$OUT"/nccl-rank-*.log 2>/dev/null | wc -l
