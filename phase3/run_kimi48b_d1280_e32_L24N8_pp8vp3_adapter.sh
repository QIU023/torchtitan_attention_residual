#!/usr/bin/env bash
# Kimi Linear 48B-layout shrunk to L=24 N=8 (paper sweet spot 3 t-blocks
# per AttnRes-block), dim=1280, num_experts=32.
# PP=8 × VP=3 = 24 chunks × 1 layer/chunk. ADAPTER mode, 300 steps.
#
# Hardware: 8× RTX 5090 PCIe. Memory headroom plan: smoke at FSDP=8
# showed 22.54 GiB/rank at d1280_e32 L=27 seq=2048 FSDP-only. PP=8 +
# adapter cache should fit similar or better (no FSDP shard memory,
# but full per-rank state for the rank's PP-stage layers).
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3/run_kimi48b_d1280_e32_L24N8_pp8vp3_adapter.log"
> "$LOG"
exec >>"$LOG" 2>&1

OUT="$WS/phase3/runs/kimi48b_d1280_e32_L24N8_pp8vp3_adapter_$(date +%Y%m%d-%H%M%S)"
rm -rf "$OUT"

echo "==============================================================="
echo "[$(date)] kimi 48B-layout d1280 e32 L24 N8 PP=8 VP=3 adapter START"
echo "==============================================================="

# Adapter on: TORCHTITAN_ATTNRES_CACHE=1
# n_layers=24, layers_per_stage=1, PP*VP=24 = 24 stages.
# LBS=24 (>= PP*VP). seq=2048 (matches smoke that fit).
(cd torchtitan && \
 env TORCHTITAN_ATTNRES_CACHE=1 ATTNRES_DBG=0 \
     PYTORCH_ALLOC_CONF="expandable_segments:True" \
     torchrun \
         --nproc_per_node=8 \
         --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
         --local-ranks-filter 7 --role rank --tee 3 \
         -m torchtitan.train \
         --module kimi_linear --config kimi_linear_48b_block_attn_res_d1280_e32_L24_N8 \
         --training.steps 300 \
         --training.local_batch_size 24 \
         --training.global_batch_size 24 \
         --training.seq_len 1024 \
         --parallelism.pipeline_parallel_degree 8 \
         --parallelism.pipeline_parallel_schedule Interleaved1F1B \
         --parallelism.pipeline_parallel_layers_per_stage 1 \
         --parallelism.pipeline_parallel_first_stage_less_layers 0 \
         --parallelism.pipeline_parallel_last_stage_less_layers 0 \
         --checkpoint.no-enable \
         --dump_folder "$OUT") 2>&1 | tail -200

echo ""
echo "==============================================================="
echo "[$(date)] DONE — out dir: $OUT"
echo "==============================================================="
