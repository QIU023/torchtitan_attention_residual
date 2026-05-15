#!/usr/bin/env bash
# Kimi Linear 48B-layout: L=32 N=8 (4 t-blocks/AttnRes-block, 1.33× paper),
# dim=1280, num_experts=32, seq=1024.
# PP=8 × VP=4 = 32 chunks × 1 layer/chunk. ADAPTER mode, 300 steps.
#
# Companion to L=24 N=8 PP=8 VP=3 run — same carrier shape but pushed to
# PP=8 × VP=4 (deeper VP, denser AttnRes block-cache traffic on adapter).
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3_attnres_pp_integration/run_kimi48b_d1280_e32_L32N8_pp8vp4_adapter.log"
> "$LOG"
exec >>"$LOG" 2>&1

OUT="$WS/phase3_attnres_pp_integration/runs/kimi48b_d1280_e32_L32N8_pp8vp4_adapter_$(date +%Y%m%d-%H%M%S)"
rm -rf "$OUT"

echo "==============================================================="
echo "[$(date)] kimi 48B-layout d1280 e32 L32 N8 PP=8 VP=4 adapter START"
echo "==============================================================="

# Adapter on. L=32 PP=8 → 4 layers/PP-rank (vs 3 at L=24).
# LBS = PP*VP = 32. seq=1024.
(cd torchtitan && \
 env TORCHTITAN_ATTNRES_CACHE=1 ATTNRES_DBG=0 \
     PYTORCH_ALLOC_CONF="expandable_segments:True" \
     torchrun \
         --nproc_per_node=8 \
         --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
         --local-ranks-filter 7 --role rank --tee 3 \
         -m torchtitan.train \
         --module kimi_linear --config kimi_linear_48b_block_attn_res_d1280_e32_L32_N8 \
         --training.steps 300 \
         --training.local_batch_size 32 \
         --training.global_batch_size 32 \
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
