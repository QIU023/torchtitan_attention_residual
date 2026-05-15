#!/usr/bin/env bash
# L16_n8 multi-shape PP/VP sweep — naive vs adapter numerical alignment.
#
# L16_n8 is the bf16-stable carrier proven this morning (loss descends
# from 11.76 → ~5.2 in 1000 steps without NaN). L32_n8 / L48_n8 with
# the same N=8 setting hit inf-grad at random init even with lower LR;
# documented as a separate stability investigation.
#
# Multi-shape stress on L16 stresses adapter's cross-stage cache
# management at different PP/VP geometries.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3_attnres_pp_integration/run_l16_multi_shape.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] L16 multi-shape sweep START"
echo "==============================================================="

# L16, num_blocks=8, layers_per_block=2. Shapes:
#   PP=8 VP=2: 16 chunks, 1 layer/chunk, LBS=16 GBS=16 DP=1
#   PP=4 VP=2: 8 chunks,  2 layers/chunk, LBS=8  GBS=16 DP=2
#   PP=4 VP=4: 16 chunks, 1 layer/chunk, LBS=16 GBS=32 DP=2
RUN_NAIVE=1 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L16_n8:8:2:16:16 \
175m_attn_res_L16_n8:4:2:8:16 \
175m_attn_res_L16_n8:4:4:16:32" \
STEPS=1000 \
bash "$WS/phase3_attnres_pp_integration/run_pp_pressure_test.sh"

echo "[$(date)] L16 multi-shape sweep DONE"
