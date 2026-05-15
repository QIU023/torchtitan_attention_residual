#!/usr/bin/env bash
# Widen-dim smoke for L=32 N=8 Block AttnRes. Find smallest dim where
# random-init forward stays bf16-finite at PP=8 × VP=4 = 32 chunks.
# 30 steps per dim, fast feedback.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3_attnres_pp_integration/run_l32n8_widen_smoke.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] L=32 N=8 widen-dim smoke START"
echo "==============================================================="

RUN_NAIVE=1 RUN_ADAPTER=0 \
SWEEP="attn_res_L32_n8_d1024:8:4:32:32 \
attn_res_L32_n8_d1280:8:4:32:32 \
attn_res_L32_n8_d1536:8:4:32:32 \
attn_res_L32_n8_d2048:8:4:32:32" \
STEPS=30 \
SWEEP_OUT_ROOT="$WS/phase3_attnres_pp_integration/runs/l32n8_widen_smoke_$(date +%Y%m%d-%H%M%S)" \
bash "$WS/phase3_attnres_pp_integration/run_pp_pressure_test.sh"

echo ""
echo "==============================================================="
echo "[$(date)] L=32 N=8 widen-dim smoke DONE"
echo "==============================================================="
