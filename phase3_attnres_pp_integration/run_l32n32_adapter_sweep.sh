#!/usr/bin/env bash
# Full AttnRes L=32 N=32 ADAPTER-ONLY sweep across PP × VP shapes.
# Strategy: run adapter for all shapes first, commit + push, then
# come back to run naive in a separate session for alignment.
#
# Why adapter first: the cache adapter is the new code under test.
# If it survives 5 shapes without diverging or hanging, we have
# confidence to commit. Naive baseline is the validation pair
# (it's old code, lower-risk).
#
# All shapes at L=32 N=32 satisfy n_layers % (PP*VP) == 0 and
# LBS >= PP*VP. GBS = LBS * DP * k.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3_attnres_pp_integration/run_l32n32_adapter_sweep.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] L32_n32 Full AttnRes ADAPTER-ONLY sweep START"
echo "==============================================================="

# Shapes (config:PP:VP:LBS:GBS). PP=8 only per user — PP=4 not needed
# (we're testing aggressive PP stacking, the point is the deep PP path).
#   PP=8 × VP=4: 32 chunks × 1 layer, LBS=32, GBS=32, DP=1  ← paper-aligned headline
#   PP=8 × VP=2: 16 chunks × 2 layer, LBS=16, GBS=16, DP=1
RUN_NAIVE=0 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L32_n32:8:4:32:32 \
175m_attn_res_L32_n32:8:2:16:16" \
STEPS=300 \
bash "$WS/phase3_attnres_pp_integration/run_pp_pressure_test.sh"

echo ""
echo "==============================================================="
echo "[$(date)] L32_n32 Full AttnRes ADAPTER sweep DONE"
echo "==============================================================="
