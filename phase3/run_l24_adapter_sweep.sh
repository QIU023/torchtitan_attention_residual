#!/usr/bin/env bash
# L=24 Block AttnRes N-sweep, adapter-only, dim=768.
#
# Hypothesis: stability is driven by intra-block standard-residual chain
# length S = L / N. Smaller S (larger N) is more stable. Sweep N from
# largest (Full AttnRes, S=1) downward to find the smallest N (largest S)
# that still trains finite-grad at L=24 dim=768.
#
# Adapter and naive are numerically identical (adapter is a pure comm
# optimization), so the stability result holds for both.
#
# PP=8 × VP=3 = 24 virtual stages × 1 transformer-block per chunk.
# LBS=24 (≥ PP*VP), GBS=24 (DP=1 since PP=8).
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3/run_l24_adapter_sweep.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] L=24 N-sweep adapter PP=8 × VP=3 dim=768 START"
echo "==============================================================="

# N from 24 (Full, S=1) down to 2 (S=12). Each 30 steps.
RUN_NAIVE=0 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L24_n24:8:3:24:24 \
175m_attn_res_L24_n12:8:3:24:24 \
175m_attn_res_L24_n8:8:3:24:24 \
175m_attn_res_L24_n6:8:3:24:24 \
175m_attn_res_L24_n4:8:3:24:24 \
175m_attn_res_L24_n3:8:3:24:24 \
175m_attn_res_L24_n2:8:3:24:24" \
STEPS=30 \
SWEEP_OUT_ROOT="$WS/phase3/runs/l24_adapter_nsweep_$(date +%Y%m%d-%H%M%S)" \
bash "$WS/phase3/run_pp_pressure_test.sh"

echo "==============================================================="
echo "[$(date)] L=24 N-sweep adapter DONE"
echo "==============================================================="
