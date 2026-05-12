#!/usr/bin/env bash
# Execute all remaining PP pressure tests:
#   (1) L16 pp4_vp4 adapter — fill-in for the 2026-05-11 12:20 sweep
#       where the adapter row truncated (naive crashed at DCP write).
#   (2) L48_n8 sweep — 4 shapes × 2 modes (naive + adapter), STEPS=300,
#       LR=5e-5 + WARMUP=100 (avoids inf-grad at random init).
#
# NO CHECKPOINT SAVING: run_pp_pressure_test.sh now passes
# --checkpoint.enable=false to override the config registry default.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3/run_all_pp_pressure.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] ALL PP PRESSURE TESTS START"
echo "==============================================================="

# Explicit per-step output roots so the two sweeps never collide even
# if they finish within the same minute.
TS="$(date +%Y%m%d-%H%M%S)"

# --- (1) L16 pp4_vp4 adapter fill-in ---
echo ""
echo "--- Step 1/2: L16 pp4_vp4 adapter fill-in ---"
SWEEP_OUT_ROOT="$WS/phase3/runs/pressure_test_${TS}_L16fill" \
RUN_NAIVE=0 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L16_n8:4:4:16:32" \
STEPS=1000 \
bash "$WS/phase3/run_pp_pressure_test.sh"

# --- (2) L48_n8 sweep ---
echo ""
echo "--- Step 2/2: L48 sweep (post-GRPO config) ---"
SWEEP_OUT_ROOT="$WS/phase3/runs/pressure_test_${TS}_L48sweep" \
RUN_NAIVE=1 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L48_n8:8:2:16:16 \
175m_attn_res_L48_n8:8:3:24:24 \
175m_attn_res_L48_n8:4:2:8:16 \
175m_attn_res_L48_n8:4:4:16:32" \
STEPS=300 \
LR=5e-5 \
WARMUP=100 \
bash "$WS/phase3/run_pp_pressure_test.sh"

echo ""
echo "==============================================================="
echo "[$(date)] ALL PP PRESSURE TESTS DONE"
echo "==============================================================="
