#!/usr/bin/env bash
# Full AttnRes L=32 N=32 PP=8 × VP=4 sweep — paper-aligned aggressive
# pressure test. Naive (Full AttnRes worst-case wire bytes) and
# adapter (cached, O(1) per hop) at 1000 steps each.
#
# Stability rationale: Block AttnRes at L=32 dim=768 inf-grads at
# random init (standard residual accumulates without bound across
# the 4 intra-block layers). Full AttnRes (N = n_layers) replaces
# every residual with a softmax aggregator; at zero-init pseudo-
# query, output is the uniform mean of preceding sources, which is
# bounded by max-source magnitude. Confirmed in a 30-step smoke
# (grad_norm 4.3e17 → 7.2e5, loss 11.76 → 10.20) — finite and
# converging.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3/run_l32n32_pp8vp4.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] L32_n32 Full AttnRes PP=8 × VP=4 sweep START"
echo "==============================================================="

# PP=8 × VP=4 → 32 virtual stages × 1 layer/stage. LBS = PP*VP = 32.
# DP = NGPU/PP = 8/8 = 1, so GBS = LBS × 1 = 32.
RUN_NAIVE=1 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L32_n32:8:4:32:32" \
STEPS=1000 \
bash "$WS/phase3/run_pp_pressure_test.sh"

echo "[$(date)] L32_n32 Full AttnRes PP=8 × VP=4 sweep DONE"
