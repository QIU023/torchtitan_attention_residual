#!/usr/bin/env bash
# PP pressure test resumed after Stage C GRPO finishes.
# Carrier switched to L48_n8 = 48 layers / 8 blocks = 6 layers/block,
# matching Kimi Linear 48B's actual config from the AttnRes paper
# (Section 4, 27 transformer blocks × 2 = 54 layers / 9 blocks).
#
# Each shape runs adapter + naive (numerical alignment check).
# Lower LR + longer warmup to avoid inf-grad at random init for the
# deeper carrier — addresses today's L32/L48 inf-grad failures.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase3/run_pp_post_grpo.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] PP POST-GRPO PRESSURE TEST START"
echo "==============================================================="

# L48_n8: 6 layers/block (paper config)
# Shape constraints:
#   n_layers >= PP * VP  (LBS divisor)
#   LBS >= PP * VP       (n_microbatches)
#   GBS = LBS * DP * k   (DP = NGPU/PP)
# Aim for NaN-free: shorter run (300 steps), warmup 100, lr 5e-5.
RUN_NAIVE=1 RUN_ADAPTER=1 \
SWEEP="175m_attn_res_L48_n8:8:2:16:16 \
175m_attn_res_L48_n8:8:3:24:24 \
175m_attn_res_L48_n8:4:2:8:16 \
175m_attn_res_L48_n8:4:4:16:32" \
STEPS=300 \
LR=5e-5 \
WARMUP=100 \
bash "$WS/phase3/run_pp_pressure_test.sh"

echo "[$(date)] PP POST-GRPO DONE"
