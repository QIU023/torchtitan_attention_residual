#!/usr/bin/env bash
# Chain: PP adapter-only sweep -> resume SFT continuation -> Stage 3 GRPO.
# Naive PP comparison runs deferred to tomorrow per user request.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase11_rlhf_grpo_infra/run_pp_then_sft_grpo.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] CHAIN START"
echo "==============================================================="

# ---------- Stage A: PP adapter-only sweep (3 configs × ~10min) ----------
echo "[$(date)] Stage A: PP adapter pressure test"
RUN_NAIVE=0 RUN_ADAPTER=1 bash "$WS/phase3_attnres_pp_integration/run_pp_pressure_test.sh"
echo "[$(date)] Stage A done"

# Find the latest pressure_test dir to reference results
PP_OUT=$(ls -dt "$WS"/phase3_attnres_pp_integration/runs/pressure_test_* 2>/dev/null | head -1)
if [[ -n "$PP_OUT" ]]; then
    echo "[$(date)] PP sweep summary at $PP_OUT/SUMMARY.md"
fi

# ---------- Stage B: SFT continuation from step-5500 -> 7000 ----------
echo "[$(date)] Stage B: SFT 3ep continuation"
bash "$WS/phase11_rlhf_grpo_infra/run_stage2_continuation.sh"
echo "[$(date)] Stage B done"

# ---------- Stage C: GRPO + KL on the final SFT ckpt ----------
echo "[$(date)] Stage C: GRPO via run_stage3.sh"
bash "$WS/phase11_rlhf_grpo_infra/run_stage3.sh"
echo "[$(date)] Stage C done"

echo "[$(date)] CHAIN COMPLETE"
