#!/usr/bin/env bash
# Auto-trigger post-SFT pipeline.
#
# Polls every 60s for:
#   * The orchestrator log to write a "phase 11 447M SFT DONE" line
#     (success criteria: last_step >= 1100 per the orchestrator's
#     gate, see run_sft_447m_llava_instruct_150k.sh)
#   * No live phase5.train_mm processes
#
# When both conditions hold, fires post_sft_vlm_smoke.sh which runs
# DCP -> HF safetensors -> SGLang Engine boot smoke -> 10-sample
# qualitative eval. Output captured to phase11/auto_post_sft.log.
#
# Idempotent — re-running it after success is a no-op (the smoke
# script overwrites the HF output dir but is otherwise safe).
set -u

WS=/root/torchtitan_attention_residual
LOG="$WS/phase11/auto_post_sft.log"
ORCH_LOG="$WS/phase11/sft_447m_orchestrator.log"

exec >>"$LOG" 2>&1

echo
echo "[$(date)] auto_post_sft starting (poll every 60s)"

while true; do
    sleep 60
    # SFT still running?
    if pgrep -f phase5.train_mm >/dev/null 2>&1; then
        continue
    fi
    # Orchestrator done flag?
    if ! grep -q "phase 11 447M SFT DONE" "$ORCH_LOG" 2>/dev/null; then
        continue
    fi
    # Did we complete enough steps?
    last_step=$(grep -aoE "step:\s*[0-9]+" "$WS/phase5/runs/sft_v_fsdp8_447m_aligned_llava_instruct_150k/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step=${last_step:-0}
    if (( last_step < 1100 )); then
        echo "[$(date)] orchestrator says DONE but last_step=$last_step < 1100; not firing"
        continue
    fi
    echo "[$(date)] SFT complete at step=$last_step. Firing post_sft_vlm_smoke.sh"
    bash "$WS/phase11/post_sft_vlm_smoke.sh"
    rc=$?
    echo "[$(date)] post_sft_vlm_smoke.sh rc=$rc"
    break
done

echo "[$(date)] auto_post_sft EXIT"
