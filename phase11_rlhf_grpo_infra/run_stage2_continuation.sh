#!/usr/bin/env bash
# Resume Stage 2 SFT from step-5000 with bumped retry cap.
# Stage 1 already done (step-7500 was the input; consumed and deleted
# to free disk). Stage 2 reached step 5000 (2.13 epochs) before
# hitting MAX_RETRIES=10. Continue toward step 7000.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"

LOG="$WS/phase11_rlhf_grpo_infra/run_stage2_continuation.log"
exec >>"$LOG" 2>&1

echo "==============================================================="
echo "[$(date)] STAGE 2 CONTINUATION (target step 7000, max 30 retries)"
echo "==============================================================="

S2_OUT="$WS/phase5_vlm_multimodal_sft/runs/vlm_447m_sft_3ep"
S2_STEPS=7000
MAX_RETRIES=30
attempt=0
last_step=0

while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    seed=$((42 + attempt))
    echo "[$(date)] ATTEMPT #$attempt seed=$seed RESUME from $S2_OUT"

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    OUT_DIR="$S2_OUT" \
    RESUME=1 \
    FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
    STEPS="$S2_STEPS" LOCAL_BS=4 GLOBAL_BS=64 SEQ_LEN=512 \
    MM_GLOBAL_SEQ_LEN=512 MM_LAYOUT=sft \
    FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
    STUDENT_CKPT="$S2_OUT/checkpoint/step-5000" \
    SEED="$seed" DETERMINISTIC=0 COMPILE=0 \
    LR=2e-5 WARMUP=100 PROJ_LR_MULT=50.0 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=3 \
    AC=full \
    JSON=/workspace/.hf_home/LLaVA-Instruct-150K/llava_instruct_150k.json \
    IMAGES=/workspace/.hf_home/coco_train2017/train2017 \
    TRACE_TIER= \
    bash "$WS/phase6_upstream_pr_prep/launch_8gpu_mm.sh"

    rc=$?
    last_step=$(grep -aoE "step:\s*[0-9]+" "$S2_OUT/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step="${last_step:-0}"
    echo "[$(date)] ATTEMPT #$attempt done rc=$rc last_step=$last_step"
    if [[ "$last_step" -ge $((S2_STEPS - 200)) ]]; then
        echo "[$(date)] STAGE 2 TARGET REACHED at step $last_step"
        break
    fi
    sleep 30
done

if [[ "$last_step" -ge $((S2_STEPS - 200)) ]]; then
    echo "[$(date)] STAGE 2 SUCCESS at step $last_step"
    echo "[$(date)] Next: run phase11_rlhf_grpo_infra/run_stage3.sh (DCP->HF + GRPO)"
else
    echo "[$(date)] STAGE 2 INCOMPLETE after $attempt attempts; last step $last_step"
fi
