#!/usr/bin/env bash
# Sequential overnight pipeline:
#   Stage 1: Continue VLM pretrain from step-2500 → step-7500 (~3-4h)
#   Stage 2: VLM SFT 3 epoch on LLaVA-Instruct-150K (~2.5h)
#   Stage 3: VLM GRPO with kl_coef + grad_clip (~2h)
#
# Each stage's output dir is short ASCII (≤30 chars). Each stage
# reads from the previous stage's checkpoint.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"

ORCH_LOG="$WS/phase11_rlhf_grpo_infra/run_overnight_pipeline.log"
exec >>"$ORCH_LOG" 2>&1

echo "==============================================================="
echo "[$(date)] FULL PIPELINE START"
echo "==============================================================="

# ---------- Stage 1: VLM pretrain extend 2500 -> 7500 ----------
S1_OUT="$WS/phase5_vlm_multimodal_sft/runs/vlm_447m_pretrain"
S1_TARGET_STEP=7500

if [[ ! -d "$S1_OUT/checkpoint/step-2500" ]]; then
    echo "[$(date)] STAGE 1 ABORT: missing $S1_OUT/checkpoint/step-2500"
    exit 1
fi

echo "[$(date)] STAGE 1: VLM pretrain $S1_OUT step-2500 -> step-${S1_TARGET_STEP}"

# Retry loop — MoE expert routing CUDA assert is data-driven and
# hits ~once per few thousand steps. Each attempt resumes from latest
# saved checkpoint (SAVE_FREQ=500), bumping SEED to dodge the
# offending sample ordering.
S1_MAX_RETRIES=10
s1_attempt=0
while [[ $s1_attempt -lt $S1_MAX_RETRIES ]]; do
    s1_attempt=$((s1_attempt + 1))
    s1_seed=$((42 + s1_attempt - 1))
    echo "[$(date)] STAGE 1 attempt #$s1_attempt seed=$s1_seed"

    OUT_DIR="$S1_OUT" \
    RESUME=1 \
    FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
    PP_MICROBATCH=8 \
    STEPS="$S1_TARGET_STEP" LOCAL_BS=16 GLOBAL_BS=128 SEQ_LEN=260 \
    FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
    STUDENT_CKPT="$S1_OUT/checkpoint/step-2500" \
    SEED="$s1_seed" DETERMINISTIC=0 COMPILE=0 \
    LR=1e-5 WARMUP=200 \
    CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=3 \
    TRACE_TIER= \
    bash "$WS/phase6_upstream_pr_prep/launch_8gpu_mm.sh"

    s1_rc=$?
    last_step=$(grep -aoE "step:\s*[0-9]+" "$S1_OUT/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    last_step="${last_step:-0}"
    echo "[$(date)] STAGE 1 attempt #$s1_attempt done rc=$s1_rc last_step=$last_step"

    if [[ "$last_step" -ge $((S1_TARGET_STEP - 100)) ]]; then
        break
    fi
    sleep 30
done

if [[ "$last_step" -lt $((S1_TARGET_STEP - 100)) ]]; then
    echo "[$(date)] STAGE 1 INCOMPLETE after $s1_attempt attempts — abort"
    exit 2
fi

# ---------- Stage 2: VLM SFT 3 epoch ----------
S2_OUT="$WS/phase5_vlm_multimodal_sft/runs/vlm_447m_sft_3ep"
S2_DCP="$S1_OUT/checkpoint/step-${last_step}"
S2_STEPS=7000  # ~3 epochs of LLaVA-Instruct-150K @ GBS=32 SEQ=512

mkdir -p "$S2_OUT"
echo "[$(date)] STAGE 2: VLM SFT 3-epoch from $S2_DCP -> $S2_OUT"

# Stage 2 retry loop. First attempt loads from stage1 ckpt
# (initial_load_path); subsequent retries auto-resume from S2_OUT
# checkpoint via RESUME=1.
S2_MAX_RETRIES=10
s2_attempt=0
s2_last_step=0
while [[ $s2_attempt -lt $S2_MAX_RETRIES ]]; do
    s2_attempt=$((s2_attempt + 1))
    s2_seed=$((43 + s2_attempt - 1))
    if [[ -d "$S2_OUT/checkpoint" ]] && \
       ls "$S2_OUT/checkpoint" 2>/dev/null | grep -q "step-"; then
        s2_resume_arg="RESUME=1"
        echo "[$(date)] STAGE 2 attempt #$s2_attempt seed=$s2_seed (RESUME from $S2_OUT)"
    else
        s2_resume_arg=""
        echo "[$(date)] STAGE 2 attempt #$s2_attempt seed=$s2_seed (FRESH from $S2_DCP)"
    fi

    eval "PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OUT_DIR='$S2_OUT' \
$s2_resume_arg \
FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
STEPS='$S2_STEPS' LOCAL_BS=4 GLOBAL_BS=64 SEQ_LEN=512 \
MM_GLOBAL_SEQ_LEN=512 MM_LAYOUT=sft \
FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
STUDENT_CKPT='$S2_DCP' \
SEED='$s2_seed' DETERMINISTIC=0 COMPILE=0 \
LR=2e-5 WARMUP=100 PROJ_LR_MULT=50.0 \
CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=3 \
AC=full \
JSON='${LLAVA_INSTRUCT_JSON:-/workspace/.hf_home/LLaVA-Instruct-150K/llava_instruct_150k.json}' \
IMAGES='${LLAVA_INSTRUCT_IMG:-/workspace/.hf_home/coco_train2017/train2017}' \
TRACE_TIER= \
bash '$WS/phase6_upstream_pr_prep/launch_8gpu_mm.sh'"

    s2_rc=$?
    s2_last_step=$(grep -aoE "step:\s*[0-9]+" "$S2_OUT/train.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9]+")
    s2_last_step="${s2_last_step:-0}"
    echo "[$(date)] STAGE 2 attempt #$s2_attempt done rc=$s2_rc last_step=$s2_last_step"

    if [[ "$s2_last_step" -ge $((S2_STEPS - 200)) ]]; then
        break
    fi
    sleep 30
done

if [[ "$s2_last_step" -lt $((S2_STEPS - 200)) ]]; then
    echo "[$(date)] STAGE 2 INCOMPLETE after $s2_attempt attempts — abort"
    exit 3
fi

# ---------- Stage 3a: DCP -> HF VLM conversion ----------
S3_DCP="$S2_OUT/checkpoint/step-${s2_last_step}"
S3_HF="$WS/phase11_rlhf_grpo_infra/hf/vlm_sft_3ep"

echo "[$(date)] STAGE 3a: DCP->HF VLM conversion $S3_DCP -> $S3_HF"
mkdir -p "$S3_HF"
PYTHONPATH="$WS/torchtitan:$WS" \
    torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
        --in "$S3_DCP" \
        --out "$S3_HF" \
        --config kimi_linear_447m_aligned_block_attn_res_n4 \
        --vision-tower google/siglip-base-patch16-224
s3a_rc=$?
if [[ "$s3a_rc" -ne 0 ]]; then
    echo "[$(date)] STAGE 3a FAILED rc=$s3a_rc — abort"
    exit 4
fi
echo "[$(date)] STAGE 3a done"

# ---------- Stage 3b: VLM GRPO with KL + clipping ----------
S3_OUT="$WS/phase11_rlhf_grpo_infra/rlhf/outputs/grpo_llava_kimi_3ep"
mkdir -p "$S3_OUT"
echo "[$(date)] STAGE 3b: GRPO 1500 steps (kl_coef=0.05) -> $S3_OUT"

SGLANG_DISABLE_SHM_MM=1 \
ATTNRES_MLA_FP32_FALLBACK=1 \
PYTHONPATH="$WS/torchtitan:$WS" \
    python phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
        --dcp-load-path "$S3_DCP" \
        --hf-model-path "$S3_HF" \
        --num-steps 1500 --num-episodes-per-step 4 --kl-coef 0.05 \
        > "$S3_OUT/run.log" 2>&1
s3b_rc=$?
echo "[$(date)] STAGE 3b done rc=$s3b_rc"
echo "[$(date)] FULL PIPELINE COMPLETE"
