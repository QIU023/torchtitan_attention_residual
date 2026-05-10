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

ORCH_LOG="$WS/phase11/run_overnight_pipeline.log"
exec >>"$ORCH_LOG" 2>&1

echo "==============================================================="
echo "[$(date)] FULL PIPELINE START"
echo "==============================================================="

# ---------- Stage 1: VLM pretrain extend 2500 -> 7500 ----------
S1_OUT="$WS/phase5/runs/vlm_447m_pretrain"
S1_TARGET_STEP=7500

if [[ ! -d "$S1_OUT/checkpoint/step-2500" ]]; then
    echo "[$(date)] STAGE 1 ABORT: missing $S1_OUT/checkpoint/step-2500"
    exit 1
fi

echo "[$(date)] STAGE 1: VLM pretrain $S1_OUT step-2500 -> step-${S1_TARGET_STEP}"

OUT_DIR="$S1_OUT" \
RESUME=1 \
FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
PP_MICROBATCH=8 \
STEPS="$S1_TARGET_STEP" LOCAL_BS=16 GLOBAL_BS=128 SEQ_LEN=260 \
FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
STUDENT_CKPT="$S1_OUT/checkpoint/step-2500" \
SEED=42 DETERMINISTIC=0 COMPILE=0 \
LR=1e-5 WARMUP=200 \
CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=3 \
TRACE_TIER= \
bash "$WS/phase6/launch_8gpu_mm.sh"

s1_rc=$?
last_step=$(grep -oE "step:\s*[0-9]+" "$S1_OUT/train.log" 2>/dev/null \
    | tail -1 | grep -oE "[0-9]+")
last_step="${last_step:-0}"
echo "[$(date)] STAGE 1 done rc=$s1_rc last_step=$last_step"

if [[ "$last_step" -lt $((S1_TARGET_STEP - 100)) ]]; then
    echo "[$(date)] STAGE 1 INCOMPLETE — abort pipeline"
    exit 2
fi

# ---------- Stage 2: VLM SFT 3 epoch ----------
S2_OUT="$WS/phase5/runs/vlm_447m_sft_3ep"
S2_DCP="$S1_OUT/checkpoint/step-${last_step}"
S2_STEPS=7000  # ~3 epochs of LLaVA-Instruct-150K @ GBS=32 SEQ=512

mkdir -p "$S2_OUT"
echo "[$(date)] STAGE 2: VLM SFT 3-epoch from $S2_DCP -> $S2_OUT"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OUT_DIR="$S2_OUT" \
FSDP=8 DP_REP=1 PP=1 TP=1 CP=1 EP=1 V=1 ADAPTER=0 \
STEPS="$S2_STEPS" LOCAL_BS=4 GLOBAL_BS=64 SEQ_LEN=512 \
MM_GLOBAL_SEQ_LEN=512 MM_LAYOUT=sft \
FLAVOR=kimi_linear_447m_aligned_block_attn_res_n4 \
STUDENT_CKPT="$S2_DCP" \
SEED=43 DETERMINISTIC=0 COMPILE=0 \
LR=2e-5 WARMUP=100 PROJ_LR_MULT=50.0 \
CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=3 \
AC=full \
JSON="${LLAVA_INSTRUCT_JSON:-/workspace/.hf_home/LLaVA-Instruct-150K/llava_instruct_150k.json}" \
IMAGES="${LLAVA_INSTRUCT_IMG:-/workspace/.hf_home/coco_train2017/train2017}" \
TRACE_TIER= \
bash "$WS/phase6/launch_8gpu_mm.sh"

s2_rc=$?
s2_last_step=$(grep -oE "step:\s*[0-9]+" "$S2_OUT/train.log" 2>/dev/null \
    | tail -1 | grep -oE "[0-9]+")
s2_last_step="${s2_last_step:-0}"
echo "[$(date)] STAGE 2 done rc=$s2_rc last_step=$s2_last_step"

if [[ "$s2_last_step" -lt $((S2_STEPS - 200)) ]]; then
    echo "[$(date)] STAGE 2 INCOMPLETE — abort pipeline"
    exit 3
fi

# ---------- Stage 3: GRPO with KL + grad clip ----------
# Skipped automatically — needs HF conversion of the new SFT ckpt
# (DCP -> safetensors) via phase10/dcp_to_hf_kimi_attn_res.py before
# SGLang Engine can load it. Conversion is per-VLM and not yet
# scripted as part of this pipeline. Leave a hint for the operator.
S3_DCP="$S2_OUT/checkpoint/step-${s2_last_step}"
echo "[$(date)] STAGE 3: GRPO setup needed:"
echo "  python phase10/dcp_to_hf_kimi_attn_res.py \\"
echo "      --src $S3_DCP \\"
echo "      --dst $WS/phase11/hf/vlm_sft_3ep"
echo "  then ATTNRES_MLA_FP32_FALLBACK=1 SGLANG_DISABLE_SHM_MM=1 \\"
echo "    PYTHONPATH=\$WS/torchtitan:\$WS python phase11/rlhf/run_grpo_llava_kimi.py \\"
echo "    --dcp-load-path $S3_DCP \\"
echo "    --hf-model-path $WS/phase11/hf/vlm_sft_3ep \\"
echo "    --num-steps 1500 --num-episodes-per-step 4 --kl-coef 0.05"
echo "[$(date)] PIPELINE STAGES 1+2 DONE"
