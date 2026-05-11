#!/usr/bin/env bash
# Stage 3: DCP->HF VLM conversion + GRPO with KL.
# Picks up from whatever step-N is the latest in vlm_447m_sft_3ep/checkpoint.
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
LOG="$WS/phase11/run_stage3.log"
exec >>"$LOG" 2>&1

S2_OUT="$WS/phase5/runs/vlm_447m_sft_3ep"
LAST=$(ls "$S2_OUT/checkpoint" 2>/dev/null | grep -oE "step-[0-9]+" | sort -V | tail -1)
if [[ -z "$LAST" ]]; then
    echo "[$(date)] no SFT ckpt found"; exit 1
fi
S3_DCP="$S2_OUT/checkpoint/$LAST"
S3_HF="$WS/phase11/hf/vlm_sft_3ep"

echo "==============================================================="
echo "[$(date)] STAGE 3 START: DCP=$S3_DCP -> HF=$S3_HF"
echo "==============================================================="

# 3a: DCP -> HF
mkdir -p "$S3_HF"
PYTHONPATH="$WS/torchtitan:$WS" \
    torchrun --nproc_per_node=1 phase11/dcp_to_hf_kimi_attn_res_vl.py \
        --in "$S3_DCP" \
        --out "$S3_HF" \
        --config kimi_linear_447m_aligned_block_attn_res_n4 \
        --vision-tower google/siglip-base-patch16-224
echo "[$(date)] STAGE 3a done"

# 3a-eval: qualitative gate. Refuse to launch GRPO if VLM still
# produces EOS-trap garbage (the !!!! pattern from undertrained
# vision attention). Gate threshold: at least 6/10 samples must be
# coherent (start with letter, < 30% bang density).
EVAL_LOG="$S3_HF/qualitative_eval.log"
echo "[$(date)] STAGE 3a-eval: qualitative gate"
SGLANG_DISABLE_SHM_MM=1 ATTNRES_MLA_FP32_FALLBACK=1 \
PYTHONPATH="$WS/torchtitan:$WS" \
    python phase11/eval_sft_3ep_qualitative.py \
        --model-path "$S3_HF" \
        --num-samples 10 \
        --gate-threshold 0.6 \
        2>&1 | tee "$EVAL_LOG"
eval_rc=${PIPESTATUS[0]}
if [[ "$eval_rc" -ne 0 ]]; then
    echo "[$(date)] STAGE 3a-eval FAILED rc=$eval_rc — skipping GRPO"
    echo "[$(date)] See $EVAL_LOG for sample outputs"
    echo "[$(date)] Likely root cause: SFT undertrained for image grounding"
    echo "[$(date)] Recommendation: extend SFT or retrain with better recipe"
    exit 5
fi
echo "[$(date)] STAGE 3a-eval PASSED — proceeding to GRPO"

# 3b: GRPO
S3_OUT="$WS/phase11/rlhf/outputs/grpo_llava_kimi_3ep"
mkdir -p "$S3_OUT"
SGLANG_DISABLE_SHM_MM=1 ATTNRES_MLA_FP32_FALLBACK=1 \
PYTHONPATH="$WS/torchtitan:$WS" \
    python phase11/rlhf/run_grpo_llava_kimi.py \
        --dcp-load-path "$S3_DCP" \
        --hf-model-path "$S3_HF" \
        --num-steps 1500 --num-episodes-per-step 4 --kl-coef 0.05 \
        > "$S3_OUT/run.log" 2>&1
echo "[$(date)] STAGE 3b done"
echo "[$(date)] FULL STAGE 3 COMPLETE"
