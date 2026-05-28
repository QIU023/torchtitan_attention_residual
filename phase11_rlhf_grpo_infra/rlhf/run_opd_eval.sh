#!/usr/bin/env bash
# Stage D-2 post-run: DCP→HF convert the OPD step-50 ckpt, then GQA eval.
# Compares against the 12.3% SFT baseline (from the earlier
# gqa_eval.py run on stage2_447m_step5200 HF).
#
# Usage:  bash run_opd_eval.sh [N_EVAL]   (default 300, matches baseline)
set -euo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

N_EVAL_DEFAULT="${1:-300}"
export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export HF_HOME=/workspace/.hf_home

DCP_STEP="${DCP_STEP:-50}"
DCP_IN="${PWD}/phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts/step-${DCP_STEP}"
HF_OUT="${PWD}/phase11_rlhf_grpo_infra/hf/opd_step${DCP_STEP}"
SRC_HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"

if [[ ! -d "$DCP_IN" ]]; then
    echo "FATAL: DCP not found at $DCP_IN — did the OPD run save?" >&2
    exit 1
fi

echo "=== [1/2] DCP→HF convert ==="
echo "    in:  $DCP_IN"
echo "    out: $HF_OUT"
torchrun --nproc_per_node=1 phase11_rlhf_grpo_infra/dcp_to_hf_kimi_attn_res_vl.py \
    --in "$DCP_IN" \
    --out "$HF_OUT" \
    --config kimi_linear_447m_aligned_block_attn_res_n4 \
    --vision-tower google/siglip-base-patch16-224 \
    --processor-source "$SRC_HF" \
    --projector-from-hf "$SRC_HF"

echo
echo "=== [2/2] GQA testdev greedy eval ==="
echo "    HF:  $HF_OUT"
echo "    N:   $N_EVAL_DEFAULT"
echo "    BASELINE (SFT step-5200): 37/300 = 12.3%"
echo
export HF_MODEL_PATH="$HF_OUT"
export N_EVAL="$N_EVAL_DEFAULT"
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_DISABLE_SHM_MM=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
exec /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/gqa_eval.py
