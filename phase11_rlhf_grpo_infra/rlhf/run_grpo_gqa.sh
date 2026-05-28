#!/usr/bin/env bash
# GRPO on the stage2 LLaVA ckpt with the GQA VQA-accuracy task — a VERIFIABLE
# capability reward (exact-match short answer), unlike BLEU-on-LLaVA-Pretrain
# (degenerate: the model already trained on that data in stage1). General
# multimodal model (LLaVA/Kimi-linear), general VQA — NOT autonomous-driving.
#
# Usage:  bash run_grpo_gqa.sh [NUM_STEPS]   (default 1 = smoke)
set -euo pipefail
ulimit -c 0   # no core dumps (a crash here previously dumped 122G to core_pattern)
cd /workspace/torchtitan_attention_residual

NUM_STEPS="${1:-1}"

export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1

DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"

exec /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$DCP" \
    --hf-model-path "$HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task gqa \
    --num-steps "$NUM_STEPS" \
    --num-episodes-per-step 4 \
    --kl-coef 0.05
