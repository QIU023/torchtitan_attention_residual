#!/usr/bin/env bash
# GRPO on the LLaVA strict-2-stage SFT ckpt (stage2_instruct_sft_447m, step-5200,
# scp'd back 2026-05-27). Aligned from run_grpo_smoke_1h.sh:
#   - DCP  : stage2_instruct_sft_447m/checkpoint/step-5200  (the 17G/8-shard ckpt)
#   - HF   : phase11_rlhf_grpo_infra/hf/stage2_447m_step5200 (DCP->HF converted)
#   - flavor kimi_linear_447m_aligned_block_attn_res_n4  (ckpt was trained _n4;
#     run_grpo_llava_kimi.py defaults to the non-_n4 name -> MUST override or the
#     trainer skeleton dims mismatch the DCP).
#
# Usage:  bash run_grpo_stage2_step5200.sh [NUM_STEPS]   (default 1 = smoke)
set -euo pipefail
cd /workspace/torchtitan_attention_residual

NUM_STEPS="${1:-1}"   # default 1-step smoke; pass 500 for the real run

export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1

DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"

exec /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$DCP" \
    --hf-model-path "$HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --num-steps "$NUM_STEPS" \
    --num-episodes-per-step 4
