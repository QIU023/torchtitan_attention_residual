#!/usr/bin/env bash
# Stage D — first OPD 1-step smoke on the real 447M Kimi student
# distilling from llava-hf/llama3-llava-next-8b-hf teacher.
#
# Goal: prove the end-to-end OPD pipeline (Stage C wiring) runs to a
# completed step on actual GPUs with real weights — NOT to converge.
#
# What this validates (boot blockers first):
#   1. Provisioner trainer bootstrap CVD expansion to "0,5,6,7" works
#   2. torch.cuda.device_count() returns 4 inside trainer process
#   3. HF accelerate spreads 8B teacher across logical cuda:1-3
#   4. SGLang generator boots on cuda:1-4 (TP=4)
#   5. Generator produces a non-trivial completion on a COCO image
#   6. teacher.score returns finite logits at response positions
#   7. compute_response_logits returns matching-shape student logits
#   8. opd_loss is finite
#   9. backward + optim.step completes without OOM / NaN
#
# Usage:  bash run_opd_smoke.sh [NUM_STEPS]   (default 1 = smoke)
set -euo pipefail
ulimit -c 0   # disable core dumps (Lance segfaults previously dumped 122G)
cd /workspace/torchtitan_attention_residual

NUM_STEPS="${1:-1}"

export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1
export TRL_EXPERIMENTAL_SILENCE=1

# Same student paths as run_grpo_stage2_step5200.sh / run_grpo_gqa.sh.
DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"

# Teacher: cached at /workspace/.hf_home/hub/models--llava-hf--llama3-llava-next-8b-hf/
# (4 safetensors, ~16G total, downloaded during Stage B).
TEACHER_ID="llava-hf/llama3-llava-next-8b-hf"

# Tokenizer for prompt/response decode in OPDTrainer.step. The student's
# HF dir has the tokenizer.json the SFT actually used (Llama-3 128256
# base vocab + 256 added).
TOKENIZER_PATH="$HF"

exec /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$DCP" \
    --hf-model-path "$HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task opd \
    --teacher-model-id "$TEACHER_ID" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --num-steps "$NUM_STEPS" \
    --num-episodes-per-step 2 \
    --kl-coef 0.0
