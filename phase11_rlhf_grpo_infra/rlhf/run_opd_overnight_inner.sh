#!/usr/bin/env bash
# Inner launcher used by run_opd_overnight.sh. Same shape as
# run_opd_50step.sh but parametrised via env vars so the orchestrator
# can drive smoke vs full-run with a single binary.
set -euo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

NUM_STEPS="${1:-1}"
export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
TEACHER_ID="llava-hf/llama3-llava-next-8b-hf"
TOKENIZER_PATH="$HF"
CKPT_DIR="${PWD}/phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts"

CKPT_INTERVAL="${CKPT_INTERVAL:-100}"
NEPS="${NEPS:-2}"
OPD_LR="${OPD_LR:-1e-5}"
OPD_WD="${OPD_WD:-0.01}"
OPD_BETA="${OPD_BETA:-0.5}"
OPD_T="${OPD_T:-2.0}"
OPD_TASK_TYPE="${OPD_TASK_TYPE:-vqa}"

exec /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$DCP" \
    --hf-model-path "$HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task opd \
    --teacher-model-id "$TEACHER_ID" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --num-steps "$NUM_STEPS" \
    --num-episodes-per-step "$NEPS" \
    --opd-beta "$OPD_BETA" \
    --opd-temperature "$OPD_T" \
    --opd-lr "$OPD_LR" \
    --opd-weight-decay "$OPD_WD" \
    --opd-task-type "$OPD_TASK_TYPE" \
    --opd-ckpt-interval "$CKPT_INTERVAL" \
    --opd-ckpt-dir "$CKPT_DIR" \
    --kl-coef 0.0
