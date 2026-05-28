#!/usr/bin/env bash
# D-6: matched-encoder OPD with Mantis (SigLIP+Llama-3) instead of LLaVA-NeXT (CLIP).
#
# Hypothesis (from D-5 negative result REPORT_OPD_D5.md, H1):
#   The dominant failure mode of D-2..D-5 is the cross-VLM-encoder gap —
#   teacher's CLIP-ViT-L/14-336 produces vision features the student's
#   SigLIP-base-224 can't approximate. Token-level JSD then pulls the
#   student lm_head toward outputs the body can't ground → degeneration.
#
# This run tests H1 by replacing ONLY the teacher with one that shares
# the SigLIP family (so400m-384, 1152d) — same student, same task,
# same hyperparams as the best prior arm (D-4: 50 steps, caption, lr=1e-5).
#
# Other setup matches D-4 directly so we can read off the delta:
#   D-4 (CLIP-336 teacher): 9.3% GQA after 50 steps
#   D-6 (SigLIP-384 teacher): TARGET > 12.3% baseline; minimum > 9.3% D-4
set -euo pipefail
ulimit -c 0
cd /workspace/torchtitan_attention_residual

NUM_STEPS="${1:-50}"
export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export HF_HOME=/workspace/.hf_home
export ATTNRES_MLA_FP32_FALLBACK=1
export SGLANG_FP8_IGNORED_LAYERS="attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts"
export SGLANG_DISABLE_SHM_MM=1
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
TEACHER_ID="TIGER-Lab/Mantis-8B-siglip-llama3"     # ← D-6 change
TOKENIZER_PATH="$HF"
CKPT_DIR="${PWD}/phase11_rlhf_grpo_infra/rlhf/outputs/opd_d6_ckpts"

# Match D-4 hyperparams (the best prior arm: 50 steps caption gave 9.3%)
# but bump temperature to T=2.0 (standard KD) since we now have a matched
# encoder and a stronger signal can be absorbed.
exec /usr/bin/python3 phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "$DCP" \
    --hf-model-path "$HF" \
    --flavor kimi_linear_447m_aligned_block_attn_res_n4 \
    --task opd \
    --teacher-model-id "$TEACHER_ID" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --num-steps "$NUM_STEPS" \
    --num-episodes-per-step 2 \
    --opd-beta 0.5 \
    --opd-temperature 2.0 \
    --opd-lr 1e-5 \
    --opd-weight-decay 0.01 \
    --opd-task-type caption \
    --opd-ckpt-interval 25 \
    --opd-ckpt-dir "$CKPT_DIR" \
    --kl-coef 0.0
