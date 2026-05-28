#!/usr/bin/env bash
# Stage D — 50-step OPD distillation on the real 447M Kimi student
# from llava-hf/llama3-llava-next-8b-hf teacher.
#
# Goal: prove the OPD pipeline actually DISTILLS, not just runs.
# Acceptance:
#   * loss curve monotonically (or near-monotonically) decreasing over
#     the last 20-30 steps.
#   * Sample completion at step 49 shows the student speaking
#     coherent English (not the un-distilled French/garbage from step 0).
#   * GQA testdev greedy acc lift vs the 12.3% SFT baseline (run
#     phase11_rlhf_grpo_infra/rlhf/gqa_eval.py after this completes,
#     using the trainer's final published weights).
#
# Expected runtime: 50 × ~180 s = ~150 min ≈ 2.5 h. The 180 s/step is
# dominated by SGLang weight-sync-via-disk; loss compute itself is
# ~5-20 s/step. weight_sync is a known GRPO bottleneck (orthogonal
# to OPD) so we accept it for this validation run.
#
# No checkpoint saving (feedback_no_ckpt_smoke_pressure): exploratory
# validation, not production. If the run crashes mid-stream, restart
# from the SFT ckpt and continue.
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
# Reduce CUDA allocator fragmentation — the OOM at step 1 of the
# first 50-step attempt showed 606 MiB reserved-but-unallocated on
# cuda:0. expandable_segments lets the allocator grow segments
# instead of reserving fixed blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DCP="${PWD}/phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200"
HF="${PWD}/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200"
TEACHER_ID="llava-hf/llama3-llava-next-8b-hf"
TOKENIZER_PATH="$HF"
# Stage D-2: ckpt the trainer DCP at half + end of run so we can convert
# back to HF and run gqa_eval against the distilled student.
CKPT_DIR="${PWD}/phase11_rlhf_grpo_infra/rlhf/outputs/opd_50step_ckpts"
CKPT_INTERVAL="${CKPT_INTERVAL:-25}"
# 4 episodes/step uses the cuda:0 headroom we got back from per-episode
# backward; 2 was leaving the trainer card half-idle outside backward.
NEPS="${NEPS:-4}"
# GKD hyperparams — defaults match the first 50-step run for direct compare.
OPD_BETA="${OPD_BETA:-0.5}"
OPD_T="${OPD_T:-1.0}"
# LR overrides the from-scratch default (8e-4). 1e-5 matches LLaMA-3
# continual-distill convention; can override with OPD_LR env if sweeping.
OPD_LR="${OPD_LR:-1e-5}"
OPD_WD="${OPD_WD:-0.01}"

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
    --opd-ckpt-interval "$CKPT_INTERVAL" \
    --opd-ckpt-dir "$CKPT_DIR" \
    --kl-coef 0.0
