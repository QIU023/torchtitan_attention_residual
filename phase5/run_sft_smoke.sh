#!/usr/bin/env bash
# SFT run: 447M aligned base + LLaVA-Pretrain + held-out val loss.
# 8-GPU pure 1D FSDP. All knobs overridable via env (smoke / verify / full run).
set -euo pipefail
cd /workspace/torchtitan_attention_residual

export STUDENT_CONFIG=kimi_linear_447m_aligned_block_attn_res_n4
export STUDENT_CKPT=/workspace/torchtitan_attention_residual/phase4/runs/lm_447m_base/checkpoint/step-12500
export DATA_DIR=/workspace/.hf_home/LLaVA-Pretrain
export JSON=/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
export IMAGES=/workspace/.hf_home/LLaVA-Pretrain
export NGPU="${NGPU:-8}"
export STEPS="${STEPS:-100}"
export LOCAL_BS="${LOCAL_BS:-4}"
export GLOBAL_BS="${GLOBAL_BS:-64}"
export MAX_NORM="${MAX_NORM:-1.0}"
export VAL_FREQ="${VAL_FREQ:-50}"
export VAL_SAMPLES="${VAL_SAMPLES:-512}"
export VAL_BATCHES="${VAL_BATCHES:-24}"
export SAVE_FREQ="${SAVE_FREQ:-100000}"
export KEEP_K="${KEEP_K:-2}"
export OUT_DIR="${OUT_DIR:-/workspace/torchtitan_attention_residual/phase5/runs/mm_sft_smoke}"

exec bash phase5/launch_train.sh
