#!/usr/bin/env bash
# Full-N VQA eval (GQA/MMBench/POPE core 3) on a DCP ckpt, /home paths, NGPU=2.
# Usage: run_vqa_eval.sh <ckpt_dir> <tag>
set -uo pipefail
REPO=/home/torchtitan_attention_residual
SD=$REPO/phase5_vlm_multimodal_sft/eval_benchmarks
CKPT="${1:?ckpt dir}"; TAG="${2:-eval}"
export HF_HOME=/home/.hf_home
export TORCHDYNAMO_DISABLE=1
export STUDENT_CONFIG=kimi_linear_447m_aligned_block_attn_res_n4   # non-fp8, matches training
export NGPU=2
export INSTRUCT_DIR=/home/.hf_home/LLaVA-Instruct
export CACHE_DIR=/home/.hf_home
# point scorers at /home eval_data
export GQA_DIR=/home/.hf_home/eval_data/gqa
export MMB_DIR=/home/.hf_home/eval_data/mmbench/en
export POPE_DIR=/home/.hf_home/eval_data/pope
export SQA_DIR=/home/.hf_home/eval_data/scienceqa
export MMMU_DIR=/home/.hf_home/eval_data/mmmu/data
export BENCHES="${BENCHES:-mmbench pope gqa}"   # core 3 first (have baseline anchors)
export GQA_LIMIT=0 MMB_LIMIT=0 POPE_LIMIT=0     # full N
export STAGE2_CKPT="$CKPT"
export RUN_DIR=$REPO/phase5_vlm_multimodal_sft/eval_benchmarks/runs/${TAG}
mkdir -p "$RUN_DIR"
cd "$REPO"
echo "[eval:$TAG] ckpt=$CKPT benches=$BENCHES"
bash "$SD/run_all_evals.sh"
echo "EVAL_DONE_$TAG rc=$?"
