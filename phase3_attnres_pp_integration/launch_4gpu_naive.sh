#!/usr/bin/env bash
# Phase 3: 4-GPU naive PP launch (no caching adapter).
#
# Adaptation of launch_8gpu_naive.sh to a 4-physical-GPU environment.
# Runs PP=4 V=2 (8 virtual stages) on 4 GPUs via ``layers_per_stage=2``
# against the existing L16_n8 config (n_layers=16, num_blocks=8). Each
# rank owns 2 virtual stages, each stage spans exactly 2 layers = 1
# block, so every stage boundary is a block boundary and the
# cross-stage caching adapter is exercised at every stage transition.
#
# Why not 8 ranks on 4 GPUs via CUDA_VISIBLE_DEVICES=LOCAL_RANK%4:
# NCCL rejects duplicate GPUs across ranks ("Duplicate GPU detected"),
# even with NCCL_P2P_DISABLE and per-process CUDA_VISIBLE_DEVICES. PP=4
# V=2 keeps 1 process/GPU while still exercising the same-rank
# consumer/producer own-commit cache-read path that the _LocalCache*
# Functions target.
#
# Usage:
#   bash phase3_attnres_pp_integration/launch_4gpu_naive.sh
#
# Env overrides:
#   STEPS=50        quick smoke (default 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/pp4_naive_4gpu}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-50}"
CONFIG="${CONFIG:-llama3_175m_attn_res_L16_n8}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-4}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

unset TORCHTITAN_ATTNRES_CACHE

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 3 --role rank --tee 3 \
    -m torchtitan.train \
    --module attn_res --config "${CONFIG}" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --parallelism.pipeline_parallel_degree 4 \
    --parallelism.pipeline_parallel_schedule "Interleaved1F1B" \
    --parallelism.pipeline_parallel_layers_per_stage 2 \
    --parallelism.pipeline_parallel_first_stage_less_layers 0 \
    --parallelism.pipeline_parallel_last_stage_less_layers 0 \
    --dump_folder "${OUT_DIR}" \
    --metrics.save_tb_folder tb \
    2>&1 | tee "${OUT_DIR}/train.log"
