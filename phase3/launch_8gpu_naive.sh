#!/usr/bin/env bash
# Phase 3: 8-GPU naive PP launch (no caching adapter).
#
# Runs AttnRes Llama3 on 8 GPUs with PP=8, VP=2, FSDP inner,
# interleaved 1F1B schedule. The naive path sends the full growing
# stacked_blocks tensor between stages -- measure step time + comm
# volume as the "before" number before enabling the adapter.
#
# Usage (from workspace root, env already activated):
#   bash phase3/launch_8gpu_naive.sh
#
# Override steps for a shorter smoke:
#   STEPS=100 bash phase3/launch_8gpu_naive.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/pp8_naive}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-1000}"
CONFIG="${CONFIG:-llama3_175m_attn_res_L16_n8}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-4}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

# Unset the cache flag explicitly so the adapter is OFF for this run
# (the adapter wrapper is no-op when this is not "1").
unset TORCHTITAN_ATTNRES_CACHE

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 7 --role rank --tee 3 \
    -m torchtitan.train \
    --module attn_res --config "${CONFIG}" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --parallelism.pipeline_parallel_degree 8 \
    --parallelism.pipeline_parallel_schedule "Interleaved1F1B" \
    --parallelism.pipeline_parallel_layers_per_stage 1 \
    --parallelism.pipeline_parallel_first_stage_less_layers 0 \
    --parallelism.pipeline_parallel_last_stage_less_layers 0 \
    --dump_folder "${OUT_DIR}" \
    --metrics.save_tb_folder tb \
    2>&1 | tee "${OUT_DIR}/train.log"
