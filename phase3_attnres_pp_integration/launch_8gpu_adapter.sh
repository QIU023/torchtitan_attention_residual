#!/usr/bin/env bash
# Phase 3: 8-GPU PP launch WITH the cross-stage caching adapter.
#
# Same config as the naive run, plus ``TORCHTITAN_ATTNRES_CACHE=1`` so
# the adapter at phase3_attnres_pp_integration/adapter.py intercepts stage forward/backward and
# caches prior-stage blocks locally. Expected: identical loss curve to
# the naive run within bf16 tolerance, measurably smaller per-stage
# send/recv bytes in steady state.
#
# Wiring is already done: torchtitan/experiments/attn_res/__init__.py
# registers `pipeline_llm_with_cache_adapter` as the ModelSpec
# pipelining_fn (see pipeline_adapter.py). Core torchtitan is untouched.
# The env flag just toggles whether the custom pipelining_fn wraps stages
# with the adapter or passes through to core pipeline_llm.
#
# Usage:
#   bash phase3_attnres_pp_integration/launch_8gpu_adapter.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/pp8_adapter}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-1000}"
CONFIG="${CONFIG:-llama3_175m_attn_res_L16_n8}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-4}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

export TORCHTITAN_ATTNRES_CACHE=1
export ATTNRES_DBG=1

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 7 --role rank --tee 3 \
    -m torchtitan.train \
    --module attention_residual --config "${CONFIG}" \
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
