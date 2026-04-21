#!/usr/bin/env bash
# Phase 3: 4-GPU PP launch WITH cross-stage caching adapter.
#
# Twin of launch_4gpu_naive.sh with TORCHTITAN_ATTNRES_CACHE=1. PP=4
# V=2 on 4 GPUs (layers_per_stage=2, n_layers=16 -> 8 virtual stages,
# 2 chunks per rank under Interleaved1F1B). Each rank owns 2 virtual
# stages and at least one of them reads a same-rank own-commit from
# the shared :class:`RankLocalCache` — the path that exercises the
# ``_LocalCacheAugment`` / ``_LocalCacheCapture`` Functions (see
# phase3/handoff_status_20260420_part3.md).
#
# Usage:
#   bash phase3/launch_4gpu_adapter.sh
#
# Env overrides:
#   STEPS=50        quick smoke (default 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/pp4_adapter_4gpu}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-50}"
CONFIG="${CONFIG:-llama3_175m_attn_res_L16_n8}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-4}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

export TORCHTITAN_ATTNRES_CACHE=1
export ATTNRES_DBG=1
# export ATTNRES_ADAPTER_DBG=1   # turn on for backward-call tracing

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
