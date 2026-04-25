#!/usr/bin/env bash
# Phase 4: PP=4 + FSDP Kimi Linear launcher with AttnRes cache adapter.
#
# Default target: kimi_linear_436m_block_attn_res
# (paper-exact 436M from Table 2: d_model=1168, d_ff=528, lr=2.20e-3,
# batch=384, n_layers=16. L=16 is paper-native at this scale and
# divides the 8 virtual stages PP=4 V=2 lps=2 Interleaved1F1B
# requires — every block boundary lines up with a stage boundary.)
#
# Schedule: Interleaved1F1B (prerequisite for the cache adapter).
# Adapter: ON by default (TORCHTITAN_ATTNRES_CACHE=1).
#
# Usage:
#   bash phase4/launch_pp4_kimi.sh                # default 100-step smoke
#   STEPS=1000 LOCAL_BS=2 bash phase4/launch_pp4_kimi.sh
#
# Env overrides:
#   CONFIG=kimi_linear_436m_block_attn_res  (default, paper-exact 436M, L=16)
#           kimi_linear_436m_baseline       (no AttnRes, same L=16)
#           kimi_linear_528m_l16_block_attn_res  (528M with L massaged to 16)
#           kimi_linear_528m_l16_full_attn_res   (Full AttnRes, N=L=16)
#   STEPS=100       smoke
#   LOCAL_BS=1      per-device micro-batch
#   GLOBAL_BS=4     global batch (= num_microbatches when PP=4 V=2 lps=2
#                   means num_stages=8, so GLOBAL_BS>=8 is ideal to keep
#                   pipeline full; for smoke GLOBAL_BS=4 is OK)
#   SEQ_LEN=2048    training context (paper uses 8192; 2048 for smoke
#                   memory headroom on 4× 5090 PCIe)
#   CACHE=1         TORCHTITAN_ATTNRES_CACHE (1 = adapter ON)
#   NGPU=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"

MODULE="${MODULE:-kimi_linear}"
CONFIG="${CONFIG:-kimi_linear_436m_block_attn_res}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-100}"
LOCAL_BS="${LOCAL_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-}"
CACHE="${CACHE:-1}"
OUT_SUFFIX="${OUT_SUFFIX:-$(echo "${CONFIG}" | tr '[:upper:]' '[:lower:]')_pp4}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/${OUT_SUFFIX}}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

if [[ "${CACHE}" = "1" ]]; then
    export TORCHTITAN_ATTNRES_CACHE=1
else
    unset TORCHTITAN_ATTNRES_CACHE
fi

EXTRA_ARGS=()
if [[ -n "${LR}" ]]; then
    EXTRA_ARGS+=(--optimizer.lr "${LR}")
fi

# Validation hooks (mirror launch_fsdp_small.sh)
VAL="${VAL:-0}"
if [[ "${VAL}" = "1" ]]; then
    EXTRA_ARGS+=(--validator.enable)
    EXTRA_ARGS+=(--validator.freq "${VAL_FREQ:-2500}")
    EXTRA_ARGS+=(--validator.steps "${VAL_STEPS:-100}")
fi

if [[ -n "${EXTRA_ARGS_APPEND:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(${EXTRA_ARGS_APPEND})
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 3 --role rank --tee 3 \
    -m torchtitan.train \
    --module "${MODULE}" --config "${CONFIG}" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --parallelism.pipeline_parallel_degree 4 \
    --parallelism.pipeline_parallel_schedule "Interleaved1F1B" \
    --parallelism.pipeline_parallel_layers_per_stage 2 \
    --parallelism.pipeline_parallel_first_stage_less_layers 0 \
    --parallelism.pipeline_parallel_last_stage_less_layers 0 \
    --parallelism.data_parallel_shard_degree 1 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    "${EXTRA_ARGS[@]}" \
    --dump_folder "${OUT_DIR}" \
    --metrics.save_tb_folder tb \
    2>&1 | tee "${OUT_DIR}/train.log"
