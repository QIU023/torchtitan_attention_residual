#!/usr/bin/env bash
# Phase 3: 4-GPU plain Llama3 baseline launch (no AttnRes, no cache adapter).
#
# Drop-in no-AttnRes counterpart to launch_4gpu_{naive,adapter}.sh. Runs
# the exact same PP=4 V=2 slicing (layers_per_stage=2) against a plain
# 16-layer Llama3 dense flavor (llama3_175m_baseline_L16) that is
# shape-identical to llama3_175m_attn_res_L16_n8 minus the AttnRes
# pseudo-queries and norms. Purpose: establish the no-AttnRes reference
# curve at PP=4 scale so every future AttnRes-under-PP claim has a
# matched control.
#
# Default steps (60000) target a ~20h overnight run on 4x RTX 5090 PCIe.
# Throughput math (with LOCAL_BS=GLOBAL_BS=4, matching the cloud's
# session-4 4-GPU config for direct comparability with the handoff
# loss table):
#
#   tokens / step = GLOBAL_BS * seq_len = 4 * 2048 = 8192
#   plain-dense tps @ PP=4 V=2 on 4x 5090 PCIe ≈ 6.5-9 k (estimate,
#     scaled from AttnRes PP=4 measured ~6.8 k by the dense/AttnRes
#     ratio observed on single GPU: 70.7 k / 49.4 k ≈ 1.43x)
#   step time @ 8192 tok/step and tps 8000 ≈ 1.02 s
#   60000 steps * 1.02 s ≈ 17 h (leaves buffer for warmup +
#     checkpointing + early iteration slowdown)
#   60000 * 8192 ≈ 491 M tokens total
#     = 6.5 x (for 75.5 M non-embed Chinchilla target of 1.5 B tokens)
#     = 0.14 x (for 174 M total-params Chinchilla 3.5 B).
#
# If you have more time or want higher pipeline utilization (current
# M = GLOBAL_BS / LOCAL_BS = 1 microbatch, versus PP=4 V=2's 8 virtual
# stages -- significant bubble), bump GLOBAL_BS to 16 or 32. Throughput
# usually goes UP with larger batch under PP (less bubble). Keep
# LOCAL_BS=4 if memory-constrained, so grad_accum = GLOBAL_BS / LOCAL_BS.
#
# Override STEPS and/or GLOBAL_BS for smoke or longer runs.
#
# Usage:
#   bash phase3_attnres_pp_integration/launch_4gpu_baseline_L16.sh
#
# Env overrides (same vars as launch_4gpu_naive.sh):
#   STEPS=19000            steps (default 19000; overnight Chinchilla target)
#   LOCAL_BS=4             per-device micro-batch
#   GLOBAL_BS=4            global batch (grad_accum = GLOBAL_BS / LOCAL_BS)
#   OUT_DIR=...            output dir (default phase3_attnres_pp_integration/runs/pp4_baseline_L16_4gpu)
#   NGPU=4                 per-node GPU count

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/pp4_baseline_L16_4gpu}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-60000}"
# CONFIG override is supported but defaults to the baseline-L16 flavor.
# Overriding to attn_res / adapter flavors here would defeat the purpose;
# use the dedicated launch_4gpu_{naive,adapter}.sh for those.
CONFIG="${CONFIG:-llama3_175m_baseline_L16}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-4}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

# Belt and suspenders: even though the baseline model has no AttnRes
# blocks for the cache adapter to hook, unset the env flag explicitly
# so a stale export from a prior adapter run can't accidentally flip
# pipeline_llm_with_cache_adapter into wrap-mode on us.
unset TORCHTITAN_ATTNRES_CACHE

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 3 --role rank --tee 3 \
    -m torchtitan.train \
    --module attention_residual --config "${CONFIG}" \
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
