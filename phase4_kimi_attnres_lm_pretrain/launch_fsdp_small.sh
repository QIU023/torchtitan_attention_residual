#!/usr/bin/env bash
# Phase 4: FSDP-only single-node launcher for AttnRes pretraining.
#
# Targets: GH200 / 8× H100 / 4× RTX 5090 PCIe — any single-node setup where
# the model fits per-GPU with FSDP sharding. NO pipeline parallel (PP
# adapter is a separate story — see phase3_attnres_pp_integration/ and README notes).
#
# ------------------------------------------------------------------------
# Why FSDP-only (not PP):
#
# On NVLink-class interconnects (GH200 C2C ~450 GB/s, H100 NVLink ~450 GB/s,
# GB200 NVL ~900 GB/s), FSDP's all-gather + reduce-scatter are cheap
# relative to compute. PP's per-stage activation send/recv saves less than
# FSDP costs, AND PP introduces pipeline bubble + bookkeeping overhead. For
# models that fit single-node with FSDP, PP is an anti-pattern.
#
# Rough rule of thumb: if the model (param + activation + optim state) fits
# in (num_gpus * per_gpu_mem) with FSDP zero-3 sharding, skip PP.
# ------------------------------------------------------------------------
#
# What this launcher runs:
#
# Uses existing ``torchtitan/experiments/attn_res/`` flavors (Llama3
# backbone + AttnRes) via ``--module attention_residual --config <flavor>``. These
# are ModelSpec-complete and train end-to-end today. Once the Kimi Linear
# ModelSpec integration lands (Phase 4c: ``KimiLinearModel(BaseModel)``
# + ``KimiLinearConfig(BaseModel.Config)`` shim), swap to
# ``--module attention_residual --config kimi_linear_<size>_<variant>``.
#
# Scaling-law flavors available TODAY (Llama3 backbone):
#   llama3_175m_baseline              # 12L dense Llama3, no AttnRes
#   llama3_175m_attn_res              # 12L + Block AttnRes N=6
#   llama3_175m_attn_res_n2/n3/n4/n12 # AttnRes N sweep
#   llama3_175m_attn_res_L16_n8       # 16L, N=8 (Phase 3 adapter target)
#
# Scaling-law flavors PENDING Phase 4c (Kimi Linear backbone, KDA+MLA+MoE):
#   kimi_linear_194m_{baseline,block_attn_res,full_attn_res}
#   kimi_linear_241m_{...}
#   kimi_linear_296m_{...}
#   kimi_linear_436m_{...}
#   kimi_linear_528m_{...}
#   (see torchtitan/experiments/kimi_linear/config_registry.py)
#
# ------------------------------------------------------------------------
# Usage:
#
#   # Most basic: 8-GPU FSDP, default config, 1000 steps
#   bash phase4_kimi_attnres_lm_pretrain/launch_fsdp_small.sh
#
#   # Pick flavor + step count
#   CONFIG=llama3_175m_attn_res_L16_n8 STEPS=60000 \
#       bash phase4_kimi_attnres_lm_pretrain/launch_fsdp_small.sh
#
#   # Change batch / LR for a bigger rental box
#   CONFIG=llama3_175m_attn_res_L16_n8 NGPU=8 STEPS=60000 \
#       LOCAL_BS=16 GLOBAL_BS=128 LR=3e-4 \
#       bash phase4_kimi_attnres_lm_pretrain/launch_fsdp_small.sh
#
#   # Kimi Linear flavor (after Phase 4c ModelSpec integration lands):
#   MODULE=attention_residual CONFIG=kimi_linear_528m_block_attn_res \
#       STEPS=100000 bash phase4_kimi_attnres_lm_pretrain/launch_fsdp_small.sh
#
# ------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${SCRIPT_DIR}/../torchtitan}"

# ---- knobs ----
MODULE="${MODULE:-attention_residual}"                            # attn_res (today) | kimi_linear (Phase 4c)
CONFIG="${CONFIG:-llama3_175m_attn_res_L16_n8}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-1000}"
LOCAL_BS="${LOCAL_BS:-4}"
GLOBAL_BS="${GLOBAL_BS:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-}"  # empty => use config default
DP_SHARD="${DP_SHARD:-${NGPU}}"   # FSDP2 full shard across all GPUs by default
OUT_SUFFIX="${OUT_SUFFIX:-$(echo "${CONFIG}" | tr '[:upper:]' '[:lower:]')_fsdp}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/${OUT_SUFFIX}}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

mkdir -p "${OUT_DIR}"
echo "$(cd "${TORCHTITAN_DIR}" && git rev-parse --short HEAD)" > "${OUT_DIR}/GIT_SHA"

cd "${TORCHTITAN_DIR}"

# AttnRes cache adapter is a PP-only concept; explicit off for FSDP runs.
unset TORCHTITAN_ATTNRES_CACHE

# Compose optional overrides
EXTRA_ARGS=()
if [[ -n "${LR}" ]]; then
    EXTRA_ARGS+=(--optimizer.lr "${LR}")
fi

# COMPILE=1 turns on per-decoder-layer torch.compile + grouped_mm
# (the kimi_linear parallelize_fn handles the actual wrapping; this
# just flips torchtitan's compile.enable knob). On 4× RTX 5090 the
# combined effect on Kimi 436M is +46% throughput vs eager + ~15
# percentage points lower peak memory. Default on for kimi_linear
# since 4d landed; explicit COMPILE=0 reverts to eager.
COMPILE="${COMPILE:-1}"
if [[ "${COMPILE}" = "1" ]]; then
    EXTRA_ARGS+=(--compile.enable)
fi

# Validation — off by default to preserve legacy launcher behavior,
# but long runs should turn it on. Uses torchtitan's default
# c4_validation dataset (see torchtitan/components/validate.py).
#   VAL=1            enable
#   VAL_FREQ=2500    run validation every N training steps
#   VAL_STEPS=100    how many val batches to consume per check
VAL="${VAL:-0}"
if [[ "${VAL}" = "1" ]]; then
    EXTRA_ARGS+=(--validator.enable)
    EXTRA_ARGS+=(--validator.freq "${VAL_FREQ:-2500}")
    EXTRA_ARGS+=(--validator.steps "${VAL_STEPS:-100}")
fi

# EXTRA_ARGS_APPEND lets callers pass arbitrary extra flags (e.g.
# --lr_scheduler.total_steps 12500 for resume-with-preserved-schedule).
# Whitespace-split the variable; empty string means no-op.
if [[ -n "${EXTRA_ARGS_APPEND:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word-split
    EXTRA_ARGS+=(${EXTRA_ARGS_APPEND})
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module "${MODULE}" --config "${CONFIG}" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree "${DP_SHARD}" \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    "${EXTRA_ARGS[@]}" \
    --dump_folder "${OUT_DIR}" \
    --metrics.save_tb_folder tb \
    2>&1 | tee "${OUT_DIR}/train.log"
