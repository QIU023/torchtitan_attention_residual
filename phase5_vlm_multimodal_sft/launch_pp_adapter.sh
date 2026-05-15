#!/usr/bin/env bash
# Phase 5 Arm 2: PP + cache-adapter cross-modality smoke.
#
# Tests whether the Phase-3-validated cache adapter preserves loss
# invariance under mixed vision+text sequences. By default targets
# PP=4 V=2 + Interleaved1F1B + TORCHTITAN_ATTNRES_CACHE=1, but can be
# scaled down via env vars (PP, V) for incremental smokes.

set -euo pipefail

# Activate the project venv if available so torchrun + torch are on PATH.
if [[ -f /venv/main/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

# ---- knobs ----
CONFIG="${CONFIG:-kimi_linear_436m_block_attn_res_n4}"
DATA_DIR="${DATA_DIR:-/root/hf_cache/LLaVA-Pretrain}"
JSON="${JSON:-${DATA_DIR}/blip_laion_cc_sbu_558k.json}"
IMAGES="${IMAGES:-${DATA_DIR}}"
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
CACHE_DIR="${CACHE_DIR:-/root/hf_cache}"

NGPU="${NGPU:-4}"
PP="${PP:-4}"
V="${V:-2}"
SCHEDULE="${SCHEDULE:-Interleaved1F1B}"
STEPS="${STEPS:-50}"
LOCAL_BS="${LOCAL_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-12}"   # >= V*PP=8 microbatches for Interleaved1F1B lookahead
SEQ_LEN="${SEQ_LEN:-260}"      # 196 vision + 60 caption + bos + eos = 258, round up
MM_GLOBAL_SEQ_LEN="${MM_GLOBAL_SEQ_LEN:-258}"
LR="${LR:-1e-5}"
PROJ_LR_MULT="${PROJ_LR_MULT:-50.0}"

ADAPTER="${ADAPTER:-1}"        # 1 = TORCHTITAN_ATTNRES_CACHE=1 ; 0 = naive PP
COMPILE_ARG=""
if [[ "${COMPILE:-0}" == "1" ]]; then
    COMPILE_ARG="--compile.enable"
fi

INIT="${INIT:-fresh}"          # fresh | weak_ckpt
INIT_CKPT="${INIT_CKPT:-}"
if [[ "$INIT" == "weak_ckpt" && -z "$INIT_CKPT" ]]; then
    echo "ERROR: INIT=weak_ckpt requires INIT_CKPT path" >&2
    exit 1
fi

SEED="${SEED:-42}"             # set to "" to omit
DETERMINISTIC="${DETERMINISTIC:-1}"  # 1 enables --debug.deterministic
DEBUG_ARGS=""
if [[ -n "${SEED}" ]]; then
    DEBUG_ARGS="${DEBUG_ARGS} --debug.seed ${SEED}"
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
    DEBUG_ARGS="${DEBUG_ARGS} --debug.deterministic"
fi

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/arm2_pp${PP}_v${V}_${INIT}_$( [[ "$ADAPTER" == "1" ]] && echo adapter || echo naive )}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${JSON}" ]]; then
    echo "ERROR: caption json not found: ${JSON}" >&2
    echo "Run python phase5_vlm_multimodal_sft/data_prep.py first." >&2
    exit 1
fi
if [[ ! -d "${IMAGES}" ]]; then
    echo "ERROR: images dir not found: ${IMAGES}" >&2; exit 1
fi

CKPT_ARGS=""
if [[ "$INIT" == "weak_ckpt" ]]; then
    CKPT_ARGS="--checkpoint.enable --checkpoint.initial_load_path ${INIT_CKPT} --checkpoint.initial_load_model_only"
fi

if [[ "$ADAPTER" == "1" ]]; then
    export TORCHTITAN_ATTNRES_CACHE=1
else
    unset TORCHTITAN_ATTNRES_CACHE
fi

export PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Log the rank that holds the LAST PP stage (where the loss is computed).
# For PP=N rank=N-1 is the last; for PP=1 rank 0 logs.
LAST_RANK="$(( PP - 1 ))"
LOG_RANK="${LOG_RANK:-${LAST_RANK}}"

torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
    -m phase5_vlm_multimodal_sft.train_mm \
    --mm.json "${JSON}" \
    --mm.images "${IMAGES}" \
    --mm.vision-model "${VISION}" \
    --mm.tokenizer "${TOKENIZER}" \
    --mm.cache-dir "${CACHE_DIR}" \
    --mm.proj-lr-mult "${PROJ_LR_MULT}" \
    --mm.global-seq-len "${MM_GLOBAL_SEQ_LEN}" \
    --module kimi_linear --config "${CONFIG}" \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --optimizer.lr "${LR}" \
    --lr_scheduler.warmup_steps 10 \
    --lr_scheduler.total_steps "${STEPS}" \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree "${PP}" \
    --parallelism.pipeline_parallel_schedule "${SCHEDULE}" \
    --parallelism.pipeline_parallel_layers_per_stage "${V}" \
    --parallelism.pipeline_parallel_first_stage_less_layers 0 \
    --parallelism.pipeline_parallel_last_stage_less_layers 0 \
    --parallelism.data_parallel_shard_degree "${FSDP:-1}" \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    ${CKPT_ARGS} \
    ${DEBUG_ARGS} \
    --metrics.save_tb_folder tb \
    --metrics.log_freq 1 \
    --dump_folder "${OUT_DIR}" \
    ${COMPILE_ARG} \
    2>&1 | tee "${OUT_DIR}/train.log"
