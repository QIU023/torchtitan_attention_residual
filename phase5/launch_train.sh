#!/usr/bin/env bash
# Phase 5 multimodal full-parameter fine-tune launcher.
# AttnRes-Kimi-436M (Phase 4 step-12500 ckpt) + frozen SigLIP-Base
# + trainable MLP projector, end-to-end on LLaVA-Pretrain-558K.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

# ---- knobs ----
STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_436m_block_attn_res_n4}"
STUDENT_CKPT="${STUDENT_CKPT:-${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500}"
DATA_DIR="${DATA_DIR:-/root/hf_cache/LLaVA-Pretrain}"
JSON="${JSON:-${DATA_DIR}/blip_laion_cc_sbu_558k.json}"
IMAGES="${IMAGES:-${DATA_DIR}}"   # LLaVA-Pretrain ZIP extracts to bucket dirs (00000/...) directly under DATA_DIR, not into ./images/
VISION="${VISION:-google/siglip-base-patch16-224}"
TOKENIZER="${TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}"
CACHE_DIR="${CACHE_DIR:-/root/hf_cache}"

STEPS="${STEPS:-30000}"
LOCAL_BS="${LOCAL_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-32}"
SEQ_LEN="${SEQ_LEN:-260}"   # 196 vision + ~60 text + bos/eos
LR="${LR:-1e-5}"            # full-param fine-tune from already-trained ckpt: small LR
PROJ_LR_MULT="${PROJ_LR_MULT:-50.0}"  # projector starts random — needs much higher LR
MAX_NORM="${MAX_NORM:-1.0}"  # explicit grad-clip (torchtitan default also 1.0).
                              # SFT v11 4D mesh 2026-05-05 saw chronic grad_norm 40k-80k
                              # which clipping CAN'T rescue — keep this <=1.0 and lower
                              # PROJ_LR_MULT if loss stays stuck.
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/mm_full_finetune}"
NGPU="${NGPU:-4}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-1000}"

if [[ ! -d "${STUDENT_CKPT}" ]]; then
    echo "ERROR: student ckpt not found: ${STUDENT_CKPT}" >&2; exit 1
fi
if [[ ! -f "${JSON}" ]]; then
    echo "ERROR: caption json not found: ${JSON}" >&2
    echo "Run python phase5/data_prep.py first." >&2
    exit 1
fi
if [[ ! -d "${IMAGES}" ]]; then
    echo "ERROR: images dir not found: ${IMAGES}" >&2; exit 1
fi

mkdir -p "${OUT_DIR}"

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5.train_mm \
    --mm.json "${JSON}" \
    --mm.images "${IMAGES}" \
    --mm.vision-model "${VISION}" \
    --mm.tokenizer "${TOKENIZER}" \
    --mm.cache-dir "${CACHE_DIR}" \
    --mm.proj-lr-mult "${PROJ_LR_MULT}" \
    --module kimi_linear --config "${STUDENT_CONFIG}" \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
    --training.max_norm "${MAX_NORM}" \
    --optimizer.lr "${LR}" \
    --lr_scheduler.warmup_steps 200 \
    --lr_scheduler.total_steps "${STEPS}" \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree "${NGPU}" \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.initial_load_path "${STUDENT_CKPT}" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval "${SAVE_FREQ}" \
    --checkpoint.keep_latest_k 2 \
    --metrics.save_tb_folder tb \
    --dump_folder "${OUT_DIR}" \
    --compile.enable \
    2>&1 | tee "${OUT_DIR}/train.log"
