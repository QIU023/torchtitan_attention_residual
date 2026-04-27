#!/usr/bin/env bash
# Phase 5 KD training launcher.
#
# Distills Kimi-Linear-48B-A3B-Base into the 436M student initialized
# from the Phase 4 step-12500 ckpt. Reuses torchtitan's Trainer
# entirely (model build, FSDP, optim, scheduler, dataloader, ckpt) and
# only overrides forward_backward_step to swap CE for the KD loss.
#
# Override knobs (all environment-variable driven):
#   STUDENT_CONFIG  flavor name (default: kimi_linear_436m_block_attn_res_n4)
#   STUDENT_CKPT    DCP ckpt directory (default: phase4 step-12500)
#   TEACHER         HF repo id (default: moonshotai/Kimi-Linear-48B-A3B-Base)
#   TEACHER_CACHE   optional --cache-dir for transformers
#   STEPS           total KD steps (default 5000)
#   LOCAL_BS        per-rank micro-batch (default 2)
#   GLOBAL_BS       global batch (-> grad accum = G/(L*W); default 8)
#   SEQ_LEN         context length (default 2048)
#   LR              constant LR (default 2e-4 — distillation phase)
#   ALPHA           KD CE weight (default 0.3)
#   T               KD temperature (default 2.0)
#   OUT_DIR         output dir (default phase5_distillation_deprecated/runs/kd_overnight)
#   NGPU            torchrun nproc_per_node (default 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_436m_block_attn_res_n4}"
STUDENT_CKPT="${STUDENT_CKPT:-${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500}"
# Default to local snapshot path (saves transformers from re-fetching
# into HF cache layout). Falls back to repo_id form if not present.
if [[ -d "/root/hf_cache/Llama-3.1-8B" && -f "/root/hf_cache/Llama-3.1-8B/config.json" ]]; then
    TEACHER="${TEACHER:-/root/hf_cache/Llama-3.1-8B}"
else
    TEACHER="${TEACHER:-NousResearch/Meta-Llama-3.1-8B}"
fi
TEACHER_CACHE="${TEACHER_CACHE:-/root/hf_cache}"
STEPS="${STEPS:-5000}"
LOCAL_BS="${LOCAL_BS:-2}"
GLOBAL_BS="${GLOBAL_BS:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-2e-4}"
ALPHA="${ALPHA:-0.3}"
T="${T:-2.0}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/kd_overnight}"
NGPU="${NGPU:-4}"

if [[ ! -d "${STUDENT_CKPT}" ]]; then
    echo "ERROR: student ckpt dir not found: ${STUDENT_CKPT}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# torchtitan-side flags. ``checkpoint.initial_load_path`` resumes the
# student from Phase 4's last ckpt without writing into that folder
# (KD writes ckpts into ``--dump_folder/checkpoint`` instead).
TT_ARGS=(
    --module kimi_linear
    --config "${STUDENT_CONFIG}"
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B"
    --training.steps "${STEPS}"
    --training.local_batch_size "${LOCAL_BS}"
    --training.global_batch_size "${GLOBAL_BS}"
    --training.seq_len "${SEQ_LEN}"
    --optimizer.lr "${LR}"
    --lr_scheduler.warmup_steps 100
    --lr_scheduler.total_steps "${STEPS}"
    --lr_scheduler.decay_ratio 0.0
    --parallelism.pipeline_parallel_degree 1
    --parallelism.data_parallel_shard_degree "${NGPU}"
    --parallelism.data_parallel_replicate_degree 1
    --parallelism.tensor_parallel_degree 1
    --checkpoint.enable
    --checkpoint.initial_load_path "${STUDENT_CKPT}"
    --checkpoint.initial_load_model_only
    --checkpoint.interval 500
    --checkpoint.keep_latest_k 3
    --metrics.save_tb_folder tb
    --dump_folder "${OUT_DIR}"
)

# KD-specific flags (parsed by train_kd.py before torchtitan's parser).
KD_ARGS=(
    --kd.teacher "${TEACHER}"
    --kd.alpha "${ALPHA}"
    --kd.temperature "${T}"
)
if [[ -n "${TEACHER_CACHE}" ]]; then
    KD_ARGS+=(--kd.teacher-cache-dir "${TEACHER_CACHE}")
fi

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_distillation_deprecated.train_kd \
    "${KD_ARGS[@]}" \
    "${TT_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/train.log"
