#!/usr/bin/env bash
# Evaluate MiniPLM-trained student on c4_validation. Mirrors
# phase5_distillation/eval_kd_student.sh — same strategy: re-launch
# trainer with training.steps = ckpt_step + 1 + --validator.enable.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

VAL_STEPS="${VAL_STEPS:-200}"
RUN="${RUN:-${SCRIPT_DIR}/runs/miniplm_continued}"
STEP="${STEP:-10000}"
CKPT="${RUN}/checkpoint/step-${STEP}"
EVAL_OUT="${SCRIPT_DIR}/runs/miniplm_eval"

if [[ ! -d "${CKPT}" ]]; then
    echo "ERROR: ckpt not found: ${CKPT}" >&2; exit 1
fi
mkdir -p "${EVAL_OUT}"
NEXT=$((STEP + 1))

PYTHONPATH="${TORCHTITAN_DIR}:${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module kimi_linear --config kimi_linear_436m_block_attn_res_n4 \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${NEXT}" \
    --training.local_batch_size 3 \
    --training.global_batch_size 12 \
    --training.seq_len 2048 \
    --optimizer.lr 2e-4 \
    --lr_scheduler.warmup_steps 100 \
    --lr_scheduler.total_steps "${STEP}" \
    --lr_scheduler.decay_ratio 0.0 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --checkpoint.enable \
    --checkpoint.initial_load_path "${CKPT}" \
    --checkpoint.initial_load_model_only \
    --checkpoint.interval 99999 \
    --checkpoint.keep_latest_k 2 \
    --validator.enable --validator.freq 1 --validator.steps "${VAL_STEPS}" \
    --metrics.save_tb_folder tb \
    --dump_folder "${EVAL_OUT}" \
    --compile.enable \
    2>&1 | tee "${EVAL_OUT}/eval.log"

echo ""
echo "=== Validation summary ==="
grep "validate step:" "${EVAL_OUT}/eval.log" | sed -E 's/\x1b\[[0-9;]*m//g' | tail -1
