#!/usr/bin/env bash
# Continue-pretrain the Phase 4 student on the MiniPLM-filtered c4 corpus.
# Pure CE loss — no teacher forward in the train loop. ~5x faster than
# online KD (~3000 tps/rank vs 647 tps/rank in the failed KD run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

STUDENT_CONFIG="${STUDENT_CONFIG:-kimi_linear_436m_block_attn_res_n4}"
STUDENT_CKPT="${STUDENT_CKPT:-${WORKSPACE_DIR}/phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500}"
FILTERED="${FILTERED:-${SCRIPT_DIR}/scored/filtered.jsonl}"
STEPS="${STEPS:-30000}"
LOCAL_BS="${LOCAL_BS:-3}"
GLOBAL_BS="${GLOBAL_BS:-12}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-2e-4}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/runs/miniplm_continued}"
NGPU="${NGPU:-4}"

if [[ ! -d "${STUDENT_CKPT}" ]]; then
    echo "ERROR: student ckpt not found: ${STUDENT_CKPT}" >&2; exit 1
fi
if [[ ! -f "${FILTERED}" ]]; then
    echo "ERROR: filtered jsonl not found: ${FILTERED}" >&2
    echo "Run launch_score.sh + filter_corpus.py first." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_distillation.miniplm.train_continued_pretrain \
    --miniplm.filtered "${FILTERED}" \
    --module kimi_linear --config "${STUDENT_CONFIG}" \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${STEPS}" \
    --training.local_batch_size "${LOCAL_BS}" \
    --training.global_batch_size "${GLOBAL_BS}" \
    --training.seq_len "${SEQ_LEN}" \
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
    --checkpoint.interval 1000 \
    --checkpoint.keep_latest_k 3 \
    --metrics.save_tb_folder tb \
    --dump_folder "${OUT_DIR}" \
    --compile.enable \
    2>&1 | tee "${OUT_DIR}/train.log"
