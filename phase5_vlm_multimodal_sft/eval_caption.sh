#!/usr/bin/env bash
# Evaluate the multimodal-finetuned student on a caption held-out set.
# Resumes the training trainer with training.steps = ckpt + 1 and
# logs the train-step loss as a proxy for caption quality. For real
# downstream eval (VQAv2 zero-shot accuracy, COCO retrieval), see
# eval_vqa.sh (TODO).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_DIR="${WORKSPACE_DIR}/torchtitan"

RUN="${RUN:-${SCRIPT_DIR}/runs/mm_full_finetune}"
STEP="${STEP:-30000}"
CKPT="${RUN}/checkpoint/step-${STEP}"
EVAL_OUT="${SCRIPT_DIR}/runs/mm_eval"

if [[ ! -d "${CKPT}" ]]; then
    echo "ERROR: ckpt not found: ${CKPT}" >&2; exit 1
fi

mkdir -p "${EVAL_OUT}"
NEXT=$((STEP + 1))

PYTHONPATH="${WORKSPACE_DIR}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m phase5_vlm_multimodal_sft.train_mm \
    --mm.json /root/hf_cache/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --mm.images /root/hf_cache/LLaVA-Pretrain \
    --mm.vision-model google/siglip-base-patch16-224 \
    --mm.tokenizer NousResearch/Meta-Llama-3.1-8B \
    --mm.cache-dir /root/hf_cache \
    --module attention_residual --config kimi_linear_436m_block_attn_res_n4 \
    --hf_assets_path "${TORCHTITAN_DIR}/assets/hf/Llama-3.1-8B" \
    --training.steps "${NEXT}" \
    --training.local_batch_size 8 --training.global_batch_size 32 \
    --training.seq_len 260 \
    --optimizer.lr 1e-7 \
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
    --checkpoint.interval 99999 --checkpoint.keep_latest_k 2 \
    --metrics.save_tb_folder tb \
    --dump_folder "${EVAL_OUT}" \
    --compile.enable \
    2>&1 | tee "${EVAL_OUT}/eval.log"
