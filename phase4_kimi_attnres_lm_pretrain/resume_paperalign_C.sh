#!/usr/bin/env bash
# Resume the lm_447m_fp8_paperalign_C run from the latest ckpt in
# runs/lm_447m_fp8_paperalign_C/checkpoint/ (auto-detects step-9700+).
#
# Why this file exists:
#   The original launcher (launch_redo_paperalign_10B.sh) writes to a
#   different OUT_DIR (lm_447m_redo_10B_fp8) with different STEPS /
#   LOCAL_BS / WARMUP, so it CANNOT be used to resume paperalign_C.
#   This script encodes the exact args from the original paperalign_C
#   ps line so torchtitan auto-resumes the latest ckpt from the
#   correct dump_folder.
#
# Safety:
#   - Does NOT touch existing ckpts (torchtitan auto-loads latest)
#   - Does NOT delete the wrong-run dir
#   - --checkpoint.keep_latest_k 2  → only step-9600 + step-9700 + new
#     are kept (rotation as before)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/runs/lm_447m_fp8_paperalign_C"

if [[ ! -d "${OUT_DIR}/checkpoint" ]]; then
    echo "ERROR: ${OUT_DIR}/checkpoint missing — refusing to resume" >&2
    exit 2
fi

LATEST=$(ls -d "${OUT_DIR}/checkpoint/step-"* 2>/dev/null | sort -V | tail -1)
echo "[$(date)] resuming from: ${LATEST:-<none>}"

# Must cd into torchtitan/ for `-m torchtitan.train` to resolve
TORCHTITAN_DIR="${SCRIPT_DIR}/../torchtitan"
cd "${TORCHTITAN_DIR}"
echo "[$(date)] cwd=${PWD}"

exec /usr/local/bin/torchrun \
    --nproc_per_node=8 \
    --rdzv_backend c10d \
    --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module kimi_linear \
    --config kimi_linear_447m_aligned_block_attn_res_n4_fp8 \
    --training.steps 12750 \
    --training.local_batch_size 4 \
    --training.global_batch_size 384 \
    --training.seq_len 2048 \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.data_parallel_shard_degree 8 \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    --optimizer.lr 1.5e-3 \
    --compile.enable \
    --validator.enable \
    --validator.freq 100 \
    --validator.steps 10 \
    --checkpoint.enable \
    --checkpoint.interval 100 \
    --checkpoint.keep_latest_k 2 \
    --lr_scheduler.warmup_steps 150 \
    --lr_scheduler.decay_ratio 0.8 \
    --lr_scheduler.min_lr_factor 0.1 \
    --dump_folder "${OUT_DIR}" \
    --metrics.save_tb_folder tb
