#!/usr/bin/env bash
# Launch 4 parallel score_corpus.py processes, one per GPU.
#
# Each process scores a disjoint slice of c4-en. Default total
# = 4 * 125,000 = 500,000 chunks (≈1B tokens, 500 MB jsonl).
#
# Memory: 23 GB / 31 GB per GPU (single-rank, no FSDP).
# Time:   ~1-2 h per shard at B=8 T=2048.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_SHARDS="${NUM_SHARDS:-4}"
NUM_CHUNKS_PER_SHARD="${NUM_CHUNKS_PER_SHARD:-125000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
TEACHER="${TEACHER:-NousResearch/Meta-Llama-3.1-8B}"
REFERENCE="${REFERENCE:-NousResearch/Llama-3.2-1B}"
CACHE_DIR="${CACHE_DIR:-/root/hf_cache}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/scored}"

mkdir -p "${OUT_DIR}"

PIDS=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    out="${OUT_DIR}/scored_${shard}.jsonl"
    log="${OUT_DIR}/scored_${shard}.log"
    echo "[launch] shard ${shard} -> ${out} (log: ${log})"
    CUDA_VISIBLE_DEVICES="${shard}" \
    python "${SCRIPT_DIR}/score_corpus.py" \
        --teacher "${TEACHER}" \
        --reference "${REFERENCE}" \
        --cache-dir "${CACHE_DIR}" \
        --shard "${shard}" --num-shards "${NUM_SHARDS}" \
        --num-chunks "${NUM_CHUNKS_PER_SHARD}" \
        --batch-size "${BATCH_SIZE}" --seq-len "${SEQ_LEN}" \
        --out "${out}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
done

echo "[launch] all 4 shards spawned (pids: ${PIDS[*]}), waiting..."
for pid in "${PIDS[@]}"; do
    wait "${pid}"
    echo "[launch] pid ${pid} exited rc=$?"
done
echo "[launch] all shards done. outputs in ${OUT_DIR}/"
