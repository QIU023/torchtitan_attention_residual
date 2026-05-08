#!/usr/bin/env bash
# Long-context + cuda-graph-ON 4-way bench sweep.
set -uo pipefail

MODEL=/root/torchtitan_attention_residual/phase11/hf_aligned_447m
OUT_DIR=/root/torchtitan_attention_residual/phase11/bench_results
mkdir -p "$OUT_DIR"

# Each prefill + decode must fit under max_position_embeddings=32768.
# Decode kept small so prefill dominates and we measure the true TTFT
# delta from the AttnRes algorithm itself, not warmup overhead.

for tp in 1 8; do
  for prefill in 4096 8192 16384 24576; do
    decode=128
    out="$OUT_DIR/tp${tp}_prefill${prefill}.json"
    echo ""
    echo "============================================================"
    echo "  TP=$tp  prefill=$prefill  decode=$decode  (cuda-graph ON)"
    echo "============================================================"
    python3 phase11/bench_attn_res.py \
        --model "$MODEL" --tp "$tp" \
        --prefill "$prefill" --decode "$decode" \
        --warmup 2 --timed 3 \
        --out "$out" 2>&1 | tail -15
  done
done

echo ""
echo "=== ALL DONE ==="
ls -la "$OUT_DIR"/*.json
