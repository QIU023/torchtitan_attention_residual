#!/bin/bash
# One pp4 x Interleaved1F1B probe cell on four 5060s, 16 micro-batches (sizing only).
# Usage: run_lb4_5060.sh <name> <tree> <cache_dir> <steps> <gpus>; env as run_lb.sh (LB_ROOT, PPMEM_*).
set -uo pipefail
name=$1; tree=$2; cache=$3; steps=$4; gpus=$5; shift 5
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_pp_lowerbound_2026-09-26
OUT=${LB_ROOT:?}/$name
rm -rf "$OUT"; mkdir -p "$OUT"
seq=${PPMEM_SEQ:-2048}
cd "$tree"
CUDA_VISIBLE_DEVICES=$gpus PPMEM_OUT="$OUT/mem" PPMEM_SEQ="$seq" PYTHONPATH="$K:/tmp/attn_gym_up:." \
TORCHINDUCTOR_CACHE_DIR="$cache/ic" TRITON_CACHE_DIR="$cache/tc" \
timeout 2400 /workspace/venv_bfx9/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  --local-ranks-filter 3 --role rank --tee 3 \
  -m torchtitan.train --module probe_lb --config lb_probe \
  --training.steps "$steps" --debug.seed 42 --debug.deterministic \
  --training.num-tokens-per-train-step "$((16 * seq))" --training.num-tokens-per-microbatch-per-dp-rank "$seq" \
  --parallelism.pipeline-parallel-degree 4 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
  --parallelism.num-pp-microbatches 16 --dump-folder "$OUT/dump" "$@" > "$OUT/train.log" 2>&1
echo "rc=$?" >> "$OUT/train.log"
