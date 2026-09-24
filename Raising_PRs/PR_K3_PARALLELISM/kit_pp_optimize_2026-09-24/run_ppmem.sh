#!/bin/bash
# One pp8 x vp2 cell of the block-memory probe on 8 GPUs.
# Usage: run_ppmem.sh <name> <tree> <cache_dir> <seq> <steps> [extra titan args...]
set -uo pipefail
name=$1; tree=$2; cache=$3; seq=$4; steps=$5; shift 5
K=$(cd "$(dirname "$0")" && pwd)
OUT=${PPMEM_ROOT:-/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/ppmem}/$name
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$tree"
PPMEM_OUT="$OUT/mem" PPMEM_SEQ="$seq" PYTHONPATH="$K:." \
TORCHINDUCTOR_CACHE_DIR="$cache/ic" TRITON_CACHE_DIR="$cache/tc" \
/workspace/venv_bfx9/bin/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:29910 \
  --local-ranks-filter 7 --role rank --tee 3 \
  -m torchtitan.train --module probe_ppmem --config ppmem_wide \
  --training.steps "$steps" --debug.seed 42 --debug.deterministic \
  --training.num-tokens-per-train-step "$((16 * seq))" --training.num-tokens-per-microbatch-per-dp-rank "$seq" \
  --parallelism.pipeline-parallel-degree 8 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
  --parallelism.num-pp-microbatches 16 --dump-folder "$OUT/dump" "$@" > "$OUT/train.log" 2>&1
echo "rc=$?" >> "$OUT/train.log"
