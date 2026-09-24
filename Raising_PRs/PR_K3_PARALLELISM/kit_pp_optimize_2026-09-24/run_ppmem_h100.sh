#!/bin/bash
# One pp4 x vp4 cell of the block-memory probe on 4 x H100 80 GB.
# Usage: run_ppmem_h100.sh <name> <tree> <cache_dir> <seq> <steps> [extra titan args...]
# Defaults (h100_probe_sizing.py): 46 layers in blocks of 6 (every block opens inside a stage, as in
# the released layout), dim 4096, 16 micro-batches, 3 layers per stage (16 stages). Override the
# shape with PPMEM_DIM / PPMEM_LAYERS / PPMEM_BLOCK and the launcher with TORCHRUN.
set -uo pipefail
name=$1; tree=$2; cache=$3; seq=$4; steps=$5; shift 5
K=$(cd "$(dirname "$0")" && pwd)
OUT=${PPMEM_ROOT:-$HOME/results/ppmem}/$name
TORCHRUN=${TORCHRUN:-torchrun}
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$tree"
PPMEM_OUT="$OUT/mem" PPMEM_SEQ="$seq" PPMEM_DIM="${PPMEM_DIM:-4096}" PPMEM_LAYERS="${PPMEM_LAYERS:-46}" \
PPMEM_BLOCK="${PPMEM_BLOCK:-6}" PYTHONPATH="$K:." \
TORCHINDUCTOR_CACHE_DIR="$cache/ic" TRITON_CACHE_DIR="$cache/tc" \
"$TORCHRUN" --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:29911 \
  --local-ranks-filter 3 --role rank --tee 3 \
  -m torchtitan.train --module probe_ppmem --config ppmem_wide \
  --training.steps "$steps" --debug.seed 42 --debug.deterministic \
  --training.num-tokens-per-train-step "$((16 * seq))" --training.num-tokens-per-microbatch-per-dp-rank "$seq" \
  --parallelism.pipeline-parallel-degree 4 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
  --parallelism.pipeline-parallel-layers-per-stage "${PPMEM_LPS:-3}" \
  --parallelism.num-pp-microbatches 16 --dump-folder "$OUT/dump" "$@" > "$OUT/train.log" 2>&1
echo "rc=$?" >> "$OUT/train.log"
