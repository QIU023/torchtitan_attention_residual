#!/bin/bash
# One pp8 x Interleaved1F1B cell of the PP lower-bound campaign on 8 GPUs, 16 micro-batches.
# Usage: run_lb.sh <name> <tree> <cache_dir> <steps> [extra titan args...]
# Env: LB_ROOT (output root, required), PPMEM_LAYERS / PPMEM_BLOCK / PPMEM_DIM / PPMEM_SEQ /
#      PPMEM_LPS (layers per stage) / PPMEM_STORE_TRACK, LB_PROFILE_STEP (trace that step).
set -uo pipefail
name=$1; tree=$2; cache=$3; steps=$4; shift 4
K=$(cd "$(dirname "$0")" && pwd)
OUT=${LB_ROOT:?}/$name
rm -rf "$OUT"; mkdir -p "$OUT"
seq=${PPMEM_SEQ:-2048}
prof=()
if [ -n "${LB_PROFILE_STEP:-}" ]; then
  prof=(--profiler.enable-profiling --profiler.profile-freq "$LB_PROFILE_STEP"
        --profiler.profiler-warmup 2 --profiler.profiler-active 1
        --profiler.save-traces-folder traces)
fi
cd "$tree"
PPMEM_OUT="$OUT/mem" PPMEM_SEQ="$seq" PYTHONPATH="$K:/tmp/attn_gym_up:." \
TORCHINDUCTOR_CACHE_DIR="$cache/ic" TRITON_CACHE_DIR="$cache/tc" \
/workspace/venv_bfx9/bin/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  --local-ranks-filter "${LB_LOG_RANK:-7}" --role rank --tee 3 \
  -m torchtitan.train --module probe_lb --config lb_probe \
  --training.steps "$steps" --debug.seed 42 --debug.deterministic \
  --training.num-tokens-per-train-step "$((16 * seq))" --training.num-tokens-per-microbatch-per-dp-rank "$seq" \
  --parallelism.pipeline-parallel-degree 8 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
  --parallelism.num-pp-microbatches 16 --dump-folder "$OUT/dump" "${prof[@]}" "$@" > "$OUT/train.log" 2>&1
echo "rc=$?" >> "$OUT/train.log"
