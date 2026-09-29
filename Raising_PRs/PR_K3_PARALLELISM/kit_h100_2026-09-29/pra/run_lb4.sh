#!/bin/bash
# One pp4 x Interleaved1F1B cell (8 stages) of the widened debug model on 4 H100s, 16 micro-batches.
# Usage: run_lb4.sh <name> <tree> <cache_dir> <steps> [extra titan args...]
# Env: LB_ROOT (output root), VENV, PPMEM_DIM / PPMEM_SEQ / PPMEM_* switches, LB_PROFILE_STEP.
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
PPMEM_OUT="$OUT/mem" PPMEM_SEQ="$seq" PYTHONPATH="$K:." \
TORCHINDUCTOR_CACHE_DIR="$cache/ic" TRITON_CACHE_DIR="$cache/tc" \
${VENV:?}/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  --local-ranks-filter "${LB_LOG_RANK:-3}" --role rank --tee 3 \
  -m torchtitan.train --module probe_lbw --config lb_probe \
  --training.steps "$steps" --debug.seed 42 --debug.deterministic --metrics.log-freq 1 \
  --training.num-tokens-per-train-step "$((16 * seq))" --training.num-tokens-per-microbatch-per-dp-rank "$seq" \
  --parallelism.pipeline-parallel-degree 4 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
  --parallelism.num-pp-microbatches 16 --dump-folder "$OUT/dump" "${prof[@]}" "$@" > "$OUT/train.log" 2>&1
echo "rc=$?" >> "$OUT/train.log"
