#!/bin/bash
# The cells of one campaign step, run one after another on 8 GPUs.
# Usage: campaign.sh <tag> <steps> <profile_step> <cell>=<tree> [<cell>=<tree> ...]
# Warms cache0 with one step of every tree, then for each cell: a memory and identity run of
# <steps> steps and a timing run traced at <profile_step>, each on its own copy of cache0.
# Env: LB_ROOT and the PPMEM_* layout variables (see run_lb.sh).
set -uo pipefail
tag=$1; steps=$2; pstep=$3; shift 3
K=$(cd "$(dirname "$0")" && pwd)
: "${LB_ROOT:?}"
mkdir -p "$LB_ROOT/cache0"
for pair in "$@"; do
  cell=${pair%%=*}; tree=${pair#*=}
  bash "$K/run_lb.sh" "${tag}_warm_${cell}" "$tree" "$LB_ROOT/cache0" 1
done
for pair in "$@"; do
  cell=${pair%%=*}; tree=${pair#*=}
  rm -rf "$LB_ROOT/cache_${tag}_${cell}"; cp -r "$LB_ROOT/cache0" "$LB_ROOT/cache_${tag}_${cell}"
  bash "$K/run_lb.sh" "${tag}_mem_${cell}" "$tree" "$LB_ROOT/cache_${tag}_${cell}" "$steps"
  rm -rf "$LB_ROOT/cache_${tag}_${cell}t"; cp -r "$LB_ROOT/cache0" "$LB_ROOT/cache_${tag}_${cell}t"
  PPMEM_STORE_TRACK=0 LB_PROFILE_STEP="$pstep" \
    bash "$K/run_lb.sh" "${tag}_time_${cell}" "$tree" "$LB_ROOT/cache_${tag}_${cell}t" "$pstep"
done
echo "campaign $tag done" > "$LB_ROOT/${tag}.done"
