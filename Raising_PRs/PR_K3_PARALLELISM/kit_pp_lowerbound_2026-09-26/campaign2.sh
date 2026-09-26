#!/bin/bash
# campaign.sh with per-cell environment: a cell is <name>=<tree>[@VAR=val[,VAR=val...]].
# Usage: campaign2.sh <tag> <steps> <profile_step> <cell> [<cell> ...]
# Warms cache0 with one step of every cell, then for each cell a memory and identity run of
# <steps> steps and a timing run traced at <profile_step>, each on its own copy of cache0.
set -uo pipefail
tag=$1; steps=$2; pstep=$3; shift 3
K=$(cd "$(dirname "$0")" && pwd)
: "${LB_ROOT:?}"
mkdir -p "$LB_ROOT/cache0"
run_cell() {  # <spec> <run name> <cache> <steps> [env ...]
  local spec=$1 run=$2 cache=$3 n=$4; shift 4
  local cell=${spec%%=*} rest=${spec#*=}
  local tree=${rest%%@*} envs=""
  [[ "$rest" == *@* ]] && envs=${rest#*@}
  env ${envs//,/ } "$@" bash "$K/run_lb.sh" "$run" "$tree" "$cache" "$n"
}
for spec in "$@"; do
  cell=${spec%%=*}
  run_cell "$spec" "${tag}_warm_${cell}" "$LB_ROOT/cache0" 1
done
for spec in "$@"; do
  cell=${spec%%=*}
  rm -rf "$LB_ROOT/cache_${tag}_${cell}"; cp -r "$LB_ROOT/cache0" "$LB_ROOT/cache_${tag}_${cell}"
  run_cell "$spec" "${tag}_mem_${cell}" "$LB_ROOT/cache_${tag}_${cell}" "$steps"
  rm -rf "$LB_ROOT/cache_${tag}_${cell}t"; cp -r "$LB_ROOT/cache0" "$LB_ROOT/cache_${tag}_${cell}t"
  run_cell "$spec" "${tag}_time_${cell}" "$LB_ROOT/cache_${tag}_${cell}t" "$pstep" \
    PPMEM_STORE_TRACK=0 LB_PROFILE_STEP="$pstep"
done
echo "campaign $tag finished" > "$LB_ROOT/${tag}.done"
