#!/bin/bash
# Memory-only cells for the balance question, on one cache lineage: every cell warms cache0 for
# one step, then runs <steps> steps on its own copy. A cell is <name>=<tree>[@VAR=val,...].
# Usage: balance_campaign.sh <tag> <steps> <cell> [<cell> ...]; env: LB_ROOT and the PPMEM_* layout.
set -uo pipefail
tag=$1; steps=$2; shift 2
K=$(cd "$(dirname "$0")" && pwd)
: "${LB_ROOT:?}"
mkdir -p "$LB_ROOT/cache0"
run_cell() {  # <spec> <run name> <cache> <steps>
  local spec=$1 run=$2 cache=$3 n=$4
  local rest=${spec#*=}
  local tree=${rest%%@*} envs=""
  [[ "$rest" == *@* ]] && envs=${rest#*@}
  env ${envs//,/ } bash "$K/run_lb.sh" "$run" "$tree" "$cache" "$n"
}
for spec in "$@"; do
  run_cell "$spec" "${tag}_warm_${spec%%=*}" "$LB_ROOT/cache0" 1
done
for spec in "$@"; do
  cell=${spec%%=*}
  rm -rf "$LB_ROOT/cache_${tag}_${cell}"; cp -r "$LB_ROOT/cache0" "$LB_ROOT/cache_${tag}_${cell}"
  run_cell "$spec" "${tag}_mem_${cell}" "$LB_ROOT/cache_${tag}_${cell}" "$steps"
  rm -rf "$LB_ROOT/cache_${tag}_${cell}"
done
echo "campaign $tag finished" > "$LB_ROOT/${tag}.done"
