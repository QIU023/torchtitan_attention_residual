#!/bin/bash
# #4380 tower benchmark on the H100 (4 GPUs): one torchrun per cell, sequential, nothing else on the box (kit tower_bench.py).
# For each AC mode (sac = the K3 recipe's, then none) and case: main at CP 1, the PR at CP 2 and 4, main at CP 2 and 4.
# Flex attention compiles without max-autotune on both sides (kit local/flex_no_autotune, via kit_setup/shims_noat).
K=/workspace/kit_cpmm; M=/workspace/tt_main; C=/workspace/tt_cpmm; O=${O:-/workspace/h100_tower}
mkdir -p $O/logs $O/cache/ic $O/cache/tc
. /workspace/kit_setup/env_cu126.sh
P=$O/progress.txt
cell() {  # name tree cp ac case
  local name=$1 tree=$2 cp=$3 ac=$4 case=$5 gpus rc
  gpus=$(seq -s, 0 $((cp - 1)))
  ( cd $tree && env CASE=$case AC=$ac TAG=$name OUT=$O/results.jsonl PYTHONPATH=/workspace/kit_setup/shims_noat:$K:. \
      TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc CUDA_VISIBLE_DEVICES=$gpus \
      timeout ${CELL_TIMEOUT:-900} torchrun --nproc_per_node=$cp --master_port=29711 $K/tower_bench.py > $O/logs/$name.log 2>&1 )
  rc=$?
  echo "$(date +%H:%M:%S) $name rc=$rc $(grep -a -o '"status": "[a-z]*"\|"max_peak_gib": [0-9.]*\|"ms": [0-9.na]*' $O/logs/$name.log | tr '\n' ' ')" >> $P
}
for ac in ${ACS:-sac none}; do
  for case in ${CASES_LIST:-1008px 2016px video16f448 4x2016px 4032px}; do
    cell main_${ac}_${case}_cp1 $M 1 $ac $case
    cell pr_${ac}_${case}_cp2 $C 2 $ac $case
    cell pr_${ac}_${case}_cp4 $C 4 $ac $case
    cell main_${ac}_${case}_cp2 $M 2 $ac $case
    cell main_${ac}_${case}_cp4 $M 4 $ac $case
  done
done
echo "$(date +%H:%M:%S) TOWER_DONE" >> $P
