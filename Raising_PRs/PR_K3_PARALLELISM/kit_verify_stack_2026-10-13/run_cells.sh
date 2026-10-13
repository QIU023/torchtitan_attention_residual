#!/bin/bash
# One arm over a list of cells on one tree (coverage: rc, steps, seed load), one cache per stream, no deletion.
# usage: run_cells.sh <arm> <tree> <out dir> <cell file: name|degrees|gpus|stream>; streams A, A2, B in parallel, then C.
ARM=$1 T=$2 O=$3 L=$4
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <cache> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" MM18_ARM=$ARM PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
note "arm $ARM tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
SEED=$O/seed
[ -d $SEED/step-0 ] || run seed 0 $O/cache_seed MM18_SEED_OUT=$SEED MODULE=mm18_probe CONFIG=seed MM18_STEPS=1
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
stream() {
  while IFS='|' read -r name deg gpus s; do
    [ "$s" = "$1" ] || continue
    run $name $gpus $O/cache_$1 MM18_CELL=$deg MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell
  done < $L
}
stream A & stream A2 & stream B & wait
stream C
note done; echo done > $O/DONE
