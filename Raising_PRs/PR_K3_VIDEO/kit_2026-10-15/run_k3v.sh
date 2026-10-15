#!/bin/bash
# K3 video PR matrix on one tree: seed, then streams A, A2, B in parallel, then C, then D. One cache per stream, no deletion.
# usage: run_k3v.sh <tree> <out dir> <cell file: name|degrees|gpus|stream>
T=$1 O=$2 L=$3
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; R=$S/k3v_run
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <cache> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$S/main_pins_1015:$R:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) MASTER_PORT=$((29500 + ${gpus%%,*})) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
SEED=$O/seed
[ -d $SEED/step-0 ] || run seed 0 $O/cache_seed K3V_SEED_OUT=$SEED MODULE=k3v_cells CONFIG=seed
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
stream() {
  while IFS='|' read -r name deg gpus s; do
    [ "$s" = "$1" ] || continue
    run $name $gpus $O/cache_$1 K3V_CELL=$deg K3V_SEED=$SEED/step-0 MODULE=k3v_cells CONFIG=cell
  done < $L
}
stream A & stream A2 & stream B & wait
stream C
stream D
note done
