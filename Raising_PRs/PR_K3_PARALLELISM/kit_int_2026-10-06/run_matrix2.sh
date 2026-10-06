#!/bin/bash
# Usage: run_matrix2.sh <out> <spec>...   (run_matrix.sh, but a spec may set LOG_RANK: the last pipeline rank prints the loss)   spec = name:tree_dir:gpus:ENV=..,ENV=..
# Each cell first warms the cell's own cache with a 1-step run of every tree in the spec list that shares its
# warm key (WARM=<key> in its env), then runs on a copy; progress in <out>/progress.txt.
O=$1; shift; K=$(cd "$(dirname "$0")" && pwd); mkdir -p $O; P=$O/progress.txt
. /workspace/venv_1006i/bin/activate
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <name> <tree> <gpus> <cache> <steps> <envs>
  local name=$1 W=$2 gpus=$3 C=$4 steps=$5 envs=$6 D=$O/$1; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $W && env LOG_RANK=0 $(echo $envs | tr ',' ' ') INT_STEPS=$steps PYTHONPATH=${PP_PRE:+$PP_PRE:}$K:. TORCHINDUCTOR_CACHE_DIR=$C/ic \
      TRITON_CACHE_DIR=$C/tc CUDA_VISIBLE_DEVICES=$gpus NGPU=$n MODULE=int_probe CONFIG=cell \
      timeout 3600 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]* *grad_norm: *[0-9.]*' | sed -n '1p;$p' | tr '\n' ' ')"
}
declare -A WARMED
for spec in "$@"; do
  IFS=: read name W gpus envs <<< "$spec"
  key=$(echo $envs | tr ',' '\n' | grep '^WARM=' | cut -d= -f2); C=$O/cache_$key; mkdir -p $C/ic $C/tc
  if [ -z "${WARMED[$key]}" ]; then
    for s2 in "$@"; do IFS=: read n2 W2 g2 e2 <<< "$s2"
      k2=$(echo $e2 | tr ',' '\n' | grep '^WARM=' | cut -d= -f2); [ "$k2" = "$key" ] && run warm_$n2 $W2 $g2 $C ${WARM_STEPS:-1} $e2
    done
    WARMED[$key]=1
  fi
  rm -rf $O/cc_$key; cp -r $C $O/cc_$key; run $name $W $gpus $O/cc_$key ${STEPS:-20} $envs; rm -rf $O/cc_$key
done
note done
