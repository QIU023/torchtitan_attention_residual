#!/bin/bash
# LoRA PR results tables on 8 x 5060, one compile lineage per table: the seed checkpoint, then every configuration runs
# once for 10 steps into ONE shared cache (sequential), then every measured cell runs on its own copy of that warm cache;
# the "_cold" rows run the reference configuration on an empty cache (the noise floor). ENTRYPOINT=mm18_train_export.
# usage: run_lora_table.sh <arm> <tree dir> <out dir> <warm-up and measured cell list file>
# cell list lines: name|degrees|gpus|stream   (stream A, A2, B run in parallel in the measured phase, C after them)
ARM=$1 T=$2 O=$3 L=$4
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
export ENTRYPOINT=mm18_train_export
run() {  # <name> <gpus> <cache dir> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" MM18_ARM=$ARM PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint') export_ranks=$(echo "$clean" | grep -c 'EXPORT_CHECK')"
}
note "arm $ARM tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(python -c 'import torch;print(torch.__version__)')"
SEED=$O/seed; rm -rf $SEED $O/cache
run seed 0 $O/cache_seed MM18_SEED_OUT=$SEED MODULE=mm18_probe CONFIG=seed MM18_STEPS=1; rm -rf $O/cache_seed
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
cellrun() { run "$1" "$2" "$3" MM18_CELL="$4" MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell; }
# warm-up: every configuration once, sequentially, into the one shared cache
while IFS='|' read -r name deg gpus stream; do
  [ -z "$name" ] && continue; case $name in *_cold) continue;; esac
  cellrun warm_$name $gpus $O/cache "$deg"
done < $L
note "warm-up done, cache $(du -sh $O/cache | cut -f1)"
measured() {  # <stream>
  while IFS='|' read -r name deg gpus stream; do
    [ "$stream" = "$1" ] || continue
    case $name in
      *_cold) rm -rf $O/cache_$name; cellrun $name $gpus $O/cache_$name "$deg" ;;
      *) rm -rf $O/cache_$name; cp -r $O/cache $O/cache_$name; cellrun $name $gpus $O/cache_$name "$deg" ;;
    esac
    rm -rf $O/cache_$name
  done < $L
}
measured A & measured A2 & measured B & wait
measured C
note "done"; echo done > $O/DONE
