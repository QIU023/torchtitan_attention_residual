#!/bin/bash
# Frozen-stage optimizer cells: LoRA on the MLA projections only (arm lora_mla), so pipeline stages that hold only KDA
# layers have nothing to train. usage: run_frozen.sh <tree> <out dir> <seed dir> <name|degrees|gpus>...
# A missing seed dir is written first by a single-GPU run of the same tree. ENTRYPOINT=mm18_train_export adds the
# export check.
T=$1 O=$2 SEED=$3; shift 3
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <env...>
  local name=$1 gpus=$2; shift 2; local D=$O/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" MM18_ARM=lora_mla PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$O/cache/ic \
    TRITON_CACHE_DIR=$O/cache/tc CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) \
    timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint') frozen_parts=$(echo "$clean" | grep -c 'has no trainable parameters')"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
if [ ! -d $SEED/step-0 ]; then
  run seed 0 MM18_SEED_OUT=$SEED MODULE=mm18_probe CONFIG=seed MM18_STEPS=1
  [ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
fi
for c in "$@"; do
  IFS='|' read -r name degrees gpus <<< "$c"
  run $name $gpus MM18_CELL=$degrees MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell
done
note done
