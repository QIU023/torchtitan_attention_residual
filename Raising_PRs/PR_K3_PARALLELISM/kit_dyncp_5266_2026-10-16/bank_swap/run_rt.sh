#!/bin/bash
# Bank-swap table on the review tree (CP2, dp1, GPUs 0,1): seed, a 5-step warm-up of each configuration on one cache,
# then the four measured 10-step cells. off = dynamic CP off; whole = on with a threshold no image reaches;
# split = on at 256; swap = split with the whole bank handed to the language model. No deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; T=$S/wt_cpmm_fix; R=$S/dyncp_fix/rt; O=$R/out
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
until [ -s $S/dyncp_fix/gpu_fix/result.txt ] && [ -s $S/dyncp_fix/gpu_stack/result.txt ]; do sleep 10; done
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <steps> <extra pythonpath> <env...>
  local name=$1 gpus=$2 steps=$3 extra=$4; shift 4; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" K3C_STEPS=$steps PYTHONPATH=$extra$S/main_pins_1016:$R:$T TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 2400 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) seed_loaded=$(echo "$clean" | grep -c 'Loading the checkpoint') swap_lines=$(echo "$clean" | grep -c 'bank_swap: call')"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
SEED=$O/seed
[ -d $SEED/step-0 ] || run seed 0 1 "" K3C_SEED_OUT=$SEED K3C_MAXP=2304 MODULE=rt_probe CONFIG=seed
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
C="cp=2"; G=0,1
COMMON="K3C_CELL=$C K3C_MAXP=2304 K3C_SEED=$SEED/step-0 MODULE=rt_probe CONFIG=cell"
for phase in warm meas; do
  steps=5; [ $phase = meas ] && steps=10
  dump() { [ $phase = meas ] && echo DUMP_DIR=$O/$1_$phase/dump || echo K3C_NODUMP=1; }
  run off_$phase $G $steps "" $COMMON $(dump off)
  run whole_$phase $G $steps "" $COMMON K3C_DYNCP=1000000000 $(dump whole)
  run split_$phase $G $steps "" $COMMON K3C_DYNCP=256 $(dump split)
  run swap_$phase $G $steps "$R/swap:" $COMMON K3C_DYNCP=256 BANK_SWAP=1 BANK_SWAP_LOG=$O/swap_$phase/bank_swap.log $(dump swap)
done
note done
