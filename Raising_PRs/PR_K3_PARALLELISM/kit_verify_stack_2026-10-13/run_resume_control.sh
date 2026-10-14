#!/bin/bash
# Warm reference for the resume pairs: run 1 again from the seed (r1b) on a copy of run 1's cache, the same cache lineage
# run 2 resumed on. Resume is exact when r1b and r2 agree on steps 6-10. Waits for the queue (DONE_GPU4). No deletion.
SEED=$1 O=$2
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; V=$S/verify_stack_1013; K=$(cd "$(dirname "$0")" && pwd); T=$S/wt_mxqat_fix
until [ -f $V/DONE_GPU4 ]; do sleep 30; done
PR=$O/progress_control.txt
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <cache> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" MM18_ARM=mxqat_muon PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) loads=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
again() {  # <name> <degrees> <gpus>
  local name=$1 deg=$2 gpus=$3
  cp -r $O/cache_$name $O/cache_${name}_r1b
  run r1b_$name $gpus $O/cache_${name}_r1b MM18_CELL=$deg MM18_SEED=$SEED MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
( again fsdp2 dp=2 0,1 ) & ( again ep2_fsdp2 dp=2,ep=2 2,3 ) & ( again fsdp2_pp2 dp=2,pp=2 4,5,6,7 ) & wait
( again fsdp2_pp2_vpp2 dp=2,pp=2,vpp=2 0,1,2,3 ) & ( again fsdp2_cp2 dp=2,cp=2 4,5,6,7 ) & wait
python3 $K/compare_resume_control.py $O >> $PR 2>&1
note done; echo done > $O/DONE_CONTROL
