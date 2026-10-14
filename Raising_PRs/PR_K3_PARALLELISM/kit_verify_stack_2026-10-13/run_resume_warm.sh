#!/bin/bash
# All-warm resume check after the control: per layout, r2b resumes from r1b's own step-5 checkpoint on a copy of r1b's
# cache, and r1c reruns r1b from the seed on another copy (warm-warm determinism). Waits for DONE_CONTROL. No deletion.
SEED=$1 O=$2
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd); T=$S/wt_mxqat_fix
until [ -f $O/DONE_CONTROL ]; do sleep 30; done
PR=$O/progress_warm.txt
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
warm() {  # <name> <degrees> <gpus>
  local name=$1 deg=$2 gpus=$3
  cp -r $O/cache_${name}_r1b $O/cache_${name}_r2b
  cp -r $O/cache_${name}_r1b $O/cache_${name}_r1c
  mkdir -p $O/r2b_$name/out/checkpoint && cp -r $O/r1b_$name/out/checkpoint/step-5 $O/r2b_$name/out/checkpoint/
  run r2b_$name $gpus $O/cache_${name}_r2b MM18_CELL=$deg MM18_SEED=$SEED MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5 MM18_LOAD_STEP=5
  run r1c_$name $gpus $O/cache_${name}_r1c MM18_CELL=$deg MM18_SEED=$SEED MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
( warm fsdp2 dp=2 0,1 ) & ( warm ep2_fsdp2 dp=2,ep=2 2,3 ) & ( warm fsdp2_pp2 dp=2,pp=2 4,5,6,7 ) & wait
( warm fsdp2_pp2_vpp2 dp=2,pp=2,vpp=2 0,1,2,3 ) & ( warm fsdp2_cp2 dp=2,cp=2 4,5,6,7 ) & wait
python3 $K/compare_resume_warm.py $O >> $PR 2>&1
note done; echo done > $O/DONE_WARM
