#!/bin/bash
# Checkpoint resume on the stack, arm mxqat_muon (MX QAT + DistMuon): per layout, run 1 trains 10 steps from the seed and
# saves full state at steps 5 and 10; run 2 copies the step-5 checkpoint and run 1's warm cache, resumes and trains to 10.
# Steps 6 to 10 must match. No deletion. usage: run_resume.sh <seed dir> <out dir>
SEED=$1 O=$2
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd); T=$S/wt_mxqat_fix
mkdir -p $O; PR=$O/progress.txt
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
pair() {  # <name> <degrees> <gpus>
  local name=$1 deg=$2 gpus=$3
  run r1_$name $gpus $O/cache_$name MM18_CELL=$deg MM18_SEED=$SEED MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5
  cp -r $O/cache_$name $O/cache_${name}_r2
  mkdir -p $O/r2_$name/out/checkpoint && cp -r $O/r1_$name/out/checkpoint/step-5 $O/r2_$name/out/checkpoint/
  run r2_$name $gpus $O/cache_${name}_r2 MM18_CELL=$deg MM18_SEED=$SEED MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5 MM18_LOAD_STEP=5
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
( pair dp1 dp=1 0 ) & ( pair fsdp2 dp=2 1,2 ) & ( pair pp2 pp=2 3,4 ) & ( pair ep2_fsdp2 dp=2,ep=2 5,6 ) & wait
( pair pp4_vpp2 pp=4,vpp=2 0,1,2,3 ) & ( pair cp2 cp=2 4,5 ) & wait
note done; echo done > $O/DONE
