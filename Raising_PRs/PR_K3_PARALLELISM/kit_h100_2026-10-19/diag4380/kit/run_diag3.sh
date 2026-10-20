#!/bin/bash
# Control: clipping off (max_norm 1e9), recipe optimizer, 20 steps, main against the PR with dynamic CP on and no split.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; D=$S/diag4380; K=$D/kit
source /workspace/venv_1006/bin/activate
P=$D/progress3.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
one() {  # <name> <tree> <gpus> <envs>
  local name=$1 W=$D/$2 gpus=$3 envs=$4 O=$D/c3_$1; rm -rf $O $D/cc3_$1; mkdir -p $O; cp -r $D/cache_w $D/cc3_$1
  ( cd $W && env $(echo $envs | tr ',' ' ') CP_STEPS=20 CP_SEED=$D/seed/step-0 DUMP_DIR=$O/dump \
      PYTHONPATH=$K:$S/main_pins_1016:. TORCHINDUCTOR_CACHE_DIR=$D/cc3_$1/ic TRITON_CACHE_DIR=$D/cc3_$1/tc \
      CUDA_VISIBLE_DEVICES=$gpus NGPU=2 LOG_RANK=0 MODULE=upd1_probe CONFIG=cell \
      timeout 1800 ./run_train.sh --output-dir $O/out > $O/run.log 2>&1 )
  local rc=$?; rm -rf $O/out $D/cc3_$1 $O/dump
  note "$name rc=$rc $(grep -a -o 'max_norm [0-9.e+]*' $O/run.log | head -1)"
}
one main main 0,1 CP_MODE=allgather,CLIP_OFF=1 &
one prwhole pr 2,3 CP_MODE=allgather,CP_MINP=1000000000,CLIP_OFF=1 &
one main_clip main 4,5 CP_MODE=allgather,CLIP_OFF=0 &
one prwhole_clip pr 6,7 CP_MODE=allgather,CP_MINP=1000000000,CLIP_OFF=0 &
wait
note DIAG3_DONE
