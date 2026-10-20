#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; D=$S/diag4380; K=$D/kit
source /workspace/venv_1006/bin/activate
P=$D/progress2.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
one() {  # <name> <tree> <gpus> <envs>
  local name=$1 W=$D/$2 gpus=$3 envs=$4 O=$D/g2_$1; rm -rf $O $D/cc2_$1; mkdir -p $O; cp -r $D/cache_w $D/cc2_$1; [ $1 = tokens ] && rm -rf $D/cc2_$1 && mkdir -p $D/cc2_$1
  ( cd $W && env $(echo $envs | tr ',' ' ') CP_STEPS=2 CP_SEED=$D/seed/step-0 \
      PYTHONPATH=$K:$S/main_pins_1016:. TORCHINDUCTOR_CACHE_DIR=$D/cc2_$1/ic TRITON_CACHE_DIR=$D/cc2_$1/tc \
      CUDA_VISIBLE_DEVICES=$gpus NGPU=2 LOG_RANK=0,1 MODULE=upd2_probe CONFIG=cell \
      timeout 1800 ./run_train.sh --output-dir $O/out > $O/run.log 2>&1 )
  local rc=$?; rm -rf $O/out $D/cc2_$1
  note "$name rc=$rc patched=$(grep -a -c 'upd2_probe\] rank' $O/run.log) dumps=$(ls $O/dump 2>/dev/null | tr '\n' ' ')"
}
one main main 0,1 CP_MODE=allgather,DUMP_DIR=$D/g2_main/dump &
one prwhole pr 2,3 CP_MODE=allgather,CP_MINP=1000000000,DUMP_DIR=$D/g2_prwhole/dump &
one tokens main 4,5 CP_MODE=allgather,CP_TYPECHECK=0,UPD2_GATHER=1 &
wait
note DIAG2_DONE
