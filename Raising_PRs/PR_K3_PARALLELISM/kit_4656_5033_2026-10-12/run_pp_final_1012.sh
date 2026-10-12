#!/bin/bash
# PR 4963 (a5c1128c9) against its base, the final PR 4656 commit (cfbbeac4f), B200 PP cell, 20 steps on copies of one cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; A=$S/m4656_5033/pp_final
until [ -f $S/m4656_5033/stack2/DONE ]; do sleep 20; done
. /workspace/venv_1006/bin/activate
declare -A TREE=([c1]=$S/wt_4656c1 [pra]=$S/wt_4963c1)
run() { local T=${TREE[$1]} name=$2 cache=$3 steps=$4; local D=$A/$name; mkdir -p $D
  ( cd $T && PYTHONPATH=$A:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 \
    MODULE=attnres_probe_1012 CONFIG=pp_cell PROBE_STEPS=$steps timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  echo "$(date +%H:%M:%S) $name rc=$? $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)" >> $A/progress.txt; rm -rf $D/out; }
run c1 warm $A/cache 10
for t in c1 pra; do rm -rf $A/cc; cp -r $A/cache $A/cc; run $t pp_$t $A/cc 20; done
rm -rf $A/cc; echo done > $A/DONE
