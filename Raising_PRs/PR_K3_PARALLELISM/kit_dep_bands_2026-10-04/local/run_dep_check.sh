#!/bin/bash
# DEP band placement (6b580e438) against the review head (6f5312fab) on 8 x 5060, torch 0928: the GPU unit test on
# the corrected tree, then the DEP cell for 10 deterministic steps on copies of one cache warmed by the head.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; R=$S/dep_1004
HEAD_T=$S/dep_head_tree; FIX_T=$S/wt_dep_1004; O=$R/runs; rm -rf $O; mkdir -p $O; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_0928/bin/activate
note "fixed $(git -C $FIX_T rev-parse --short HEAD) dirty=$(git -C $FIX_T status --short | wc -l)"
( cd $FIX_T && CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1800 python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -rA -p no:cacheprovider > $O/gpu_test.log 2>&1 )
note "gpu test rc=$? $(tail -1 $O/gpu_test.log)"
run() {  # <tree> <name> <cache> <steps>
  local T=$1 name=$2 cache=$3 D=$O/$2; rm -rf $D; mkdir -p $D
  ( cd $T && PROBE_STEPS=$4 PYTHONPATH=$R:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc NGPU=8 \
    LOG_RANK=0,1,2,3,4,5,6,7 MODULE=dep_probe_1004 CONFIG=dep_cell timeout 2400 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) | $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: [^\[]*' | head -1)"
}
C0=$O/cache0; mkdir -p $C0
run $HEAD_T warm_head $C0 1
for x in head:$HEAD_T fixed:$FIX_T; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; run ${x#*:} dep_${x%%:*} $O/cache_c 10; done
rm -rf $O/cache_c $C0
note "done"; echo done > $O/DONE
