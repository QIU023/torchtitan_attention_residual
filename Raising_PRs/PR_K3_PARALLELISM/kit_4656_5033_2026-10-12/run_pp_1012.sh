#!/bin/bash
# The B200 suite's 8-GPU PP cell (FSDP 2, TP 2, EP 2, PP 2, VPP 4): #4963 rebased (d90de637b) against its base, the #4656
# review head (ba8f071d1), 20 steps on copies of one cache warmed by a 10-step run of the base; then the stack branch
# k3_mxqat_stack_20261012 (DEP on in that cell's recipe), 10 steps on its own cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
A=$S/m4656_5033; O=$A/pp4963; mkdir -p $O; PR=$O/progress.txt
declare -A TREE=([c2]=$S/wt_4656rb [pra]=$S/wt_4963rb [stack]=$S/wt_mxqat)
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <tree> <name> <cache> <env...>
  local T=${TREE[$1]} name=$2 cache=$3; shift 3; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $T && env "$@" PYTHONPATH=$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
for t in c2 pra stack; do note "$t $(git -C ${TREE[$t]} rev-parse --short HEAD) dirty=$(git -C ${TREE[$t]} status --short | wc -l)"; done
C=$A/cache_pra; mkdir -p $C
run c2 warm_pp_c2 $C MODULE=attnres_probe_1012 CONFIG=pp_cell PROBE_STEPS=10
for t in c2 pra; do rm -rf $A/cache_prc; cp -r $C $A/cache_prc; run $t pp_$t $A/cache_prc MODULE=attnres_probe_1012 CONFIG=pp_cell PROBE_STEPS=20; done
rm -rf $A/cache_prc
run stack pp_stack_dep $A/cache_stack MODULE=tests.integration_tests.b200 CONFIG=kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4
note "done"; echo done > $O/DONE
