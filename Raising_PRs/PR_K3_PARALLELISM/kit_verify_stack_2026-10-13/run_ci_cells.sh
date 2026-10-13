#!/bin/bash
# Every Kimi K3 CI cell on the stack head against main b663c1673 (8 x 5060, venv_1006, main's pins first), the protocol
# of kit_stack_full_2026-10-12: each pair on copies of one cache warmed by 10 steps of both trees, 20 measured steps;
# then Elfie's MX QAT recipe and the MXFP4 checkpoint integration test on the stack. Fresh output dirs, no deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
A=$S/verify_stack_1013/ci_cells; O=$A/runs; mkdir -p $O; PR=$O/progress.txt
declare -A TREE=([main]=$S/wt_mainref [stack]=$S/wt_mxqat_fix)
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <tree> <name> <gpus> <cache> <env...>
  local T=${TREE[$1]} name=$2 gpus=$3 cache=$4; shift 4; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=$gpus \
    NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
pair() {  # <cfg> <gpus>: warm both trees on one cache, then 20-step runs on separate copies of it
  local cfg=$1 gpus=$2 C=$A/cache_$1; mkdir -p $C
  for t in main stack; do run $t warm_${t}_$cfg $gpus $C MODULE=stack_probe_1012 CONFIG=$cfg PROBE_STEPS=10; done
  for t in main stack; do cp -r $C ${C}_run_$t; run $t ${cfg}_$t $gpus ${C}_run_$t MODULE=stack_probe_1012 CONFIG=$cfg PROBE_STEPS=20; done
}
for t in main stack; do note "$t $(git -C ${TREE[$t]} rev-parse --short HEAD) dirty=$(git -C ${TREE[$t]} status --short | wc -l)"; done
( pair mm 0,1,2,3 ) &
( pair mm_muon 4,5; pair cp_ulysses 4,5 ) &
( pair cp_allgather 6,7; mkdir -p $A/cache_mxqat; run stack mx_qat_stack 6 $A/cache_mxqat MODULE=stack_probe_1012 CONFIG=mx_qat PROBE_STEPS=20 ) &
wait
pair pp_vpp4 0,1,2,3,4,5,6,7
( cd ${TREE[stack]} && PYTHONPATH=$S/main_pins:${TREE[stack]}:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$A/cache_mxqat/ic TRITON_CACHE_DIR=$A/cache_mxqat/tc CUDA_VISIBLE_DEVICES=0,1,2,3 \
  timeout 3000 python -m pytest -q -p no:cacheprovider tests/integration_tests/test_kimi_k3_mxfp4_checkpoint.py > $O/mxfp4_checkpoint_test.log 2>&1 )
note "mxfp4 checkpoint integration test rc=$? $(tail -1 $O/mxfp4_checkpoint_test.log)"
note "done"; echo done > $O/DONE
