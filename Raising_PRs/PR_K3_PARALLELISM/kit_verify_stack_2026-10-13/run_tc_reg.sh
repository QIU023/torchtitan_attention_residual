#!/bin/bash
# The two type-checked MX QAT cells with the torchao Functions registered as local, 10 steps each. No deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd); T=$S/wt_mxqat_fix
O=$S/verify_stack_1013/typecheck_reg; mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <config>
  local name=$1 gpus=$2 cfg=$3 D=$O/$1; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env MODULE=stack_probe_tc_reg CONFIG=$cfg PROBE_STEPS=10 PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$O/cache_$name/ic \
    TRITON_CACHE_DIR=$O/cache_$name/tc CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) $(grep -hoE 'SpmdTypeError: .{0,120}' $D/run.log | head -1)"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
( run mm_mxqat_reg 0,1,2,3 mm_mxqat_reg ) & ( run mm_muon_mxqat_reg 4,5 mm_muon_mxqat_reg ) & wait
note done; echo done > $O/DONE
