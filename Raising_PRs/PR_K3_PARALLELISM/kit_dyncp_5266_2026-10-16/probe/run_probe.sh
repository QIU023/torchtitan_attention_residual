#!/bin/bash
# Goal step 1: dynamic CP off and on (threshold 128, every image split) at CP2 dp1 on the stack, one step each, dumps.
T=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_mxqat_fix
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/dyncp_fix; C=$S/k3v_run/combo; O=$G/probe; mkdir -p $O
note() { echo "$(date +%H:%M:%S) $*" >> $O/progress.txt; }
. /workspace/venv_1006/bin/activate
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
run() {  # <name> <env...>
  local name=$1; shift; local D=$O/$name; mkdir -p $D
  ( cd $T && env "$@" K3C_CELL="cp=2" K3C_MAXP=2304 K3C_STEPS=1 K3C_SEED=$C/out1/seed/step-0 DUMP_DIR=$D/dump MODULE=dyncp_probe CONFIG=cell \
    PYTHONPATH=$S/main_pins:$G:$C:$T TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    CUDA_VISIBLE_DEVICES=0,1 NGPU=2 LOG_RANK=0,1 timeout 1500 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  note "$name rc=$? dumps=$(ls $D/dump 2>/dev/null | wc -l)"
}
run off
run on K3C_DYNCP=128
note done
