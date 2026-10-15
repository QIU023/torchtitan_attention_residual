#!/bin/bash
# dep_dyncp with the probe: unpatched 2 steps, then (if K3C_PATCHED tree given) the patched tree 10 steps. No deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; R=$S/k3v_run/combo; O=$R/out1; T=$S/wt_mxqat_fix
. /workspace/venv_1006/bin/activate
name=$1 steps=$2; D=$O/$name; mkdir -p $D
( cd $T && env K3C_PROBE=1 K3C_STEPS=$steps K3C_CELL="dp=2,ep=2,pp=2,cp=2" K3C_MAXP=2304 K3C_DEP=1 K3C_DYNCP=256 K3C_SEED=$O/seed/step-0 \
  MODULE=combo_cells CONFIG=cell PYTHONPATH=$R/probe:$S/main_pins:$R:$T TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
echo "$(date +%H:%M:%S) $name rc=$?" >> $O/progress.txt
