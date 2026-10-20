#!/bin/bash
# After the 20-step control: one step of main and of the PR (dynamic CP on, no split) logging clip_grad_norm_'s mesh groups.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; D=$S/diag4380; K=$D/kit
until [ $(grep -c "DIAG3_DONE" $D/progress3.txt 2>/dev/null) -ge 3 ]; do sleep 10; done
source /workspace/venv_1006/bin/activate
one() {  # <name> <tree> <gpus> <envs>
  local name=$1 W=$D/$2 gpus=$3 envs=$4 O=$D/m_$1; rm -rf $O $D/ccm_$1; mkdir -p $O; cp -r $D/cache_w $D/ccm_$1
  ( cd $W && env $(echo $envs | tr ',' ' ') CP_STEPS=1 CP_SEED=$D/seed/step-0 PYTHONPATH=$K:$S/main_pins_1016:. \
      TORCHINDUCTOR_CACHE_DIR=$D/ccm_$1/ic TRITON_CACHE_DIR=$D/ccm_$1/tc CUDA_VISIBLE_DEVICES=$gpus NGPU=2 LOG_RANK=0 \
      MODULE=meshlog_probe CONFIG=cell timeout 1200 ./run_train.sh --output-dir $O/out > $O/run.log 2>&1 )
  echo "$(date +%H:%M:%S) $name rc=$? $(grep -a -o 'MESHLOG.*' $O/run.log | head -1)" >> $D/progress_mesh.txt; rm -rf $O/out $D/ccm_$1
}
one main main 0,1 CP_MODE=allgather &
one prwhole pr 2,3 CP_MODE=allgather,CP_MINP=1000000000 &
wait
echo MESH_DONE >> $D/progress_mesh.txt
