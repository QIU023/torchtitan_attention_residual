#!/bin/bash
# Warm determinism for main's DistMuon arm at fsdp2 x pp2 x vpp2: r1c reruns r1b from the seed on another copy of r1b's cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; V=$S/verify_stack_1013; K=$(cd "$(dirname "$0")" && pwd)
T=$S/wt_mainref; O=$V/resume_vpp_main2; SEED=$V/resume_vpp_main/seed; PR=$O/progress.txt
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
a=base_muon; D=$O/r1c_$a; mkdir -p $D
cp -r $O/cache_${a}_r1b $O/cache_${a}_r1c
( cd $T && env MM18_ARM=$a MM18_CELL=dp=2,pp=2,vpp=2 MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5 \
  PYTHONPATH=$S/main_pins:$K:$T TORCHINDUCTOR_CACHE_DIR=$O/cache_${a}_r1c/ic TRITON_CACHE_DIR=$O/cache_${a}_r1c/tc \
  CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 LOG_RANK=0,1,2,3 timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
note "r1c_$a rc=$? last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
echo done > $O/DONE_R1C
