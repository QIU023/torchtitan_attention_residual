#!/bin/bash
# Locate the inexact VPP2 resume: on main b663c1673, fsdp2 x pp2 x vpp2, DistMuon (arm base_muon) and AdamW (arm base).
# Per arm: r1 from the seed (cold cache, saves step 5), r1b again on a copy of r1's cache, r2b resumes r1b's step 5 on
# a copy of r1b's cache. Resume is exact when r1b and r2b agree on steps 6-10. Waits for DONE_WARM. No deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; V=$S/verify_stack_1013; K=$(cd "$(dirname "$0")" && pwd)
T=${T:-$S/wt_mainref}; O=${O:-$V/resume_vpp_main}
until [ -f $V/resume2/DONE_WARM ]; do sleep 30; done
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <cache> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$S/main_pins:$K:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) loads=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
SEED=${SEED:-$O/seed}
[ -d $SEED/step-0 ] || run seed 0 $O/cache_seed MM18_ARM=base MM18_SEED_OUT=$SEED MODULE=mm18_probe CONFIG=seed MM18_STEPS=1
[ -d $SEED/step-0 ] || { note "no seed, stop"; exit 1; }
arm() {  # <arm> <gpus>
  local a=$1 gpus=$2 deg=dp=2,pp=2,vpp=2
  local common="MM18_ARM=$a MM18_CELL=$deg MM18_SEED=$SEED/step-0 MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5"
  run r1_$a $gpus $O/cache_$a $common
  cp -r $O/cache_$a $O/cache_${a}_r1b
  run r1b_$a $gpus $O/cache_${a}_r1b $common
  cp -r $O/cache_${a}_r1b $O/cache_${a}_r2b
  mkdir -p $O/r2b_$a/out/checkpoint && cp -r $O/r1b_$a/out/checkpoint/step-5 $O/r2b_$a/out/checkpoint/
  run r2b_$a $gpus $O/cache_${a}_r2b $common MM18_LOAD_STEP=5
}
( arm base_muon 0,1,2,3 ) & ( arm base 4,5,6,7 ) & wait
python3 $K/compare_vpp_resume.py $O >> $PR 2>&1
note done; echo done > $O/DONE
