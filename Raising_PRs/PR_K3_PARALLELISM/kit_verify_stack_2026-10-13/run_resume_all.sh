#!/bin/bash
# Checkpoint resume on the stack, arm mxqat_muon, five FSDP layouts, all on warm copies of one cache lineage:
# r1 from the seed on a cold cache (warms it), r1b from the seed on a copy (saves step 5 and 10), r2b resumes r1b's
# step 5 on another copy, r1c reruns r1b on a third copy. Exact resume: r1b == r2b on steps 6-10, with r1b == r1c.
SEED=$1 O=$2
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$S/verify_stack_1013/kit_more; T=$S/wt_mxqat_fix
mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date -u +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <name> <gpus> <cache> <env...>
  local name=$1 gpus=$2 cache=$3; shift 3; local D=$O/$name; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" MM18_ARM=mxqat_muon PYTHONPATH=$S/main_pins:$K:$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log)
  note "$name rc=$rc last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) loads=$(echo "$clean" | grep -c 'Loading the checkpoint')"
}
layout() {  # <name> <degrees> <gpus>
  local name=$1 deg=$2 gpus=$3 c="MM18_CELL=$2 MM18_SEED=$SEED MODULE=mm18_probe CONFIG=cell_resume MM18_CKPT_INTERVAL=5"
  run r1_$name $gpus $O/cache_$name $c
  cp -r $O/cache_$name $O/cache_${name}_r1b
  run r1b_$name $gpus $O/cache_${name}_r1b $c
  cp -r $O/cache_${name}_r1b $O/cache_${name}_r2b; cp -r $O/cache_${name}_r1b $O/cache_${name}_r1c
  mkdir -p $O/r2b_$name/out/checkpoint && cp -r $O/r1b_$name/out/checkpoint/step-5 $O/r2b_$name/out/checkpoint/
  run r2b_$name $gpus $O/cache_${name}_r2b $c MM18_LOAD_STEP=5
  run r1c_$name $gpus $O/cache_${name}_r1c $c
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
( layout fsdp2 dp=2 0,1 ) & ( layout ep2_fsdp2 dp=2,ep=2 2,3 ) & ( layout fsdp2_pp2 dp=2,pp=2 4,5,6,7 ) & wait
( layout fsdp2_pp2_vpp2 dp=2,pp=2,vpp=2 0,1,2,3 ) & ( layout fsdp2_cp2 dp=2,cp=2 4,5,6,7 ) & wait
python3 $K/compare_resume_warm.py $O >> $PR 2>&1
note done; echo done > $O/DONE
