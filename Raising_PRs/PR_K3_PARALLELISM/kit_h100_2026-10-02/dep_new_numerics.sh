#!/bin/bash
# 10-02 (the user: one old/new pair identical over 100 steps, then everything on the new code): waits for
# dep_refactor_check.sh, then at seq 2048, 224 px, one image per sample, M16, cost ratio 0.116, deterministic, automatic
# dynamic shapes off, 100 steps, every run on a copy of one cache warmed by one step of each mode:
#   bubble on the measured tree a93cd48ea (~/mep/w/dep) and on the refactor b2a57dff7 (~/mep/w/dep_new): the pair;
#   DEP off twice (the floor), K2.5 on b2a57dff7: the rest of the numerics table, bubble taken from the pair.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src
OLD=$M/w/dep; NEW=$M/w/dep_new; O=$M/results/dep_new_numerics; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/dep_refactor_check/DONE ]; do sleep 20; done
for t in $OLD $NEW; do note "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done
run() {  # <tree> <name> <config> <cache> <steps>
  local tree=$1 name=$2 cfg=$3 cache=$4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && DEPN_STEPS=$5 PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run $NEW warm_$c $c $C0 1; done
for x in bubble_old:$OLD:w_dep_bubble bubble_new:$NEW:w_dep_bubble off_a:$NEW:w_dep_off off_b:$NEW:w_dep_off k25:$NEW:w_dep_k25; do
  IFS=: read -r name tree cfg <<< "$x"
  rm -rf $O/cc; cp -r $C0 $O/cc; run $tree num_$name $cfg $O/cc 100
  if [ $name = bubble_new ]; then
    $V/bin/python $KN/cmp_steps.py $O/num_bubble_old/run.log $O/num_bubble_new/run.log > $O/cmp_old_new.txt 2>&1
    note "old vs new bubble: $(tail -1 $O/cmp_old_new.txt)"
  fi
done
rm -rf $O/cc $C0
note "numerics done"
echo done > $O/DONE
