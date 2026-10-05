#!/bin/bash
# dep_review1 6cda7daca (the DEP stage composed with the AttnRes stage instead of subclassing it) against 8518f473f
# on 4 x H100: the DEP NCCL unit test on the new head, then the bubble and K2.5 cells of the body's numerics table
# (seq 2048, 224 px, M16, cost ratio 0.116, deterministic, automatic dynamic shapes off), 20 steps, old and new on
# copies of one cache warmed by both trees; cmp_steps.py old vs new.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src
O=$M/results/dep_decouple; P=$O/progress.txt; rm -rf $O; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
git -C $M/tt fetch -q origin dep_review1
OLD=$M/w/dep_fig11; NEW=$M/w/dep_1005
[ -d $NEW ] || git -C $M/tt worktree add -q --detach $NEW 6cda7daca
for t in $OLD $NEW; do note "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done
( cd $NEW && timeout 1800 $V/bin/python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -rA > $O/gputest.log 2>&1 )
note "gpu test rc=$? $(tail -1 $O/gputest.log)"
run() {  # <tree> <name> <config> <cache> <steps>
  local tree=$1 name=$2 cfg=$3 cache=$4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && DEPN_STEPS=$5 PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3600 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
C=$O/cache; mkdir -p $C
for c in w_dep_bubble w_dep_k25; do for x in old:$OLD new:$NEW; do run ${x#*:} warm_${x%%:*}_$c $c $C 1; done; done
for c in w_dep_bubble w_dep_k25; do
  for x in old:$OLD new:$NEW; do rm -rf $O/cc; cp -r $C $O/cc; run ${x#*:} num_${x%%:*}_$c $c $O/cc 20; done
  $V/bin/python $KN/cmp_steps.py $O/num_old_$c/run.log $O/num_new_$c/run.log > $O/cmp_$c.txt 2>&1
  note "$c old vs new: $(tail -1 $O/cmp_$c.txt)"
done
rm -rf $O/cc $C
note "all done"; echo done > $O/DONE
