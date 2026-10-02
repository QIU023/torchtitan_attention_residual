#!/bin/bash
# 10-02 (the user: the MoonEP refactor should not change numerics, check it first), after the DEP jobs: the MoonEP refactor
# 7b73b1ed5 (~/mep/w/moonep_new) against the H100-checked cec583a44 (~/mep/w/moonep), real MoonEP 33327eb:
#   the GPU test on 7b73b1ed5 (5 cases, two GPUs);
#   one cell (the removed h100 recipe's shape, moonep_probe_1002b.py), deterministic, 100 steps, each tree on a copy of
#   one warm cache; loss and grad norm compared step by step (cmp_steps.py).
M=~/mep; KN=~/kit/kit_h100_2026-10-02; V=$M/venv_src; OLD=$M/w/moonep; NEW=$M/w/moonep_new
O=$M/results/moonep_refactor_check; P=$O/progress.txt; mkdir -p $O
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/dep_new_gputest/DONE ]; do sleep 20; done
for t in $OLD $NEW; do note "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done
. $V/bin/activate
( cd $NEW && CUDA_VISIBLE_DEVICES=0,1 timeout 1800 python -m pytest tests/unit_tests/gpu/test_moonep.py -q -rA > $O/gpu_test.log 2>&1 )
note "gpu test (new) rc=$? $(tail -1 $O/gpu_test.log)"
cell() {  # <tree> <name> <cache> <steps>
  local tree=$1 name=$2 cache=$3; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && PROBE_STEPS=$4 PYTHONPATH=$KN:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1800 torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_probe_1002b --config moonep_cell --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc"
}
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
cell $OLD warm $C0 1
for x in old:$OLD new:$NEW; do rm -rf $O/cc; cp -r $C0 $O/cc; cell ${x#*:} num_${x%%:*} $O/cc 100; done
rm -rf $O/cc $C0
python $KN/cmp_steps.py $O/num_old/run.log $O/num_new/run.log > $O/cmp_old_new.txt 2>&1
note "old vs new: $(tail -1 $O/cmp_old_new.txt)"
echo done > $O/DONE
