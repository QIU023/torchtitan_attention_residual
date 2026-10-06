#!/bin/bash
# 10-06, MoonEP review head moonep_review1 = 16ff9da7f (rebased onto main 3f087cf15, op region names from the modules,
# the hook test, the refusal test without disable_cuda_graphs) on 4 x H100, torch from venv_src as on 10-05:
#   A. GPU tests: test_moonep.py and test_moe_shared_experts_stream.py (expected 11 passed);
#   B. the body's end-to-end cells, 20 steps (h100_moonep_final_1005.sh part B): standard (twice), MoonEP, MoonEP FullAC,
#      standard and MoonEP at dp_shard 4 x EP 2 and HSDP 2 x 2 x EP 2. One cache warmed by a 1-step run of every cell on
#      both the new head and the old head 6e3de1b8d; every measured run on a copy of it. The new head runs first; the
#      old head on the same cache separates a change of code from a change of cache against the 10-05 table.
#   C. plain main 3f087cf15, the standard cell only: equal to the new head's standard cell when the shift from the 10-05
#      table is main's own. Every run imports local/mpp_shim (torch 68e0ae4 lacks param_dtype_override_fn, which main
#      passes as None since #5068); the shim announces itself in every run.log.
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/moonep_recheck_1006; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
git -C $M/tt fetch -q origin moonep_review1
git -C $M/tt fetch -q upstream main
NEW=$M/w/mep_16ff; OLD=$M/w/mep_head; MAIN=$M/w/main_3f08
[ -d $NEW ] || git -C $M/tt worktree add -q --detach $NEW 16ff9da7f
[ -d $MAIN ] || git -C $M/tt worktree add -q --detach $MAIN 3f087cf15
for W in $NEW $OLD $MAIN; do note "$(basename $W) $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"; done
note "origin/moonep_review1 $(git -C $M/tt rev-parse --short origin/moonep_review1); torch $(python -c 'import torch; print(torch.__version__, torch.version.git_version[:9])')"
[ -s $O/gpu_tests.log ] || ( cd $NEW && timeout 1800 python -m pytest -q -p no:cacheprovider tests/unit_tests/gpu/test_moonep.py \
    tests/unit_tests/gpu/test_moe_shared_experts_stream.py > $O/gpu_tests.log 2>&1 )
note "gpu tests rc=$? $(tail -1 $O/gpu_tests.log)"
cell() {  # <tree dir> <name> <config> <cache> <steps> [module]
  local W=$1 D=$O/$2 mod=${6:-moonep_probe_1003}; rm -rf $D; mkdir -p $D
  ( cd $W && PROBE_STEPS=$5 PROBE_DET=1 PYTHONPATH=$K/local/mpp_shim:$K:. TORCHINDUCTOR_CACHE_DIR=$4/ic TRITON_CACHE_DIR=$4/tc \
    NGPU=4 LOG_RANK=0,1,2,3 MODULE=$mod CONFIG=$3 timeout 1800 \
    ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$2 rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a '^\[rank0\]' | grep -a -o 'step: *[0-9]* *loss: *[0-9.]* *grad_norm: *[0-9.]*' | tail -1)"
}
CELLS="standard moonep moonep_full standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp"
C=$O/cache_e; rm -rf $C; mkdir -p $C
for t in new old; do W=$NEW; [ $t = old ] && W=$OLD
  for c in $CELLS; do cell $W warm_${t}_$c ${c}_cell $C 1; done
done
cell $MAIN warm_main_standard standard_cell $C 1 moonep_probe_main_1006
for t in new old; do W=$NEW; [ $t = old ] && W=$OLD
  for c in $CELLS; do rm -rf $O/cc; cp -r $C $O/cc; cell $W num_${t}_$c ${c}_cell $O/cc 20; done
  rm -rf $O/cc; cp -r $C $O/cc; cell $W num_${t}_standard_b standard_cell $O/cc 20
  note "e2e $t done"
done
rm -rf $O/cc; cp -r $C $O/cc; cell $MAIN num_main_standard standard_cell $O/cc 20 moonep_probe_main_1006
rm -rf $O/cc $C
note "all done"; echo done > $O/DONE
