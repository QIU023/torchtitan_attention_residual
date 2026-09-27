#!/bin/bash
# DEP on the new head: pp4 x vp2 vit_dep with the bubble on and off, seed 42, one warm cache, 10 steps; each setting run twice.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_bubble; W=/tmp/wt_depnew
cd $W && git apply $S/lbplan/pr5_torch_compat_shim.patch || exit 1
run() {  # <name> <config> <gpus> <cache> <steps>
  local D=$R/$1; rm -rf $D; mkdir -p $D/tmp
  CUDA_VISIBLE_DEVICES=$3 TMPDIR=$D/tmp PYTHONPATH=$R:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$4/ic TRITON_CACHE_DIR=$4/tc \
  timeout 1800 /workspace/venv_bfx9/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module dep_bubble --config $2 \
    --debug.seed 42 --debug.deterministic --training.steps $5 --metrics.log_freq 1 --dump-folder $D/out > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
}
C0=$R/cache0; mkdir -p $C0
run warm_on dep_bubble_on 0,1,2,3 $C0 1
run warm_off dep_bubble_off 0,1,2,3 $C0 1
for c in on off on2 off2; do rm -rf $R/cache_$c; cp -r $C0 $R/cache_$c; done
run on dep_bubble_on 0,1,2,3 $R/cache_on 10 &
run off dep_bubble_off 4,5,6,7 $R/cache_off 10 &
wait
run on2 dep_bubble_on 0,1,2,3 $R/cache_on2 10 &
run off2 dep_bubble_off 4,5,6,7 $R/cache_off2 10 &
wait
git -C $W diff > $R/shim_applied.diff
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
git -C $W status --short > $R/worktree_after.txt
echo done > $R/ALL_DONE
