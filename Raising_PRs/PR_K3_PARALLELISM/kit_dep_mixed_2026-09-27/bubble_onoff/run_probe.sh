#!/bin/bash
# After run.sh: step-1 and step-2 gradients of the bubble on and off cells on copies of the same warm cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_bubble; W=/tmp/wt_depnew
until [ -f $R/ALL_DONE ]; do sleep 15; done
cd $W && git apply $S/lbplan/pr5_torch_compat_shim.patch || exit 1
run() {  # <name> <config> <gpus>
  local D=$R/$1; rm -rf $D $R/cache_$1; mkdir -p $D/tmp $D/grads; cp -r $R/cache0 $R/cache_$1
  DEP_GRAD_DUMP=$D/grads CUDA_VISIBLE_DEVICES=$3 TMPDIR=$D/tmp PYTHONPATH=$R:/tmp/attn_gym_up:. \
  TORCHINDUCTOR_CACHE_DIR=$R/cache_$1/ic TRITON_CACHE_DIR=$R/cache_$1/tc \
  timeout 1800 /workspace/venv_bfx9/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module dep_bubble_probe --config $2 \
    --debug.seed 42 --debug.deterministic --training.steps 3 --metrics.log_freq 1 --dump-folder $D/out > $D/run.log 2>&1
  echo "rc=$?" > $D/rc
}
run probe_on probe_on 0,1,2,3 &
run probe_off probe_off 4,5,6,7 &
wait
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
git -C $W status --short > $R/worktree_after_probe.txt
/workspace/venv_bfx9/bin/python $R/cmp_grads.py $R/probe_on/grads $R/probe_off/grads > $R/cmp_grads.txt 2>&1
echo done > $R/PROBE_DONE
