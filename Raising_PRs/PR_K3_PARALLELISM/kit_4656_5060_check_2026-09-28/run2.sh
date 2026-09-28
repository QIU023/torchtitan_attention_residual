#!/bin/bash
# 5060: the mm cell with type checking off, main f35966713 against #4656 5d469fdf3, seed 42, deterministic,
# one warm cache, 10 steps, 4 GPUs.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/mmcheck/res2; mkdir -p $R; V=/workspace/venv_bfx9/bin; SHIM=$S/lbplan/pr5_torch_compat_shim.patch
declare -A T=([main]=$S/wt_s_main [pr]=$S/wt_4656v2)
for t in main pr; do git -C ${T[$t]} apply $SHIM || { echo "shim failed on $t" >> $R/progress.txt; exit 1; }; done
cell() {  # <name> <tree> <cache> <steps>
  local D=$R/$1; rm -rf $D; mkdir -p $D
  ( cd ${T[$2]} && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$S/mmcheck:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$3/ic TRITON_CACHE_DIR=$3/tc \
    timeout 1800 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module mm_notc --config mm_notc --debug.seed 42 --debug.deterministic \
    --training.steps $4 --metrics.log_freq 1 --dump-folder $D/out > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $1 $(cat $D/rc)" >> $R/progress.txt; rm -rf $D/out
}
C0=$R/cache0; rm -rf $C0; mkdir -p $C0
for t in main pr; do cell w_$t $t $C0 1; done
for t in main pr; do rm -rf $R/cache_c; cp -r $C0 $R/cache_c; cell mm_$t $t $R/cache_c 10; done
rm -rf $C0 $R/cache_c
for t in main pr; do git -C ${T[$t]} checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in main pr; do echo "$t $(git -C ${T[$t]} rev-parse --short HEAD) dirty=$(git -C ${T[$t]} status --short | wc -l)"; done > $R/worktrees_after.txt
echo done > $R/DONE
