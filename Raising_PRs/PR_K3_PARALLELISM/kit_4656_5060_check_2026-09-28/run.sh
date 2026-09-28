#!/bin/bash
# 5060: main f35966713 against #4656 5d469fdf3, seed 42, deterministic, one warm cache per recipe, 10 steps:
#  mm   = torchtitan_recipes.tests.b200:kimi_k3_debugmodel_mm (fsdp2 x tp2 with SP x ep2, SPMD type checking on), 4 GPUs
#  pp   = torchtitan_recipes.tests.b200:kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4, 8 GPUs
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/mmcheck/res; mkdir -p $R; V=/workspace/venv_bfx9/bin; SHIM=$S/lbplan/pr5_torch_compat_shim.patch
declare -A T=([main]=$S/wt_s_main [pr]=$S/wt_4656v2)
for t in main pr; do git -C ${T[$t]} apply $SHIM || { echo "shim failed on $t" >> $R/progress.txt; exit 1; }; done
cell() {  # <name> <tree> <cache> <steps> <nproc> <config> <gpus>
  local D=$R/$1; rm -rf $D; mkdir -p $D
  ( cd ${T[$2]} && CUDA_VISIBLE_DEVICES=$7 PYTHONPATH=/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$3/ic TRITON_CACHE_DIR=$3/tc \
    timeout 1800 $V/torchrun --nproc_per_node=$5 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module torchtitan_recipes.tests.b200 --config $6 --debug.seed 42 --debug.deterministic \
    --training.steps $4 --metrics.log_freq 1 --dump-folder $D/out > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; echo "$(date +%H:%M:%S) $1 $(cat $D/rc)" >> $R/progress.txt; rm -rf $D/out
}
for spec in "mm:kimi_k3_debugmodel_mm:4:0,1,2,3" "pp:kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4:8:0,1,2,3,4,5,6,7"; do
  IFS=: read -r tag cfg n gpus <<< "$spec"
  C0=$R/cache_$tag; rm -rf $C0; mkdir -p $C0
  for t in main pr; do cell w_${tag}_$t $t $C0 1 $n $cfg $gpus; done
  for t in main pr; do rm -rf $R/cache_c; cp -r $C0 $R/cache_c; cell ${tag}_$t $t $R/cache_c 10 $n $cfg $gpus; done
  rm -rf $C0 $R/cache_c
done
for t in main pr; do git -C ${T[$t]} checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in main pr; do echo "$t $(git -C ${T[$t]} rev-parse --short HEAD) dirty=$(git -C ${T[$t]} status --short | wc -l)"; done > $R/worktrees_after.txt
echo done > $R/DONE
