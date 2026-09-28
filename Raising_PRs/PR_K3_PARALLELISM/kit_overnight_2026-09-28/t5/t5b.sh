#!/bin/bash
# T5b: DEP bb3e38d4a unchanged, every cell with CUDA_MODULE_LOADING=EAGER (the default lazy loading
# deadlocks step 1: a first kernel load waits on the receives vision_dep posts at step start).
# Gate: the committed B200 cell for 10 steps; then numerics on one warm cache and a traced step each.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
Q=$S/overnight; P=$Q/progress.txt; V=/workspace/venv_bfx9/bin; W=$S/wt_dep_new; R=$Q/t5
SHIM=$S/lbplan/pr5_torch_compat_shim.patch
export CUDA_MODULE_LOADING=EAGER
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
git -C $W apply $SHIM || { note "t5b shim failed on DEP"; exit 1; }
run() {  # <name> <module> <config> <cache> <steps> [extra args...]
  local name=$1 module=$2 cfg=$3 cache=$4 steps=$5; shift 5
  local D=$R/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$Q/dep:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1500 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $module --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; note "$name rc=$rc"; return $rc
}
SEED="--debug.seed 42 --debug.deterministic"
mkdir -p $R/cache_cell
if run cell_eager torchtitan_recipes.tests.b200 kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep $R/cache_cell 10; then
  C0=$R/cache0; rm -rf $C0; mkdir -p $C0
  for c in dep_off dep_k25 dep_bubble; do run warm_${c}_eager dep_new $c $C0 1 $SEED; done
  for x in "off_a:dep_off" "off_b:dep_off" "k25_a:dep_k25" "k25_b:dep_k25" "bubble_a:dep_bubble"; do
    n=${x%%:*}; c=${x#*:}; rm -rf $R/cache_c; cp -r $C0 $R/cache_c
    DEP_GRAD_DUMP=$R/$n/grads DEP_GRAD_STEPS=1,2 DEPN_MEM_OUT=$R/$n/mem run $n dep_new $c $R/cache_c 3 $SEED
  done
  for c in dep_off dep_k25 dep_bubble; do
    rm -rf $R/cache_c; cp -r $C0 $R/cache_c
    DEPN_MEM_OUT=$R/trace_$c/mem run trace_$c dep_new $c $R/cache_c 12 $SEED --profiler.enable-profiling \
      --profiler.profile-freq 10 --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
  done
else
  note "t5b: the committed cell fails with eager loading too, stopping"
fi
rm -rf $R/cache_cell $R/cache0 $R/cache_c
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
echo "$W $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)" >> $Q/worktrees_after.txt
note "T5b done"
echo done > $Q/T5B_DONE
