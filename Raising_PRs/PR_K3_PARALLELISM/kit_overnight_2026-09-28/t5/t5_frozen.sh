#!/bin/bash
# T5: the new DEP (k3_pp_mm bb3e38d4a) on 8 x 5060, validation only (audit section 8): CPU tests on this
# environment, the committed B200 DEP cell for 10 steps, numerics on one warm cache (DEP off twice, K2.5
# mode twice, bubble once; gradients dumped at steps 1 and 2), and a traced step for each configuration.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
Q=$S/overnight; P=$Q/progress.txt; V=/workspace/venv_bfx9/bin; W=$S/wt_dep_new; R=$Q/t5
SHIM=$S/lbplan/pr5_torch_compat_shim.patch
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
mkdir -p $R
( cd $W && timeout 1800 $V/python -m pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_integration_test_definitions.py -q -p no:cacheprovider > $Q/t5_dep.pytest.log 2>&1 ); note "t5_dep pytest rc=$? $(tail -1 $Q/t5_dep.pytest.log)"
git -C $W apply $SHIM || { note "shim failed on DEP"; exit 1; }
run() {  # <name> <module> <config> <cache> <steps> [extra args...]
  local name=$1 module=$2 cfg=$3 cache=$4 steps=$5; shift 5
  local D=$R/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$Q/dep:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 2400 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $module --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  echo "rc=$?" > $D/rc; note "$name $(cat $D/rc)"
}
SEED="--debug.seed 42 --debug.deterministic"
# the committed cell, as CI runs it
mkdir -p $R/cache_cell; run cell torchtitan_recipes.tests.b200 kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep $R/cache_cell 10; rm -rf $R/cache_cell
# one warm cache for every configuration
C0=$R/cache0; mkdir -p $C0
for c in dep_off dep_k25 dep_bubble; do run warm_$c dep_new $c $C0 1 $SEED; done
for x in "off_a:dep_off" "off_b:dep_off" "k25_a:dep_k25" "k25_b:dep_k25" "bubble_a:dep_bubble"; do
  n=${x%%:*}; c=${x#*:}; rm -rf $R/cache_c; cp -r $C0 $R/cache_c
  DEP_GRAD_DUMP=$R/$n/grads DEP_GRAD_STEPS=1,2 DEPN_MEM_OUT=$R/$n/mem run $n dep_new $c $R/cache_c 3 $SEED
done
for c in dep_off dep_k25 dep_bubble; do
  rm -rf $R/cache_c; cp -r $C0 $R/cache_c
  DEPN_MEM_OUT=$R/trace_$c/mem run trace_$c dep_new $c $R/cache_c 12 $SEED --profiler.enable-profiling \
    --profiler.profile-freq 10 --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
done
rm -rf $R/cache0 $R/cache_c
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
echo "$W $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)" >> $Q/worktrees_after.txt
