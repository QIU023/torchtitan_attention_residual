#!/bin/bash
# DEP transport fix 55e4274c4 on 8 x 5060, default (lazy) module loading: CPU tests, the new NCCL
# GPU test, the committed B200 cell as the gate, then numerics on one warm cache (DEP off twice,
# K2.5 twice, bubble once; gradients at steps 1 and 2; per-rank memory) and a traced step each.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_fix; P=$R/progress.txt; V=/workspace/venv_bfx9/bin; W=$S/wt_dep_new; D=$S/overnight/dep
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
( cd $W && PYTHONPATH=/tmp/attn_gym_up:. timeout 1800 $V/python -m pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py \
  tests/unit_tests/cpu/test_kimi_k3_vision_dep.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py \
  tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py \
  tests/unit_tests/cpu/test_integration_test_definitions.py -q -p no:cacheprovider > $R/cpu.log 2>&1 )
note "cpu rc=$? $(tail -1 $R/cpu.log)"
( cd $W && PYTHONPATH=/tmp/attn_gym_up:. CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1200 $V/python -m pytest \
  tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -p no:cacheprovider > $R/gpu.log 2>&1 )
note "gpu test rc=$? $(tail -1 $R/gpu.log)"
git -C $W apply $R/shim_5dc97a3e7_pp.patch || { note "shim failed"; exit 1; }
run() {  # <name> <module> <config> <cache> <steps> [extra args...]
  local name=$1 module=$2 cfg=$3 cache=$4 steps=$5; shift 5
  local O=$R/$name; rm -rf $O; mkdir -p $O
  ( cd $W && PYTHONPATH=$D:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1500 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $module --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --dump-folder $O/out "$@" > $O/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $O/rc; note "$name rc=$rc"; return $rc
}
SEED="--debug.seed 42 --debug.deterministic"
mkdir -p $R/cache_cell
if run cell torchtitan_recipes.tests.b200 kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep $R/cache_cell 10; then
  C0=$R/cache0; rm -rf $C0; mkdir -p $C0
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
else
  note "the committed cell fails, stopping"
fi
rm -rf $R/cache_cell $R/cache0 $R/cache_c
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
note "T5c done dirty=$(git -C $W status --short | wc -l)"
echo done > $R/T5C_DONE
