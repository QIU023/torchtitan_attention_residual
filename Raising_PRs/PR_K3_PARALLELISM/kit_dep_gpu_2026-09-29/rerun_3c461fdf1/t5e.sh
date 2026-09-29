#!/bin/bash
# DEP 3c461fdf1 (replica init under fork_rng) on 8 x 5060, lazy loading, no probe patch:
# CPU tests, the NCCL GPU test, the B200 cell, then DEP off / K2.5 / bubble on one warm cache
# with initial parameters, gradients (steps 1 and 2) and per-rank memory dumped.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/dep_fix; P=$R/progress_e.txt; V=/workspace/venv_bfx9/bin; W=$S/wt_dep_new; D=$S/overnight/dep
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "tree $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"
( cd $W && PYTHONPATH=/tmp/attn_gym_up:. timeout 1800 $V/python -m pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py \
  tests/unit_tests/cpu/test_kimi_k3_vision_dep.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py \
  tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py \
  tests/unit_tests/cpu/test_integration_test_definitions.py -q -p no:cacheprovider > $R/e_cpu.log 2>&1 )
note "cpu rc=$? $(tail -1 $R/e_cpu.log)"
( cd $W && PYTHONPATH=/tmp/attn_gym_up:. CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1200 $V/python -m pytest \
  tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -p no:cacheprovider > $R/e_gpu.log 2>&1 )
note "gpu test rc=$? $(tail -1 $R/e_gpu.log)"
git -C $W apply $R/shim_5dc97a3e7_pp.patch || { note "shim failed"; exit 1; }
run() {  # <name> <module> <config> <cache> <steps> [extra args...]
  local name=$1 module=$2 cfg=$3 cache=$4 steps=$5; shift 5
  local O=$R/$name; rm -rf $O; mkdir -p $O
  ( cd $W && PYTHONPATH=$R/probe:$D:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 1500 $V/torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $module --config $cfg --training.steps $steps --metrics.log_freq 1 \
    --dump-folder $O/out "$@" > $O/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $O/rc; note "$name rc=$rc"; return $rc
}
SEED="--debug.seed 42 --debug.deterministic"
mkdir -p $R/cache_cell
if run e_cell torchtitan_recipes.tests.b200 kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep $R/cache_cell 10; then
  C0=$R/cacheE; rm -rf $C0; mkdir -p $C0
  for c in dep_off dep_k25 dep_bubble; do run ewarm_$c dep_p2 $c $C0 1 $SEED; done
  for x in "e_off:dep_off" "e_k25:dep_k25" "e_bubble:dep_bubble"; do
    n=${x%%:*}; c=${x#*:}; rm -rf $R/cache_c; cp -r $C0 $R/cache_c
    DEP_PARAM_DUMP=$R/$n/params DEP_GRAD_DUMP=$R/$n/grads DEP_GRAD_STEPS=1,2 DEPN_MEM_OUT=$R/$n/mem \
      run $n dep_p2 $c $R/cache_c 3 $SEED
  done
else
  note "the committed cell fails, stopping"
fi
rm -rf $R/cache_cell $R/cacheE $R/cache_c
git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py
note "T5e done dirty=$(git -C $W status --short | wc -l)"
echo done > $R/T5E_DONE
