#!/bin/bash
# DEP a03f74981 on 4 x H100: CPU tests, the NCCL GPU test, numerics (pp2 x vpp4 x tp2 x ep2, every
# micro-batch with images; DEP off twice, K2.5, bubble; one warm cache; 3 steps; initial parameters,
# gradients at steps 1 and 2, per-rank peaks), then the widened model's step time over steps 12 to 30
# and one traced step per configuration. VENV picks the environment; SHIM, if set, is applied first.
M=~/mep; K=~/kit/dep; O=$M/results/dep; P=$O/progress.txt; mkdir -p $O
VENV=${VENV:-$M/venv_src}; . $VENV/bin/activate
W=$M/w/dep
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
if [ -n "${SHIM:-}" ]; then git -C $W apply $SHIM || { note "shim failed"; exit 1; }; fi
note "tree $(git -C $W rev-parse --short HEAD) torch $(python -c 'import torch; print(torch.__version__)') shim=${SHIM:-none}"
( cd $W && timeout 1800 python -m pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py \
  tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py \
  tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_integration_test_definitions.py \
  -q -p no:cacheprovider > $O/cpu.log 2>&1 )
note "cpu rc=$? $(tail -1 $O/cpu.log)"
( cd $W && CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1200 python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py \
  -q -p no:cacheprovider > $O/gpu.log 2>&1 )
note "gpu test rc=$? $(tail -1 $O/gpu.log)"
run() {  # <name> <config> <cache> <steps> [args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 2400 torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep4 --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; note "$name rc=$rc"; return $rc
}
SEED="--debug.seed 42 --debug.deterministic"
C0=$O/cache0; rm -rf $C0; mkdir -p $C0
for c in dep_off dep_k25 dep_bubble; do run warm_$c $c $C0 1 $SEED; done
for x in off_a:dep_off off_b:dep_off k25:dep_k25 bubble:dep_bubble; do
  n=${x%%:*}; c=${x#*:}; rm -rf $O/cache_c; cp -r $C0 $O/cache_c
  DEP_PARAM_DUMP=$O/$n/params DEP_GRAD_DUMP=$O/$n/grads DEP_GRAD_STEPS=1,2 DEPN_MEM_OUT=$O/$n/mem \
    run $n $c $O/cache_c 3 $SEED
done
for p in off_b k25 bubble; do
  { echo "== initial parameters, off_a vs $p"; python $K/cmp_params.py $O/off_a/params $O/$p/params
    echo "== gradients, off_a vs $p"; python $K/cmp_grads.py $O/off_a/grads $O/$p/grads; } >> $O/cmp.txt 2>&1
done
rm -rf $O/cache_c $C0 $O/*/params $O/*/grads
export DEPW_DIM=${DEPW_DIM:-6144}
T0=$O/cache_t0; rm -rf $T0; mkdir -p $T0
if ! run twarm_w_dep_bubble w_dep_bubble $T0 2; then
  if grep -qi "out of memory" $O/twarm_w_dep_bubble/run.log; then
    export DEPW_DIM=5120; note "dim 6144 does not fit, widening to 5120 instead"
    rm -rf $T0; mkdir -p $T0; run twarm_w_dep_bubble w_dep_bubble $T0 2
  fi
fi
note "widened dim $DEPW_DIM"
for c in w_dep_off w_dep_k25; do run twarm_$c $c $T0 2; done
for c in w_dep_off w_dep_k25 w_dep_bubble; do
  rm -rf $O/cache_t; cp -r $T0 $O/cache_t
  DEPN_MEM_OUT=$O/time_$c/mem run time_$c $c $O/cache_t 30
done
for c in w_dep_off w_dep_k25 w_dep_bubble; do
  rm -rf $O/cache_t; cp -r $T0 $O/cache_t
  run trace_$c $c $O/cache_t 16 --profiler.enable-profiling --profiler.profile-freq 15 --profiler.profiler-warmup 2 \
    --profiler.profiler-active 1 --profiler.save-traces-folder traces
  t=$(find $O/trace_$c/out -name "rank0_trace.json*" | head -1)
  [ -n "$t" ] && python $K/analyze_trace.py $(dirname $t) > $O/trace_$c/analysis.txt 2>&1
done
rm -rf $O/cache_t $T0
if [ -n "${SHIM:-}" ]; then git -C $W checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; fi
note "dep done dirty=$(git -C $W status --short | wc -l)"; echo done > $O/DONE
