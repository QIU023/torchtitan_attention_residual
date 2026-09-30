#!/bin/bash
# DEP on 4 x H100 (09-30 box): the widened debug text (dim 6144, 17 layers) with the released K3 tower (DEPV_TOWER=k3),
# pp2 x vpp4 x tp2 x ep2, 1008 px images from t2i1024_k4. LEVELS="name:seq:cap:ratio:mbs ..." from calib.sh's levels.txt,
# ratio = measured encode forward / text stage forward at that seq (vision_dep.bubble_cost_ratio), mbs a comma list of
# micro-batch counts (default 4). Per level and count: DEP off / K2.5 / bubble warmed 2 steps on one cache, 20 timed steps
# each on a copy, one profiled step (step 10) each; then numerics at the first level, M4: DEP off twice, K2.5, bubble, seed 42,
# deterministic, 20 steps on one cache. Results in ~/mep/results/dep_h100; pull them as they land.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src; W=$M/w/dep
O=$M/results/dep_h100; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full NCCL_NVLS_ENABLE=0
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "dep $(git -C $W rev-parse --short HEAD) torch $($V/bin/python -c 'import torch; print(torch.__version__)') levels: $LEVELS"
run() {  # <name> <config> <cache> <steps> [args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc DEPN_MEM_OUT=$D/mem \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_ratio_local --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
for lvl in $LEVELS; do
  IFS=: read -r name seq cap ratio mlist <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio
  for mbs in ${mlist//,/ }; do
    B="--parallelism.num-pp-microbatches $mbs --training.num-tokens-per-train-step $((mbs * seq))"
    T0=$O/cache_${name}_m$mbs; rm -rf $T0; mkdir -p $T0
    for c in w_dep_off w_dep_k25 w_dep_bubble; do run warm_${name}_m${mbs}_$c $c $T0 2 $B; done
    for c in w_dep_off w_dep_k25 w_dep_bubble; do
      rm -rf $O/cc; cp -r $T0 $O/cc; run time_${name}_m${mbs}_$c $c $O/cc 20 $B; rm -rf $O/time_${name}_m${mbs}_$c/out
    done
    for c in w_dep_off w_dep_k25 w_dep_bubble; do
      rm -rf $O/cc; cp -r $T0 $O/cc
      run trace_${name}_m${mbs}_$c $c $O/cc 12 $B --profiler.enable-profiling --profiler.profile-freq 10 \
        --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
      t=$(find $O/trace_${name}_m${mbs}_$c/out -name "rank0_trace.json*" | head -1)
      [ -n "$t" ] && $V/bin/python $KD/analyze_trace.py $(dirname $t) > $O/trace_${name}_m${mbs}_$c/analysis.txt 2>&1
    done
    rm -rf $O/cc $T0
  done
done
IFS=: read -r name seq cap ratio mlist <<< "${LEVELS%% *}"
export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio
B="--parallelism.num-pp-microbatches 4 --training.num-tokens-per-train-step $((4 * seq))"
C0=$O/cache_num; rm -rf $C0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run nwarm_$c $c $C0 1 $B --debug.seed 42 --debug.deterministic; done
for x in off_a:w_dep_off off_b:w_dep_off k25:w_dep_k25 bubble:w_dep_bubble; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run num_${x%%:*} ${x#*:} $O/cc 20 $B --debug.seed 42 --debug.deterministic
  rm -rf $O/num_${x%%:*}/out
done
rm -rf $O/cc $C0
note "dep h100 done"; echo done > $O/DONE
