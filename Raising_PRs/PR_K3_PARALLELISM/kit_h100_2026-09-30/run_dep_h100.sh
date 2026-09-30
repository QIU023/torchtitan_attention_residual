#!/bin/bash
# DEP on 4 x H100 (09-30 box): the widened debug text (dim 6144, 17 layers) with the released K3 tower (DEPV_TOWER=k3),
# 1008 px images from t2i1024_k4, full AC, in the layout DEPR_LAYOUT (default pp4vpp4: pp4 x vpp4 x tp1 x ep1, core's
# 16 stage split; pp2tp2 is the B200 cell without FSDP). LEVELS="name:seq:cap:ratio:mbs[:per16] ..." from calib.sh's
# levels.txt, ratio = the planner's cost ratio for that split (vision_dep.bubble_cost_ratio), mbs a comma list of
# micro-batch counts, per16 (optional) = DEPR_IMG_PER16, how many of every 16 samples carry images.
# Per level and count: DEP off / K2.5 / bubble warmed 2 steps on one cache, 20 timed steps each on a copy, one profiled
# step (step 10) each; then numerics at the first level and its first count: DEP off twice, K2.5, bubble, seed 42,
# deterministic, 20 steps on one cache. Results in ~/mep/results/dep_h100_<layout>; pull them as they land.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src; W=$M/w/dep
export DEPR_LAYOUT=${DEPR_LAYOUT:-pp4vpp4}; O=$M/results/dep_h100_$DEPR_LAYOUT; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full NCCL_NVLS_ENABLE=0
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "dep $(git -C $W rev-parse --short HEAD) layout $DEPR_LAYOUT torch $($V/bin/python -c 'import torch; print(torch.__version__)') levels: $LEVELS"
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
  IFS=: read -r name seq cap ratio mlist per16 <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio
  if [ -n "$per16" ]; then export DEPR_IMG_PER16=$per16; else unset DEPR_IMG_PER16; fi
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
    $V/bin/python ~/kit/kit_h100_2026-09-30/ana_fill.py $O/trace_${name}_m${mbs}_w_dep_{off,k25,bubble}/out \
      > $O/fill_${name}_m${mbs}.txt 2>&1
    note "fill ${name} M$mbs: $(grep -a '^all ranks' $O/fill_${name}_m${mbs}.txt)"
    rm -rf $O/cc $T0
  done
done
IFS=: read -r name seq cap ratio mlist per16 <<< "${LEVELS%% *}"
export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio; mbs=${mlist%%,*}
if [ -n "$per16" ]; then export DEPR_IMG_PER16=$per16; else unset DEPR_IMG_PER16; fi
note "numerics at $name M$mbs"
B="--parallelism.num-pp-microbatches $mbs --training.num-tokens-per-train-step $((mbs * seq))"
C0=$O/cache_num; rm -rf $C0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run nwarm_$c $c $C0 1 $B --debug.seed 42 --debug.deterministic; done
for x in off_a:w_dep_off off_b:w_dep_off k25:w_dep_k25 bubble:w_dep_bubble; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run num_${x%%:*} ${x#*:} $O/cc 20 $B --debug.seed 42 --debug.deterministic
  rm -rf $O/num_${x%%:*}/out
done
rm -rf $O/cc $C0
note "dep h100 done"; echo done > $O/DONE
