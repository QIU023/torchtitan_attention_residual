#!/bin/bash
# DEP ratio cells for the next H100 session (4 x H100, DEP tree ~/mep/w/dep = a03f74981, ~/mep/venv_src).
# Data: /root/dep_data/t2i1024_k4 (see README: rebuild from the HF shard with build_dataset.py, or rsync).
# Per level (L1 224 px, L2 448 px, L3 1024 px; seq 2048; the debug model widened to dim 6144; pp2 x vpp4 x tp2 x ep2):
# DEP off / K2.5 / bubble warmed 2 steps on one cache, 30 timed steps each (steps 12 to 30), one traced step each;
# then numerics at L2: DEP off twice, K2.5, bubble, seed 42, deterministic, 20 steps, one cache.
M=~/mep; K=~/kit/dep_ratio; V=$M/venv_src; W=$M/w/dep; O=$M/results/dep_ratio; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=${DEPR_DATA:-/root/dep_data/t2i1024_k4} DEPR_SEQ=2048 DEPW_DIM=${DEPW_DIM:-6144}
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <name> <config> <cache> <steps> [args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:~/kit/dep:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc DEPN_MEM_OUT=$D/mem \
    timeout 2400 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_ratio_local --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -o 'vision_dep: .*' | head -1)"
}
declare -A RES=([L1]=224 [L2]=448 [L3]=1024)
for L in L1 L2 L3; do
  export DEPR_RES=${RES[$L]}; T0=$O/cache_$L; rm -rf $T0; mkdir -p $T0
  for c in w_dep_off w_dep_k25 w_dep_bubble; do run warm_${L}_$c $c $T0 2; done
  for c in w_dep_off w_dep_k25 w_dep_bubble; do rm -rf $O/cc; cp -r $T0 $O/cc; run time_${L}_$c $c $O/cc 30; rm -rf $O/time_${L}_$c/out; done
  for c in w_dep_off w_dep_k25 w_dep_bubble; do
    rm -rf $O/cc; cp -r $T0 $O/cc
    run trace_${L}_$c $c $O/cc 16 --profiler.enable-profiling --profiler.profile-freq 15 --profiler.profiler-warmup 2 \
      --profiler.profiler-active 1 --profiler.save-traces-folder traces
    t=$(find $O/trace_${L}_$c/out -name "rank0_trace.json*" | head -1)
    [ -n "$t" ] && $V/bin/python ~/kit/dep/analyze_trace.py $(dirname $t) > $O/trace_${L}_$c/analysis.txt 2>&1
  done
  rm -rf $O/cc $T0
done
export DEPR_RES=448; C0=$O/cache_num; rm -rf $C0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run nwarm_$c $c $C0 1 --debug.seed 42 --debug.deterministic; done
for x in off_a:w_dep_off off_b:w_dep_off k25:w_dep_k25 bubble:w_dep_bubble; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run num_${x%%:*} ${x#*:} $O/cc 20 --debug.seed 42 --debug.deterministic
done
rm -rf $O/cc $C0
note "dep ratio h100 done"; echo done > $O/DONE
