#!/bin/bash
# Rerun one level's three trace cells on a quiet box (09-30: the r10 M16 traces overlapped a CPU probe of mine and are void).
# Env as run_dep_h100.sh: DEPR_LAYOUT (default pp4vpp4); LEVEL="name:seq:cap:ratio:mbs:per16" (one M); TAG (default trace2).
# Warms DEP off / K2.5 / bubble 2 steps on one fresh cache, then traces each on a copy (step 10), analyze_trace and ana_fill.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src; W=$M/w/dep; TAG=${TAG:-trace2}
export DEPR_LAYOUT=${DEPR_LAYOUT:-pp4vpp4}; O=$M/results/dep_h100_$DEPR_LAYOUT; P=$O/progress.txt
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full NCCL_NVLS_ENABLE=0
IFS=: read -r name seq cap ratio mbs per16 <<< "$LEVEL"
export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio
if [ -n "$per16" ]; then export DEPR_IMG_PER16=$per16; else unset DEPR_IMG_PER16; fi
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <name> <config> <cache> <steps> [args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_ratio_local --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint; note "$name rc=$rc"
}
B="--parallelism.num-pp-microbatches $mbs --training.num-tokens-per-train-step $((mbs * seq))"
T0=$O/cache_${TAG}_${name}_m$mbs; rm -rf $T0; mkdir -p $T0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run ${TAG}warm_${name}_m${mbs}_$c $c $T0 2 $B; done
for c in w_dep_off w_dep_k25 w_dep_bubble; do
  rm -rf $O/cc2; cp -r $T0 $O/cc2
  run ${TAG}_${name}_m${mbs}_$c $c $O/cc2 12 $B --profiler.enable-profiling --profiler.profile-freq 10 \
    --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
  t=$(find $O/${TAG}_${name}_m${mbs}_$c/out -name "rank0_trace.json*" | head -1)
  [ -n "$t" ] && $V/bin/python $KD/analyze_trace.py $(dirname $t) > $O/${TAG}_${name}_m${mbs}_$c/analysis.txt 2>&1
done
$V/bin/python ~/kit/kit_h100_2026-09-30/ana_fill.py $O/${TAG}_${name}_m${mbs}_w_dep_{off,k25,bubble}/out > $O/fill_${TAG}_${name}_m${mbs}.txt 2>&1
note "fill ${TAG} ${name} M$mbs: $(grep -a '^all ranks' $O/fill_${TAG}_${name}_m${mbs}.txt)"
rm -rf $O/cc2 $T0
