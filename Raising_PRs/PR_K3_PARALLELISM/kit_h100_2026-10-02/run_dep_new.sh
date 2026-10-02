#!/bin/bash
# DEP on 4 x H100 at K3-range levels, on the rebased tree (dep_review1 a93cd48ea on main db050eb3f): the port of
# run_dep_h100_k3range.sh to the Python-config loader (dep_new_local.py reads steps, micro-batches, profiling and
# determinism from DEPN_*). Widened debug text (dim 6144, 17 layers), base tower, full AC, pp4 x vpp4.
# LEVELS="name:seq:cap:ratio:mbs:res[:per16] ...", ratio = the planner's cost ratio measured from a K2.5 trace
# (dep_new_calib.sh). Per level and count: DEP off / K2.5 / bubble warmed 2 steps on one cache, 20 timed steps each on a
# copy, one profiled step (step 10 of 12) each, fill from the three traces; then numerics at the first level and count:
# DEP off twice, K2.5, bubble, seed 42, deterministic, automatic dynamic shapes off, 20 steps on one cache.
M=~/mep; KN=~/kit/kit_h100_2026-10-02; K30=~/kit/kit_h100_2026-09-30; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src
W=$M/w/dep; O=${OUT:-$M/results/dep_new_k3range}; P=$O/progress.txt; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPW_DIM=6144 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "dep $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l) levels: $LEVELS"
run() {  # <name> <config> <cache> <steps>; DEPN_*, DEPR_* and DEPV_* from the caller's environment
  local name=$1 cfg=$2 cache=$3; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && DEPN_STEPS=$4 PYTHONPATH=$KN:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_new_local --config $cfg --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
for lvl in $LEVELS; do
  IFS=: read -r name seq cap ratio mlist res per16 <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio DEPR_RES=$res
  if [ -n "$per16" ]; then export DEPR_IMG_PER16=$per16; else unset DEPR_IMG_PER16; fi
  for mbs in ${mlist//,/ }; do
    export DEPN_MBS=$mbs
    T0=$O/cache_${name}_m$mbs; rm -rf $T0; mkdir -p $T0
    for c in w_dep_off w_dep_k25 w_dep_bubble; do run warm_${name}_m${mbs}_$c $c $T0 2; done
    for c in w_dep_off w_dep_k25 w_dep_bubble; do
      rm -rf $O/cc; cp -r $T0 $O/cc; run time_${name}_m${mbs}_$c $c $O/cc 20; rm -rf $O/time_${name}_m${mbs}_$c/out
    done
    for c in w_dep_off w_dep_k25 w_dep_bubble; do
      rm -rf $O/cc; cp -r $T0 $O/cc
      DEPN_PROFILE=1 run trace_${name}_m${mbs}_$c $c $O/cc 12
    done
    PYTHONPATH=$K30 $V/bin/python $K30/ana_fill.py $O/trace_${name}_m${mbs}_w_dep_{off,k25,bubble}/out \
      > $O/fill_${name}_m${mbs}.txt 2>&1
    note "fill ${name} M$mbs: $(grep -a '^all ranks' $O/fill_${name}_m${mbs}.txt)"
    rm -rf $O/cc $T0
  done
done
if [ "${NUMERICS:-1}" = 0 ]; then note "dep done (no numerics)"; echo done > $O/DONE; exit 0; fi
IFS=: read -r name seq cap ratio mlist res per16 <<< "${LEVELS%% *}"
export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio DEPR_RES=$res DEPN_MBS=${mlist%%,*} DEPN_DET=1 DEPN_STATIC=1
if [ -n "$per16" ]; then export DEPR_IMG_PER16=$per16; else unset DEPR_IMG_PER16; fi
note "numerics at $name M$DEPN_MBS"
C0=$O/cache_num; rm -rf $C0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run nwarm_$c $c $C0 1; done
for x in off_a:w_dep_off off_b:w_dep_off k25:w_dep_k25 bubble:w_dep_bubble; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run num_${x%%:*} ${x#*:} $O/cc 20; rm -rf $O/num_${x%%:*}/out
done
rm -rf $O/cc $C0
note "dep done"
echo done > $O/DONE
