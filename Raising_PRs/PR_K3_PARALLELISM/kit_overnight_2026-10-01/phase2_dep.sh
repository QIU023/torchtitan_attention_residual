#!/bin/bash
# 10-01 night, DEP on 8 x 5060, phase 2: pp4 x vpp4 (core's 16 stage split), the widened debug text at DEPW_DIM (1024)
# with the non-widened tower (DEPV_TOWER=base), full AC, at LEVELS="name:seq:cap:ratio:mbs:res[:per16] ..." where ratio
# is the planner's cost ratio (vision_dep.bubble_cost_ratio) from phase 1, res the image side (DEPR_RES). Per level and
# micro-batch count: DEP off / K2.5 / bubble warmed 2 steps on one cache, then 12 steps each on a copy with one profiled
# step (step 10); the planner's line from each log and ana_fill.py on the three traces. 5060 is PCIe without P2P, so
# step times are indicative only.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_overnight_2026-09-29/dep_ratio
KD=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_h100_2026-09-29/dep
KH=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_h100_2026-09-30
V=/workspace/venv_0928; W=${W:-$S/wt_dep_1001}; O=${OUT:-$S/dep_1001/phase2}; P=$S/dep_1001/progress.txt; mkdir -p $O
export DEPR_LAYOUT=pp4vpp4 DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=${DEPW_DIM:-1024}
export DEPV_TOWER=${DEPV_TOWER:-base} DEPR_AC=full
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "phase2 dep $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l) dim $DEPW_DIM tower $DEPV_TOWER levels: $LEVELS"
run() {  # <name> <config> <cache> <steps> [args...]
  local name=$1 cfg=$2 cache=$3 steps=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$K:$KD:. TORCHINDUCTOR_CACHE_DIR=$cache/ic \
    TRITON_CACHE_DIR=$cache/tc DEPN_MEM_OUT=$D/mem timeout 2400 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d \
    --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_ratio_local --config $cfg \
    --training.steps $steps --metrics.log-freq 1 --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
}
for lvl in $LEVELS; do
  IFS=: read -r name seq cap ratio mlist res per16 <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio DEPR_RES=$res
  if [ -n "$per16" ]; then export DEPR_IMG_PER16=$per16; else unset DEPR_IMG_PER16; fi
  for mbs in ${mlist//,/ }; do
    B="--parallelism.num-pp-microbatches $mbs --training.num-tokens-per-train-step $((mbs * seq))"
    T0=$O/cache_${name}_m$mbs; rm -rf $T0; mkdir -p $T0
    for c in w_dep_off w_dep_k25 w_dep_bubble; do run warm_${name}_m${mbs}_$c $c $T0 2 $B; done
    for c in w_dep_off w_dep_k25 w_dep_bubble; do
      rm -rf $O/cc; cp -r $T0 $O/cc
      run trace_${name}_m${mbs}_$c $c $O/cc 12 $B --profiler.enable-profiling --profiler.profile-freq 10 \
        --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
      t=$(find $O/trace_${name}_m${mbs}_$c/out -name "rank0_trace.json*" | head -1)
      [ -n "$t" ] && $V/bin/python $KD/analyze_trace.py $(dirname $t) > $O/trace_${name}_m${mbs}_$c/analysis.txt 2>&1
    done
    $V/bin/python $KH/ana_fill.py $O/trace_${name}_m${mbs}_w_dep_{off,k25,bubble}/out > $O/fill_${name}_m${mbs}.txt 2>&1
    note "fill ${name} M$mbs: $(grep -a '^all ranks' $O/fill_${name}_m${mbs}.txt)"
    rm -rf $O/cc $T0
  done
done
note "phase2 done"; echo done > $O/DONE
