#!/bin/bash
# PR 4656 rebased onto main b663c1673 (#5033 compiles the aggregation) against main, 8 x 5060, torch 2.15.0.dev20261005 (venv_1006).
# Trees: main (b663c1673), c1 (list carrier, e34e0ed53), c2 (c1 + weighted sum as a reduction, ba8f071d1).
# Numerics cells run on copies of one cache warmed by a 10-step run of each configuration; memory cells run cold, 3 steps.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
A=$S/m4656_5033; O=$A/runs; mkdir -p $O; PR=$O/progress.txt
declare -A TREE=([main]=$S/wt_mainref [c1]=$S/wt_4656c1 [c2]=$S/wt_4656rb)
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_1006/bin/activate
run() {  # <tree> <name> <gpu list> <cache> <env...>
  local T=${TREE[$1]} name=$2 gpus=$3 cache=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$K:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=$gpus \
    NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 3000 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
note "start: main $(git -C ${TREE[main]} rev-parse --short HEAD) c1 $(git -C ${TREE[c1]} rev-parse --short HEAD) c2 $(git -C ${TREE[c2]} rev-parse --short HEAD) torch $(python -c 'import torch;print(torch.__version__)')"
for t in main c1 c2; do note "$t dirty=$(git -C ${TREE[$t]} status --short | wc -l)"; done

# 1. numerics, default seq, 20 steps, one GPU each: AC none/selective/full/region per tree; main twice and main with eager aggregation (AC none)
C0=$A/cache0; mkdir -p $C0
for t in main c1 c2; do for ac in none selective full region; do
  run $t warm_${t}_$ac 0 $C0 MODULE=attnres_probe_1012 CONFIG=ac_cell PROBE_AC=$ac PROBE_STEPS=10
done; done
run main warm_maineager_none 0 $C0 MODULE=attnres_probe_1012 CONFIG=ac_cell PROBE_AC=none PROBE_STEPS=10 PROBE_EAGER_AR=1
note "cache0 $(du -sh $C0 | cut -f1)"
cells=()
for t in main c1 c2; do for ac in none selective full region; do cells+=("$t num_${t}_$ac PROBE_AC=$ac"); done; done
cells+=("main num_main2_none PROBE_AC=none" "main num_maineager_none PROBE_AC=none PROBE_EAGER_AR=1")
g=0
for c in "${cells[@]}"; do
  set -- $c; t=$1 name=$2; shift 2
  rm -rf $A/cache_$g; cp -r $C0 $A/cache_$g
  run $t $name $g $A/cache_$g MODULE=attnres_probe_1012 CONFIG=ac_cell PROBE_STEPS=20 "$@" &
  g=$((g + 1)); if [ $g -eq 8 ]; then wait; rm -rf $A/cache_[0-9]*; g=0; fi
done
wait; rm -rf $A/cache_[0-9]*

# 2. memory, seq 8192 and 16384, AC none/selective/full/region, 3 steps, cold per-cell caches
cells=()
for seq in 8192 16384; do for t in main c1 c2; do for ac in none selective full region; do cells+=("$t mem${seq}_${t}_$ac PROBE_AC=$ac PROBE_SEQ=$seq"); done; done; done
g=0
for c in "${cells[@]}"; do
  set -- $c; t=$1 name=$2; shift 2
  run $t $name $g $A/cache_m$g MODULE=attnres_probe_1012 CONFIG=ac_cell PROBE_STEPS=3 "$@" &
  g=$((g + 1)); if [ $g -eq 8 ]; then wait; rm -rf $A/cache_m*; g=0; fi
done
wait; rm -rf $A/cache_m*

# 3. the B200 suite's 8-GPU PP cell, 20 steps, warmed by a 10-step main run; main twice, c1, c2 on copies
C2=$A/cache2; mkdir -p $C2
run main warm_pp_main 0,1,2,3,4,5,6,7 $C2 MODULE=attnres_probe_1012 CONFIG=pp_cell PROBE_STEPS=10
for t in main c1 c2 main2; do tt=$t; [ $t = main2 ] && tt=main
  rm -rf $A/cache_pp; cp -r $C2 $A/cache_pp; run $tt pp_$t 0,1,2,3,4,5,6,7 $A/cache_pp MODULE=attnres_probe_1012 CONFIG=pp_cell PROBE_STEPS=20
done
rm -rf $A/cache_pp
note "done"
echo done > $O/DONE
