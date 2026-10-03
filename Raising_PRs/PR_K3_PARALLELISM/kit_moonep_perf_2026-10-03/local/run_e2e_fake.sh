#!/bin/bash
# End-to-end Kimi K3 debug cells on 4 x 5060 against the async fake MoonEP: old head (OLD) and new head (NEW),
# 10 steps, deterministic, every cell on a copy of one warm compile cache per tree (warmed by a 1-step run of
# each cell). Probe recipes: moonep_probe_1003.py.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; P=$S/moonep_perf_1003
OLD=${OLD:-$S/wt_moonep_old_5e45}; NEW=${NEW:-$S/wt_moonep_clean}; FK=${FK:-$S/fake_moonep_async}
O=$P/e2e_${TAG:-1003}; rm -rf $O; mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. /workspace/venv_0928/bin/activate
cell() {  # <tree dir> <name> <config> <cache> <steps>
  local T=$1 name=$2 cfg=$3 cache=$4 D=$O/$2; rm -rf $D; mkdir -p $D
  ( cd $T && PROBE_STEPS=$5 PROBE_DET=1 PYTHONPATH=$P:$FK:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_probe_1003 CONFIG=$cfg \
    timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
for side in old new; do
  T=$OLD; cells="standard moonep moonep_full"; [ $side = new ] && T=$NEW && cells="standard moonep moonep_full moonep_shared_stream standard_shared_stream"
  note "$side tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
  C0=$O/cache_$side; rm -rf $C0; mkdir -p $C0
  for c in $cells; do cell $T warm_${side}_$c ${c}_cell $C0 1; done
  for c in $cells; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell $T ${side}_$c ${c}_cell $O/cache_c 10; done
  rm -rf $O/cache_c $C0
done
note "done"
echo done > $O/DONE
