#!/bin/bash
# PR 4656, list-only (252ee7abd on main 838e6962e) against main itself, 8 x 5060, torch 2.15.0.dev20261003 (venv_1003b).
# Every comparison runs on copies of one cache warmed by a 1-step run of each configuration.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; A=$S/attnres_list_1004
MAIN=$S/attnres_main_tree; LIST=$S/wt_attnres_list; V=/workspace/venv_1003b
O=$A/runs; rm -rf $O; mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. $V/bin/activate
run() {  # <tree dir> <name> <gpu list> <cache> <env...> -- runs one cell in the foreground
  local T=$1 name=$2 gpus=$3 cache=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$A:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=$gpus \
    NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) timeout 2400 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
note "main $(git -C $LIST rev-parse --short HEAD~1) list $(git -C $LIST rev-parse --short HEAD) dirty=$(git -C $LIST status --short | wc -l) torch $(python -c 'import torch;print(torch.__version__)')"

# 1. numerics, default seq, AC none/selective/full/region, 10 steps, one GPU each
C0=$O/cache0; mkdir -p $C0
for t in main list; do T=$MAIN; [ $t = list ] && T=$LIST
  for ac in none selective full region; do run $T warm_${t}_$ac 0 $C0 MODULE=attnres_probe_1004 CONFIG=ac_cell PROBE_AC=$ac PROBE_STEPS=1; done
done
g=0
for t in main list; do T=$MAIN; [ $t = list ] && T=$LIST
  for ac in none selective full region; do
    cp -r $C0 $O/cache_$g
    run $T num_${t}_$ac $g $O/cache_$g MODULE=attnres_probe_1004 CONFIG=ac_cell PROBE_AC=$ac PROBE_STEPS=10 &
    g=$((g + 1))
  done
done
wait; rm -rf $O/cache_[0-9]*

# 2. memory, seq 8192, AC none/selective/full, 3 steps
C1=$O/cache1; mkdir -p $C1
for t in main list; do T=$MAIN; [ $t = list ] && T=$LIST
  for ac in none selective full; do run $T warmmem_${t}_$ac 0 $C1 MODULE=attnres_probe_1004 CONFIG=ac_cell PROBE_AC=$ac PROBE_SEQ=8192 PROBE_STEPS=1; done
done
g=0
for t in main list; do T=$MAIN; [ $t = list ] && T=$LIST
  for ac in none selective full; do
    cp -r $C1 $O/cache_$g
    run $T mem_${t}_$ac $g $O/cache_$g MODULE=attnres_probe_1004 CONFIG=ac_cell PROBE_AC=$ac PROBE_SEQ=8192 PROBE_STEPS=3 &
    g=$((g + 1))
  done
done
wait; rm -rf $O/cache_[0-9]*

# 3. the B200 suite's 8-GPU cell (FSDP 2, TP 2, EP 2, PP 2, VPP 4), main then list, on one warmed cache
C2=$O/cache2; mkdir -p $C2
run $MAIN warm_pp_main 0,1,2,3,4,5,6,7 $C2 MODULE=torchtitan_recipes.tests.suites.b200 CONFIG=kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4
for t in main list; do T=$MAIN; [ $t = list ] && T=$LIST
  cp -r $C2 $O/cache_pp; run $T pp_$t 0,1,2,3,4,5,6,7 $O/cache_pp MODULE=torchtitan_recipes.tests.suites.b200 CONFIG=kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4
  rm -rf $O/cache_pp
done
rm -rf $C0 $C1 $C2
note "done"
echo done > $O/DONE
