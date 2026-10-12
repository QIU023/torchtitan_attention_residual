#!/bin/bash
# Step-1 gradient dumps, AC none, one GPU each, on copies of the numerics cache: main twice, main with eager aggregation, c1, c2.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; K=$(cd "$(dirname "$0")" && pwd)
A=$S/m4656_5033; O=$A/grads; mkdir -p $O; PR=$O/progress.txt
declare -A TREE=([main]=$S/wt_mainref [c1]=$S/wt_4656c1 [c2]=$S/wt_4656rb)
. /workspace/venv_1006/bin/activate
g=0
for c in "main main" "main main2" "main maineager PROBE_EAGER_AR=1" "c1 c1" "c2 c2"; do
  set -- $c; t=$1 name=$2; shift 2; D=$O/$name; rm -rf $D; mkdir -p $D; rm -rf $A/gcache_$g; cp -r $A/cache0 $A/gcache_$g
  ( cd ${TREE[$t]} && env "$@" PROBE_AC=none PROBE_STEPS=1 MODULE=attnres_probe_1012 CONFIG=ac_cell GRAD_DUMP=$O/$name/grads \
    PYTHONPATH=$K/local/grad_dump:$K:${TREE[$t]} TORCHINDUCTOR_CACHE_DIR=$A/gcache_$g/ic TRITON_CACHE_DIR=$A/gcache_$g/tc \
    CUDA_VISIBLE_DEVICES=$g NGPU=1 LOG_RANK=0 timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1; echo "$(date +%H:%M:%S) $name rc=$?" >> $PR; rm -rf $D/out ) &
  g=$((g + 1))
done
wait; rm -rf $A/gcache_*
for name in main2 maineager c1 c2; do echo "== main vs $name"; python $K/cmp_grad_dump.py $O/main/grads $O/$name/grads; done > $O/compare.txt 2>&1
echo done > $O/DONE
