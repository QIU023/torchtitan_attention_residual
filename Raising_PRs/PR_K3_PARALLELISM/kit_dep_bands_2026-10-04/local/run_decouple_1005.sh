#!/bin/bash
# 6cda7daca (DEP stage composed with the AttnRes stage) against 8518f473f on 8 x 5060, torch 0928: the B200 suite's
# DEP cell (FSDP 2, TP 2, EP 2, PP 2, VPP 4) for 10 deterministic steps; both trees warm one cache, each measured run
# starts from a copy of it.
OLD=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_dep_1004; NEW=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_dep_1005; O=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dep_1005/runs; rm -rf $O; mkdir -p $O; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_0928/bin/activate
for t in $OLD $NEW; do note "$t $(git -C $t rev-parse --short HEAD) dirty=$(git -C $t status --short | wc -l)"; done
run() {  # <tree> <name> <cache> <steps>
  local T=$1 name=$2 cache=$3 D=$O/$2; rm -rf $D; mkdir -p $D
  ( cd $T && PROBE_STEPS=$4 PYTHONPATH=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/dep_1005:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc NGPU=8 \
    LOG_RANK=0,1,2,3,4,5,6,7 MODULE=dep_probe_1004 CONFIG=dep_cell timeout 2400 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) | $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: [^\[]*' | head -1)"
}
C0=$O/cache0; mkdir -p $C0
run $OLD warm_old $C0 1
run $NEW warm_new $C0 1
for x in old:$OLD new:$NEW; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; run ${x#*:} dep_${x%%:*} $O/cache_c 10; done
rm -rf $O/cache_c $C0
python3 /workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_4656_list_2026-10-04/local/cmp_pp_steps.py $O/dep_old/run.log $O/dep_new/run.log > $O/cmp.txt 2>&1
note "compare: $(tail -1 $O/cmp.txt)"
note "done"; echo done > $O/DONE
