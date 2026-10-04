#!/bin/bash
# PR 4656 list-only (7dea4cbaf) against main 838e6962e on the B200 suite's 8-GPU cell (FSDP 2, TP 2, EP 2, PP 2, VPP 4),
# 8 x 5060, torch 2.15.0.dev20261003 (venv_1003b), with PR-5025's stage.py line applied to both trees as a local patch
# (never committed). Seed 42, deterministic, 10 steps (attnres_probe_1004.pp_cell); both trees warm one cache with a 1-step run, then every measured
# run starts from a copy of it; main runs twice for the noise floor.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; A=$S/attnres_list_1004
MAIN=$S/attnres_main_tree; LIST=$S/wt_attnres_list; V=/workspace/venv_1003b; PATCH=$S/pr5025_stage.patch
O=$A/runs_pp5025; rm -rf $O; mkdir -p $O; PR=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $PR; }
. $V/bin/activate
for T in $MAIN $LIST; do
  if grep -q set_manual_backward_finalization $T/torchtitan/models/kimi_k3/pipeline_parallel/stage.py; then note "patch already in $T"
  else ( cd $T && git apply $PATCH ) && note "patched $T" || { note "patch FAILED $T"; exit 1; }; fi
done
note "list $(git -C $LIST rev-parse --short HEAD) dirty=$(git -C $LIST status --short | tr '\n' ' ') torch $(python -c 'import torch;print(torch.__version__)')"
run() {  # <tree dir> <name> <cache> <steps>
  local T=$1 name=$2 cache=$3 steps=$4; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $T && MODULE=attnres_probe_1004 CONFIG=pp_cell PROBE_STEPS=$steps \
    PYTHONPATH=$A:$T TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
C=$O/cache; mkdir -p $C
run $MAIN warm_main $C 1
run $LIST warm_list $C 1
for cell in main:$MAIN main2:$MAIN list:$LIST; do
  rm -rf $O/cc; cp -r $C $O/cc; run ${cell#*:} pp_${cell%%:*} $O/cc 10
done
rm -rf $O/cc $C
python /workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_4656_list_2026-10-04/local/cmp_pp_steps.py $O/pp_main/run.log $O/pp_main2/run.log $O/pp_list/run.log > $O/cmp.txt 2>&1
note "compare: $(tail -2 $O/cmp.txt | tr '\n' ' ')"
for T in $MAIN $LIST; do ( cd $T && git apply -R $PATCH ) && note "unpatched $T"; done
note "list dirty after=$(git -C $LIST status --short | wc -l)"
echo done > $O/DONE
