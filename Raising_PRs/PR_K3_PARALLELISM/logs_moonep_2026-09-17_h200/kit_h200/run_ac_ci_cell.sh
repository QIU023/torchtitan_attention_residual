#!/bin/bash
# The b200 CI cell on 4 GPUs (torchtitan_recipes.tests.b200:kimi_k3_debugmodel_mm: SPMD typechecking, fsdp2 x tp2 (SP) x ep2),
# main vs PR, as the recipe runs (selective AC) and with activation checkpointing off; one shared warm cache. Waits for the dp1 table.
source /workspace/kit/h200/env.sh
until O=$(ls -d /workspace/ac_* | tail -1) && grep -q "^DONE" $O/results.txt 2>/dev/null; do sleep 20; done
OUT=/workspace/acci_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; echo "main=$(git -C /workspace/tt_main rev-parse --short HEAD) pr=$(git -C /workspace/tt_ac rev-parse --short HEAD)" > $R
B="--debug.seed 42 --debug.deterministic --metrics.log_freq 1"
SHARED=$OUT/cache_shared
run(){ local nm=$1 tree=$2 cache=$3 steps=$4; shift 4; local d=$OUT/$nm; mkdir -p $d
  ( cd $tree && CUDA_VISIBLE_DEVICES=0,1,2,3 TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor PYTHONPATH=$tree timeout 2400 torchrun --nproc_per_node=4 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module torchtitan_recipes.tests.b200 --config kimi_k3_debugmodel_mm --training.steps $steps $B --dump-folder $d "$@" > $d/log.txt 2>&1 ); local rc=$?
  sed "s/\x1b\[[0-9;]*m//g" $d/log.txt | grep -oE "step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+ +memory: *[0-9.]+GiB\([0-9.]+%\) +tps: *[0-9,]+" | tr -s " " > $d/steps.txt
  echo "$nm rc=$rc steps=$(wc -l < $d/steps.txt) | $(grep -E "^step: 1 " $d/steps.txt | head -1) | $(grep -E "^step: 10 " $d/steps.txt | head -1)" >> $R; tail -1 $R
  grep -a -m1 "Error\|error:" $d/log.txt | grep -v lspci | cut -c1-160 >> $R; }
declare -A TREE=([main]=/workspace/tt_main [pr]=/workspace/tt_ac)
for t in main pr; do run warm_${t}_recipe ${TREE[$t]} $SHARED 1; run warm_${t}_none ${TREE[$t]} $SHARED 1 activation-checkpoint:none; done
for t in main pr; do run ${t}_recipe ${TREE[$t]} $SHARED 10; run ${t}_none ${TREE[$t]} $SHARED 10 activation-checkpoint:none; done
echo "DONE $OUT" >> $R
