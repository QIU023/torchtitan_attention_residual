#!/bin/bash
# The MoonEP test-plan cells on 4 x H100. Usage: bash run_cells.sh <cell>
# Cells follow phase13_k3like_48b_posttrain/MOONEP_TEST_PLAN_2026-09-16.md.
source /workspace/kit/env.sh
cd "$TITAN"
CELL=${1:-list}
COMMON="--debug.seed 42 --debug.deterministic --training.dataset c4_test"
case "$CELL" in
  2)  # ep4 x fsdp4, standard dispatcher, 10 steps: the reference row
      NAME=cell2_standard_ep4; STEPS=10
      ARGS="--parallelism.expert_parallel_degree 4 --parallelism.data_parallel_shard_degree 4" ;;
  3)  # ep4 x fsdp4, MoonEP with B = E/R = 8, 10 steps
      NAME=cell3_moonep_ep4; STEPS=10
      ARGS="--parallelism.expert_parallel_degree 4 --parallelism.data_parallel_shard_degree 4 --model.flavor debugmodel_moonep" ;;
  7)  # ep2 x fsdp2 on two GPUs, 3 steps: the smallest cell, the CI story
      NAME=cell7_moonep_ep2; STEPS=3
      ARGS="--parallelism.expert_parallel_degree 2 --parallelism.data_parallel_shard_degree 2 --model.flavor debugmodel_moonep" ;;
  *)  echo "cells: 2 (standard ref), 3 (moonep ep4), 7 (moonep ep2)"; exit 0 ;;
esac
NP=${NP:-4}
echo "### $NAME NP=$NP steps=$STEPS"
timeout 3600 torchrun --nproc_per_node=$NP --master_port=$((42000+CELL)) \
  -m torchtitan.train $COMMON $ARGS --training.steps $STEPS \
  > /workspace/results/${NAME}.log 2>&1
echo "rc=$?"
grep -E "step: *[0-9]+ .*loss" /workspace/results/${NAME}.log | tail -3
echo "### CELL_${CELL}_DONE"
