#!/bin/bash
# Apples-to-apples control against the old tree's text arm.
# The 32-layer flavor drops 12.48 -> 3.59 in 10 steps; the old tree's text arm
# dropped 7.70 -> 4.87. Comparing step-10 gaps across those two curves compares
# amplification, not the parallelism. This runs dp1 vs pp2 on the flatter
# 24-layer text flavor so the step-2 relative gap is directly comparable to the
# old tree's text/pp2 (7.19e-4).
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"
P="--parallelism.pipeline_parallel_degree"
MB="--parallelism.num-pp-microbatches 8"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"

TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel_text \
BATCH="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256" \
CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB $LESS
pp4|4|$D 1 $P 4 $MB $LESS" $MX flat24

echo "FLAT CONTROL DONE"
