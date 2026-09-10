#!/bin/bash
# The fallback counterpart for every virtual-stage cell. Only vp cells differ:
# plain 1F1B gives each rank one stage, so there is no rank-shared stack to
# reuse and both flavors take the same path. dp1 is here as this matrix's own
# baseline, not as a comparison row.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel_32l_naive \
BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256" \
CELLS="dp1|1|$D
pp2_vp2|2|$D $P 2 $L 8 $IL $LESS
pp2_vp4|2|$D $P 2 $L 4 $IL $LESS
pp4_vp2|4|$D $P 4 $L 4 $IL $LESS
pp4_vp4|4|$D $P 4 $L 2 $IL $LESS
pp8_vp2|8|$D $P 8 $L 2 $IL $LESS
pp8_vp4|8|$D $P 8 $L 1 $IL $LESS" $MX naive_pp
echo "NAIVE GROUP DONE"
