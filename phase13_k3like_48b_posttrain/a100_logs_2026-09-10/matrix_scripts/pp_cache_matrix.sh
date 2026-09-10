#!/bin/bash
# PP with the delta block transport ENGAGED. The previous PP tables ran with it
# off -- the adapter logs a line when it wraps stages and that line is in none
# of those logs -- so they measured plain PP, not this PR's centrepiece.
#
# Every pipelined cell here uses the cache flavor. The three vp cells are run
# both ways so the pair sits side by side; pp2 keeps one naive row as the
# reference for what the transport is compared against.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"

# cache on: the adapter path
TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel_32l BATCH="$B" \
CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB $LESS
pp4|4|$D 1 $P 4 $MB $LESS
pp8|8|$D 1 $P 8 $MB $LESS
pp2_vp2|2|$D 1 $P 2 $L 8 $IL $LESS
pp2_vp4|2|$D 1 $P 2 $L 4 $IL $LESS
pp4_vp2|4|$D 1 $P 4 $L 4 $IL $LESS
pp4_vp4|4|$D 1 $P 4 $L 2 $IL $LESS
pp8_vp2|8|$D 1 $P 8 $L 2 $IL $LESS
pp8_vp4|8|$D 1 $P 8 $L 1 $IL $LESS" $MX cache_pp

# the fallback, for the side-by-side rows
TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel_32l_naive BATCH="$B" \
CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB $LESS
pp2_vp2|2|$D 1 $P 2 $L 8 $IL $LESS
pp4_vp4|4|$D 1 $P 4 $L 2 $IL $LESS
pp8_vp4|8|$D 1 $P 8 $L 1 $IL $LESS" $MX naive_pp
echo "PP CACHE MATRICES DONE"
