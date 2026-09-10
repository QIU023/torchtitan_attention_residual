#!/bin/bash
# DEP and the delta block transport, together, as evidence rather than a smoke
# test. The probe showed the four combinations RUN; this asks whether they
# compute the same thing.
#
# Every pipelined cell here has the tower on its own stages AND the transport
# engaged -- a combination that could not run at all before 0242649e8. The
# earlier DEP table is not a substitute: none of its cells wrapped a stage.
#
# Stage arithmetic (DEP takes 2 stages for the tower out of the total, the rest
# are text stages, total % pp == 0):
#   pp2 x vp2  layers-per-stage 6 -> 4 total, 2 text
#   pp2 x vp4  layers-per-stage 3 -> 8 total, 6 text
#   pp4 x vp2  layers-per-stage 3 -> 8 total, 6 text
#   pp8 x vp4  layers-per-stage 1 -> 24 total, 22 text (uneven, which is why the
#              layout has to be read from the split rather than assumed)
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"
P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
B="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256 --comm.init-timeout-seconds 3600"

CELLS="dp1|1|$D 1
pp2_vp2|2|$D 1 $P 2 $L 6 $IL $LESS
pp2_vp4|2|$D 1 $P 2 $L 3 $IL $LESS
pp4_vp2|4|$D 1 $P 4 $L 3 $IL $LESS
pp8_vp4|8|$D 1 $P 8 $L 1 $IL $LESS"

TITAN=/workspace/tt_depfix CFG=kimi_k3_debugmodel BATCH="$B" CELLS="$CELLS" $MX dep_cache
echo "DEP+CACHE MATRIX DONE"
