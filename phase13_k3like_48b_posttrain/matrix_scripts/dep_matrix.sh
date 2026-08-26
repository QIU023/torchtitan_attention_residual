#!/bin/bash
# DEP (report sec 5.2.3) on the default configuration: does giving the vision
# tower pipeline stages of its own change what the model trains?
#
# One table, one run, one seed, one flavor. DEP engages only at pp > 1, so dp1
# is this table's disabled side. What each pp cell actually runs is printed by
# the wiring log and recorded next to the losses:
#
#   pp2  two stages, so the tower's requested share of two does not fit and is
#        shortened to one -- the whole tower on one stage, no decoder layer
#   pp4  four stages: two vision (head, tail), two text of twelve layers
#   pp8  eight stages: two vision, six text of four layers
#
# The pp2 row is therefore not a lesser cell but the other DEP form, measured
# against the same seed as the split one.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"
P="--parallelism.pipeline_parallel_degree"
MB="--parallelism.num-pp-microbatches"
# Cold KDA/tilelang compilation runs past the default 300s NCCL watchdog at
# pp4+; that is a compile-cache artifact, not a hung collective.
B="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256 --comm.init-timeout-seconds 3600"

CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB 8
pp4|4|$D 1 $P 4 $MB 8
pp8|8|$D 1 $P 8 $MB 8"

TITAN=/workspace/tt_rebase CFG=kimi_k3_debugmodel BATCH="$B" CELLS="$CELLS" $MX mm_dep
echo "DEP MATRIX DONE"
