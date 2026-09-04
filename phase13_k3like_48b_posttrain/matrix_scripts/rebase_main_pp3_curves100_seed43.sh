#!/bin/bash
# Seed-to-seed spread of one configuration, on the 32-sample debug set: dp1, pp2 x vp4 and pp8 x vp4
# at seed 43 for 100 steps, against the seed-42 curves. The question is whether pp8's slower
# descent at seed 42 exceeds what one configuration does between seeds.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30 MEASURE_STEPS=100 SEED=43
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun3 CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL" $MX pp3c100s43_pp
echo "PP CURVES100 SEED43 DONE"
