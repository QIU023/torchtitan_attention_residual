#!/bin/bash
# PP sanity on the rebased tree (pp_review3 on post-EP main): 30-layer debugmodel,
# cache on (dp1 + three vp cells) then the naive transport for the side-by-side row.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL $LESS
pp4_vp4|4|$D 1 $P 4 $L 2 $IL $LESS
pp8_vp4|8|$D 1 $P 8 $L 1 $IL $LESS" $MX main30_pp
TITAN=/tmp/wt_pprun CFG=kimi_k3_debugmodel_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL $LESS" $MX main30_ppn
echo "PP MAIN30 MATRICES DONE"
