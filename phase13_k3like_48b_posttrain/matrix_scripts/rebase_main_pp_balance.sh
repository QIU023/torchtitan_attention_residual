#!/bin/bash
# pp_balance on pp_balance_review1: pp2 x vp4 with rank 0 parking its saved activations on
# rank 1 through the Mooncake Transfer Engine, once as designed and once with
# K3_PPBAL_KEEP_LOCAL=1 (every transfer runs, the local storage stays: the bitwise check).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_ppbalrun CFG=kimi_k3_debugmodel_pp_balance BATCH="$B" \
CELLS="pp2_vp4_balance|2|$D 1 $P 2 $L 4 $IL" $MX ppbal_pp
K3_PPBAL_KEEP_LOCAL=1 TITAN=/tmp/wt_ppbalrun CFG=kimi_k3_debugmodel_pp_balance BATCH="$B" \
CELLS="pp2_vp4_balance_keeplocal|2|$D 1 $P 2 $L 4 $IL" $MX ppbal_pp_keeplocal
echo "PPBAL MATRIX DONE"
