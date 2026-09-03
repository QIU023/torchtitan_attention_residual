#!/bin/bash
# The pp x vp cells the way the 30-layer flavor was designed: first/last_stage_less_layers
# at their default 1, so the embedding and the head count as units (32 units) and every
# split is uneven. The earlier pass wrongly forced them to 0 (a 32-layer-flavor habit).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp4_vp4|4|$D 1 $P 4 $L 2 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL" $MX main30_pp_fix
TITAN=/tmp/wt_pprun CFG=kimi_k3_debugmodel_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL" $MX main30_ppn_fix
echo "PP MAIN30 FIX DONE"
